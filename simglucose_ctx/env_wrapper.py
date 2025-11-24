from __future__ import annotations

from collections import deque
from typing import Optional
import logging
import pandas as pd

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import Action

from .context import ContextStream
from .sim_hook import apply_vmx_multiplier, apply_p2u_multiplier, apply_vm0_multiplier

logger = logging.getLogger(__name__)


class ContextAwareT1DSimEnv(T1DSimEnv):
    """T1DSimEnv wrapper that contextually modulates Vmx/p2u/Vm0 per mini_step.

    Usage: provide `context_stream` during construction. The wrapper stores
    baseline values of Vmx, p2u and Vm0 and recalculates them at each mini_step via
    m(t) from the ContextStream. If `context_stream=None`, behavior remains identical
    to the original T1DSimEnv.
    """

    def __init__(
        self,
        patient,
        sensor,
        pump,
        scenario,
        *,
        context_stream: Optional[ContextStream] = None,
        modulate_vmx: bool = True,
        modulate_p2u: bool = False,
        modulate_vm0: bool = False,
        max_log_size: int = 300,
    ):
        """Initialize context-aware simulation environment.

        Parameters
        ----------
        patient, sensor, pump, scenario : SimGlucose components
        context_stream : ContextStream, optional
            Context stream providing m(t). If None, no modulation.
        modulate_vmx, modulate_p2u, modulate_vm0 : bool
            Flags to enable/disable modulation of specific parameters.
        max_log_size : int, default=300
            Maximum number of log entries to retain (~25 hours at 5min intervals).
            Prevents unbounded memory growth in long simulations.
        """
        super().__init__(patient, sensor, pump, scenario)
        self._ctx = context_stream
        self._mod_vmx = bool(modulate_vmx)
        self._mod_p2u = bool(modulate_p2u)
        self._mod_vm0 = bool(modulate_vm0)
        self._max_log_size = int(max_log_size)

        # store baseline values for safe reset
        self._vmx_base = float(self.patient._params.Vmx)
        self._p2u_base = float(self.patient._params.p2u)
        self._vm0_base = float(self.patient._params.Vm0)

        # Hysteresis state (owned by env, not patient)
        self._ctx_hyst = {"exercise": 0, "stress": 0}

        # Current step context (for info dict)
        self._current_m = 1.0
        self._current_hr = None
        self._current_eda = None

        # Logging buffers (using deque for O(1) append/eviction)
        self._log_time = deque(maxlen=self._max_log_size)
        self._log_m = deque(maxlen=self._max_log_size)
        self._log_vmx = deque(maxlen=self._max_log_size)
        self._log_p2u = deque(maxlen=self._max_log_size)
        self._log_vm0 = deque(maxlen=self._max_log_size)

    def mini_step(self, action: Action):
        # 0) Contextual modulation before state update
        now_ts = self.time
        self._current_m = 1.0  # default
        self._current_hr = None
        self._current_eda = None

        if self._ctx is not None:
            # calculate one m(t) for consistent logging and reuse
            ts = pd.Timestamp(now_ts)
            mval = self._ctx.m(ts, steps_above_thresh=self._ctx_hyst)

            # Store for info dict
            self._current_m = float(mval)

            # Retrieve current HR/EDA for observability
            try:
                idx_index = self._ctx.hr.index
                if ts in idx_index:
                    idx = ts
                else:
                    pos = idx_index.searchsorted(ts, side="right")
                    idx = idx_index[0] if pos == 0 else idx_index[pos - 1]
                self._current_hr = float(self._ctx.hr.loc[idx])
                self._current_eda = float(self._ctx.eda.loc[idx])
            except KeyError as e:
                # Expected: timestamp not found (normal with forward-fill)
                logger.debug(f"Context data unavailable at {ts}: {e}")
            except (IndexError, AttributeError) as e:
                # Unexpected: potential bug in ContextStream
                logger.error(f"Unexpected error accessing context at {ts}: {e}", exc_info=True)
                # Re-raise in strict mode for debugging
                if getattr(self._ctx.cfg, "strict_validation", False):
                    raise

            # Apply modulation with pre-computed m(t)
            if self._mod_vmx:
                apply_vmx_multiplier(self.patient, now_ts, self._ctx, self._vmx_base, m_value=mval)
            if self._mod_p2u:
                apply_p2u_multiplier(self.patient, now_ts, self._ctx, self._p2u_base, m_value=mval)
            if self._mod_vm0:
                apply_vm0_multiplier(self.patient, now_ts, self._ctx, self._vm0_base, m_value=mval)

            # logging
            self._log_time.append(pd.Timestamp(now_ts))
            self._log_m.append(float(mval))
            self._log_vmx.append(float(self.patient._params.Vmx))
            self._log_p2u.append(float(self.patient._params.p2u))
            self._log_vm0.append(float(self.patient._params.Vm0))

        # 1) Continue with standard mini_step logic (basal/bolus + state update)
        return super().mini_step(action)

    def step(self, action, reward_fun=None):
        """Override step to add context info to the info dict."""
        if reward_fun is None:
            from simglucose.simulation.env import risk_diff

            reward_fun = risk_diff

        # Store last action for logging
        self._last_action = action
        result = super().step(action, reward_fun=reward_fun)

        # Add context info (result.info is a dict with all kwargs)
        from simglucose.simulation.env import Step

        new_info = dict(result.info)  # copy existing info
        new_info["context_m"] = self._current_m
        new_info["context_hr"] = self._current_hr
        new_info["context_eda"] = self._current_eda

        # log the requested action (controller output)
        try:
            new_info["action_basal"] = float(action.basal)
            new_info["action_bolus"] = float(action.bolus)
        except (AttributeError, TypeError, ValueError):
            # Action might not have basal/bolus attributes or non-numeric values
            pass

        return Step(
            observation=result.observation, reward=result.reward, done=result.done, **new_info
        )

    def reset(self):
        # restore baseline parameters before full reset
        self.patient._params.Vmx = self._vmx_base
        self.patient._params.p2u = self._p2u_base
        self.patient._params.Vm0 = self._vm0_base

        # reset hysteresis state
        self._ctx_hyst = {"exercise": 0, "stress": 0}

        # reset logs
        self._log_time.clear()
        self._log_m.clear()
        self._log_vmx.clear()
        self._log_p2u.clear()
        self._log_vm0.clear()

        return super().reset()

    def get_context_log(self) -> Optional[pd.DataFrame]:
        """Return DataFrame with context modulation logging for analysis.

        Note: Log buffer is limited to max_log_size samples (default: 300).
        For simulations >25h, only the most recent data is retained.
        This prevents unbounded memory growth while maintaining observability.
        """
        if not self._log_time:
            return None
        # Convert deque to list for DataFrame creation
        df = pd.DataFrame(
            {
                "m": list(self._log_m),
                "Vmx": list(self._log_vmx),
                "p2u": list(self._log_p2u),
                "Vm0": list(self._log_vm0),
            },
            index=pd.to_datetime(list(self._log_time)),
        )
        df.index.name = "Time"
        return df
