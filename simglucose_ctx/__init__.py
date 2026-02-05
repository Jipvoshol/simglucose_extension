# Makes this directory a package for easy imports

from .context import ContextStream, ContextConfig
from .sim_hook import apply_vm0_multiplier, apply_vmx_multiplier, apply_p2u_multiplier


def __getattr__(name):
    """Lazy import for ContextAwareT1DSimEnv to avoid hard simglucose dependency.

    This allows `from simglucose_ctx.context import ContextStream` to work
    without simglucose installed. The env_wrapper (which depends on simglucose)
    is only imported when actually accessed.
    """
    if name == "ContextAwareT1DSimEnv":
        from .env_wrapper import ContextAwareT1DSimEnv

        return ContextAwareT1DSimEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ContextStream",
    "ContextConfig",
    "ContextAwareT1DSimEnv",
    "apply_vm0_multiplier",
    "apply_vmx_multiplier",
    "apply_p2u_multiplier",
]
