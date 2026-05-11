import importlib
import sys


def _backend_is_ready() -> bool:
    aa = sys.modules.get("active_adaptation")
    if aa is None:
        return False
    try:
        aa.get_backend()
    except RuntimeError:
        return False
    return True


def _register_mdp_components() -> None:
    globals()["MotionTrackingCommand"] = importlib.import_module(
        ".command",
        __name__,
    ).MotionTrackingCommand
    globals()["reward"] = importlib.import_module(".reward", __name__)


def __getattr__(name: str):
    if name in {"MotionTrackingCommand", "reward"}:
        _register_mdp_components()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if _backend_is_ready():
    _register_mdp_components()
