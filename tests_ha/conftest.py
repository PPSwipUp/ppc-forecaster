import os
# import pandas before the test clock is frozen: first imported while freezegun has replaced `datetime`, pandas 3 on
# Python 3.13 crashes the interpreter (stack overflow in the garbage collector, in pandas' NaT, a datetime subclass)
import pandas  # noqa: F401

# The repo root is deliberately NOT put on sys.path: `ppc` must come from the installed wheel (with its compiled
# engine), not from this source folder.  Run with the `pytest` command (not `python -m pytest`, which adds the cwd).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytest_plugins = "pytest_homeassistant_custom_component"


def use_our_custom_components():
    """The harness ships its own `custom_components` package (importable once `hass` is set up) and Home Assistant
    finds custom integrations through that package's __path__, so add this repository's folder to it."""
    import custom_components
    ours = os.path.join(ROOT, "custom_components")
    if ours not in custom_components.__path__:
        custom_components.__path__.append(ours)
