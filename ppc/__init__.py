"""PPC -- Plastic Predictive Cortex: a recurrent network that learns on every tick."""
from .brain import Brain, Modulator
from .core import PPCCore
from .forecaster import Forecaster
from .heads import GVFHead, PredictHead
from .panel import PanelForecaster
from .stream import evaluate, run, score, targets_for

__all__ = ["Brain", "Forecaster", "PanelForecaster", "Modulator", "PPCCore", "GVFHead", "PredictHead",
           "evaluate", "run", "score", "targets_for"]
