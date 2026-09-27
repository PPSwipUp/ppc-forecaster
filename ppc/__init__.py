"""PPC -- Plastic Predictive Cortex: a recurrent network that learns on every tick."""
from .brain import Brain, Modulator
from .core import PPCCore
from .heads import GVFHead, PredictHead
from .stream import evaluate, run, score, targets_for

__all__ = ["Brain", "Modulator", "PPCCore", "GVFHead", "PredictHead",
           "evaluate", "run", "score", "targets_for"]
