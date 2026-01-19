"""
Objectives Module - Modular fitness objective implementations.

Usage:
    from Objectives.FitnessObjectives import FitnessObjective
    from Datastructures.dataclass import ObjectiveContext

    # Instantiate objectives from enum
    objectives = [
        obj_enum.value(model_data=model_data, device='cuda', ...)
        for obj_enum in active_objectives
    ]

    # Evaluate
    scores = {type(obj).__name__: obj.calculate_score(context) for obj in objectives}
"""

from Objectives.base import BaseObjective

__all__ = [
    "BaseObjective",
]
