from torch import Tensor
from dataclasses import dataclass

@dataclass
class InferenceResult:
    h_text: Tensor
    h_aligned: Tensor
    f0_pred: Tensor
    a_pred: Tensor
    n_pred: Tensor
    style_vector_prosodic: Tensor