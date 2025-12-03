import torch
from torch import Tensor
import math

from enum import Enum, auto

class AttackMode(Enum):
    TARGETED = "targeted"
    NOISE_UNTARGETED = "noise-untargeted"
    UNTARGETED = "untargeted"

class FitnessObjective(Enum):
    AVG_LOGPROB = auto()
    PPL = auto()

    # --- Quality & Intelligibility ---

    UTMOS = auto()
    # Predicts perceived MOS quality (higher = better).
    # Uses a neural proxy model for human MOS ratings.
    # Objective: MAXIMIZE audio naturalness & clarity.

    WER = auto()
    # Word Error Rate of ASR transcription (lower = better).
    # High WER means the audio is unintelligible.
    # Objective: MINIMIZE transcription errors.

    # --- Speaker Similarity (Voice Identity) ---

    WAV2VEC_GT = auto()
    # Cosine similarity between generated audio and ground-truth audio
    # using wav2vec2 speech embeddings.
    # Objective: MAXIMIZE similarity to original speaker/voice.

    WAV2VEC_TARGET = auto()
    # Cosine similarity between generated audio and target speaker audio.
    # Objective: MAXIMIZE similarity to target speaker.
    # (Useful for voice conversion / targeted mixing.)

    # --- Content Similarity (Text/Meaning Preservation) ---

    SBERT_GT = auto()
    # Semantic similarity between generated and GT text embeddings
    # using sentence-BERT.
    # Objective: MAXIMIZE preservation of original meaning/content.

    SBERT_TARGET = auto()
    # Semantic similarity between generated audio's ASR text
    # and target text using sentence-BERT.
    # Objective: MAXIMIZE resemblance to target semantics.

    # --- Text Embedding Distance (Direct Embedding Control) ---

    TEXT_EMB_GT = auto()
    # Embedding distance between GT text and generated text.
    # Often used when ASR is expensive.
    # Objective: MINIMIZE distance (stay close to GT).

    TEXT_EMB_TARGET = auto()
    # Embedding distance between target text and generated text.
    # Objective: MINIMIZE distance (move semantics toward target).

    # --- Vector Constraints (Regularization) ---

    L1 = auto()
    # L1 norm penalty on interpolation vector.
    # Encourages sparsity (few phonemes change strongly).
    # Objective: MINIMIZE absolute magnitude of perturbation.

    L2 = auto()
    # L2 norm penalty on interpolation vector.
    # Keeps perturbation small & prevents adversarial collapse.
    # Objective: MINIMIZE vector energy (smooth, small changes).

def length_to_mask(lengths: Tensor) -> Tensor:
    mask = torch.arange(lengths.max())  # Creates a Vector [0,1,2,3,...,x], where x = biggest value in lengths
    mask = mask.unsqueeze(0)  # Creates a Matrix [1,x] from Vector [x]
    mask = mask.expand(lengths.shape[0], -1)  # Expands the matrix from [1,x] to [y,x], where y = number of elements in lengths
    mask = mask.type_as(lengths)  # Assign mask the same type as lengths
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))  # gt = greater than, compares each value from lengths to a row of values in mask; unsqueeze = splits vector lengths into vectors of size 1
    return mask  # returns a mask of shape (batch_size, max_length) where mask[i, j] = 1 if j < lengths[i] and mask[i, j] = 0 otherwise.

def _pad_with_pattern(tensor: Tensor, amount: int, pattern: list[int]) -> Tensor:

    padding = torch.as_tensor(pattern, device=tensor.device, dtype=tensor.dtype)[torch.arange(amount, device=tensor.device) % len(pattern)]

    for _ in range(tensor.dim() - 1):
        padding = padding.unsqueeze(0)
    padding = padding.expand(*tensor.shape[:-1], amount)

    return torch.cat([tensor, padding], dim=-1)

def addNumbersPattern(a: Tensor, b: Tensor, pattern: list[int]) -> tuple[Tensor, Tensor]:

    len_a = a.size(-1)
    len_b = b.size(-1)

    # If equal: nothing to do
    if len_a == len_b:
        return a, b

    # Determine which tensor needs padding
    if len_a < len_b:
        a = _pad_with_pattern(a, len_b - len_a, pattern)
    else:
        b = _pad_with_pattern(b, len_a - len_b, pattern)

    return a, b

def adjustInterpolationVector(IV: Tensor, matrix: Tensor, size_per_phoneme: int) -> Tensor:

    # Matrix Multiplication, since IV not Scalar Value
    if size_per_phoneme != 1:
        IV = IV @ matrix

    IV = IV.unsqueeze(0)
    IV = IV.permute(0, 2, 1)

    return IV

def text_naturalness_from_ppl(ppl, min_loss=1.0, max_loss=10.0):
    """
    ppl: perplexity from your LM
    Returns score in [0,1], where 1 = very natural/common.
    """
    loss = math.log(ppl)             # log PPL = cross-entropy-ish
    loss = max(min(loss, max_loss), min_loss)
    return 1.0 - (loss - min_loss) / (max_loss - min_loss)

