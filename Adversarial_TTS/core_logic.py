"""
Core optimization logic for adversarial TTS.

This module contains the main optimization loop that:
1. Generates audio from interpolation vectors
2. Runs ASR to get transcriptions
3. Evaluates fitness objectives using ObjectiveManager
4. Updates the optimizer based on fitness scores
"""

import torch
import numpy as np
import re
import torchaudio.functional as torchaudio_functional
from tqdm import tqdm
import whisper
from whisper.tokenizer import get_tokenizer

from Datastructures.dataclass import FitnessData, StepContext
from Datastructures.enum import FitnessObjective
from Objectives.manager import ObjectiveManager


def run_optimization_generation(
    config_data,
    model_data,
    audio_data,
    embedding_data,
    objective_manager: ObjectiveManager,
    iteration: int,
    device: str
):
    """
    Runs the optimization loop for one iteration.

    Args:
        config_data: Configuration data
        model_data: Model data (TTS, ASR, optimizer)
        audio_data: Audio data (GT, target audio)
        embedding_data: Pre-computed embeddings
        objective_manager: ObjectiveManager instance for fitness evaluation
        iteration: Current iteration number
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        FitnessData, progress_bar, stop_optimization flag, final generation number
    """

    # ==== Extract Data ====
    asr_model = model_data.asr_model
    active_objectives = config_data.active_objectives
    batch_size = config_data.batch_size

    # ==== Main Optimization Loop ====
    pareto_fitness_history = []
    mean_fitness_history = []
    total_fitness_history = []
    stop_optimization = False

    progress_bar = tqdm(
        range(config_data.num_generations),
        desc=f"Current Generation {iteration + 1}",
        leave=False
    )
    gen = -1

    # ==== WHISPER_PROB Setup (if active) ====
    target_tokens_template = None
    real_asr_model = _get_real_asr_model(asr_model)

    if FitnessObjective.WHISPER_PROB in active_objectives:
        target_tokens_template = _prepare_whisper_tokens(
            real_asr_model, config_data.text_target, device
        )

    # ==== Generation Loop ====
    for gen in progress_bar:

        gen_scores: dict[FitnessObjective, list[float]] = {obj: [] for obj in active_objectives}
        interpolation_vectors_full = torch.from_numpy(
            model_data.optimizer.get_x_current()
        ).to(device).float()

        options = whisper.DecodingOptions()

        # ==== Batch Loop ====
        for batch_idx in range(0, config_data.pop_size, batch_size):

            # 1. TTS Inference - Generate audio from interpolation vectors
            audio_mixed_batch, current_batch_size, interpolation_vectors = \
                model_data.tts_model.inference_on_interpolation_vectors(
                    interpolation_vectors_full, batch_idx, batch_size, config_data, audio_data
                )

            # 2. Prepare audio for ASR
            audio_tensor = torch.from_numpy(audio_mixed_batch).squeeze(1).float().to(device)
            audio_tensor = torchaudio_functional.resample(audio_tensor, 24000, 16000)
            audio_tensor = whisper.pad_or_trim(audio_tensor)

            # 3. Create Mel Spectrogram
            mel_batch = whisper.log_mel_spectrogram(
                audio_tensor, n_mels=real_asr_model.dims.n_mels
            ).to(device)

            # 4. Compute WHISPER_PROB batch values (if active)
            batch_whisper_values = _compute_whisper_prob_batch(
                asr_model, mel_batch, target_tokens_template,
                current_batch_size, active_objectives
            )

            # 5. Run ASR decoding
            results = whisper.decode(real_asr_model, mel_batch, options)

            # 6. Process ASR results and clean text
            asr_texts = [r.text for r in results]
            clean_texts = [re.sub(r'[^a-zA-Z\s]', '', t).strip() for t in asr_texts]

            # 7. Create StepContext for batch evaluation
            context = StepContext(
                audio_mixed=torch.from_numpy(audio_mixed_batch).to(device),
                asr_text=asr_texts,
                clean_text=clean_texts,
                interpolation_vector=interpolation_vectors,
                whisper_prob=batch_whisper_values
            )

            # 8. Evaluate all objectives on the batch
            batch_scores = objective_manager.evaluate_batch(context, audio_data)

            # 9. Collect scores and check early stopping
            for i in range(current_batch_size):
                current_ind_scores: dict[FitnessObjective, float] = {}

                # Handle garbage text (too short)
                if len(clean_texts[i]) < 2:
                    for obj in active_objectives:
                        gen_scores[obj].append(1.0)
                        current_ind_scores[obj] = 1.0
                else:
                    # Extract scores for this individual from batch results
                    for obj in active_objectives:
                        score = batch_scores[obj][i]
                        gen_scores[obj].append(score)
                        current_ind_scores[obj] = score

                # Early stopping check
                if _check_early_stopping(config_data, current_ind_scores):
                    stop_optimization = True

        # ==== End of Generation ====

        # Calculate per-generation statistics
        gen_mean, gen_total, total_fitness = _compute_generation_stats(
            gen, gen_scores, config_data
        )

        # Update optimizer
        model_data.optimizer.assign_fitness(gen_total)
        model_data.optimizer.update()

        # Capture Pareto front
        current_front = np.array([c.fitness for c in model_data.optimizer.best_candidates])

        # Add to history
        mean_fitness_history.append(gen_mean)
        total_fitness_history.append(total_fitness)
        pareto_fitness_history.append(current_front)

        if stop_optimization:
            print(f"\n[!] Early Stopping Triggered at Generation {gen + 1} (Thresholds met).")
            break

    return FitnessData(mean_fitness_history, pareto_fitness_history, total_fitness_history), \
           progress_bar, stop_optimization, gen


# ==============================================================================
# Helper Functions
# ==============================================================================

def _get_real_asr_model(asr_model):
    """Extract the actual model from DataParallel wrapper if needed."""
    if isinstance(asr_model, torch.nn.DataParallel):
        return asr_model.module
    return asr_model


def _prepare_whisper_tokens(real_asr_model, text_target: str, device: str) -> torch.Tensor:
    """Prepare tokenized target text for WHISPER_PROB computation."""
    tokenizer = get_tokenizer(real_asr_model.is_multilingual)
    target_ids = list(tokenizer.sot_sequence) + tokenizer.encode(text_target) + [tokenizer.eot]
    return torch.tensor([target_ids]).to(device)


def _compute_whisper_prob_batch(
    asr_model,
    mel_batch: torch.Tensor,
    target_tokens_template: torch.Tensor,
    batch_size: int,
    active_objectives: list
) -> list:
    """
    Compute WHISPER_PROB fitness values for a batch.

    Returns list of fitness values, or list of None if WHISPER_PROB not active.
    """
    if FitnessObjective.WHISPER_PROB not in active_objectives:
        return [None] * batch_size

    if target_tokens_template is None:
        return [None] * batch_size

    target_tokens_batch = target_tokens_template.expand(batch_size, -1)

    with torch.no_grad():
        logits = asr_model(mel_batch, target_tokens_batch)

    # Remove start and end token
    logits_shifted = logits[:, :-1, :]
    targets_shifted = target_tokens_batch[:, 1:]

    # Cross-entropy loss per token
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    raw_losses = loss_fct(
        logits_shifted.reshape(-1, logits_shifted.size(-1)),
        targets_shifted.reshape(-1)
    )

    # Average loss over sentence length
    sample_losses = raw_losses.reshape(batch_size, -1).mean(dim=1)

    # Convert to fitness score (0.0 = best, 1.0 = worst)
    probs = torch.exp(-sample_losses)
    vals = 1.0 - probs

    return vals.detach().cpu().tolist()


def _check_early_stopping(config_data, current_ind_scores: dict) -> bool:
    """Check if current individual meets all threshold criteria."""
    if not config_data.thresholds:
        return False

    for obj in config_data.active_objectives:
        if obj in config_data.thresholds:
            if current_ind_scores[obj] > config_data.thresholds[obj]:
                return False

    return True


def _compute_generation_stats(gen: int, gen_scores: dict, config_data):
    """Compute statistics for the current generation."""
    gen_mean: dict[str, float] = {"Generation": gen}
    gen_total: list[np.ndarray] = []

    for obj in config_data.objective_order:
        if obj not in config_data.active_objectives:
            continue

        arr = np.array(gen_scores[obj], dtype=float)
        gen_mean[f"{obj.name}_Mean"] = float(np.mean(arr))
        gen_total.append(arr)

    total_fitness = np.column_stack(gen_total)

    return gen_mean, gen_total, total_fitness
