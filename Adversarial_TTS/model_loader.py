"""
Model loader and environment initialization for adversarial TTS.

This module handles:
1. Configuration parsing and validation
2. Required model loading (TTS, ASR)
3. Audio data generation (GT, target)
4. Optimizer initialization
5. ObjectiveManager creation (objectives lazy-load their own models)
"""

import numpy as np
import torch
import whisper

# Local imports
from Models.styletts2 import StyleTTS2
from helper import addNumbersPattern

# Import Pymoo components
from Optimizer.pymoo_optimizer import PymooOptimizer
from pymoo.algorithms.moo.nsga2 import NSGA2

# Import dataclasses and enums
from Datastructures.dataclass import ModelData, ConfigData, AudioData, EmbeddingData
from Datastructures.enum import FitnessObjective, AttackMode

# Import ObjectiveManager
from Objectives.manager import ObjectiveManager


def initialize_environment(args, device):
    """
    Initialize the complete environment for adversarial TTS optimization.

    Args:
        args: Parsed command-line arguments
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Tuple of (config_data, model_data, audio_data, embedding_data, objective_manager)
    """
    # 1. Load configuration
    config_data = _load_configuration(args, device)
    if config_data is None:
        return None, None, None, None, None

    config_data.print_summary()

    # 2. Load required models (TTS + ASR)
    tts_model, asr_model = _load_required_models(device)

    # 3. Generate audio data (GT + target)
    audio_data = _generate_audio_data(config_data, tts_model, device)

    # 4. Initialize optimizer
    optimizer = load_optimizer(audio_data, config_data)

    # 5. Create model_data container
    model_data = ModelData(tts_model=tts_model, asr_model=asr_model, optimizer=optimizer)

    # 6. Initialize embedding_data (empty - will be populated by objectives)
    embedding_data = EmbeddingData()

    # 7. Create ObjectiveManager (objectives lazy-load their models here)
    print("\n[INFO] Initializing ObjectiveManager...")
    objective_manager = ObjectiveManager(
        config=config_data,
        model_data=model_data,
        device=device,
        embedding_data=embedding_data
    )

    # 8. Pre-compute reference embeddings that need audio_data
    _precompute_audio_embeddings(objective_manager, audio_data, config_data, device)

    return config_data, model_data, audio_data, embedding_data, objective_manager


def _load_configuration(args, device):
    """Parse and validate configuration from command-line arguments."""

    # 1. Define objective order (determines column order in fitness matrix)
    objective_order: list[FitnessObjective] = [
        FitnessObjective.PHONEME_COUNT,
        FitnessObjective.UTMOS,
        FitnessObjective.PPL,
        FitnessObjective.PESQ,
        FitnessObjective.L1,
        FitnessObjective.L2,
        FitnessObjective.WER_TARGET,
        FitnessObjective.SBERT_TARGET,
        FitnessObjective.TEXT_EMB_TARGET,
        FitnessObjective.WHISPER_PROB,
        FitnessObjective.WER_GT,
        FitnessObjective.SBERT_GT,
        FitnessObjective.TEXT_EMB_GT,
        FitnessObjective.WAV2VEC_SIMILAR,
        FitnessObjective.WAV2VEC_DIFFERENT,
        FitnessObjective.WAV2VEC_ASR,
    ]

    # 2. Subspace optimization setup
    subspace_optimization = args.subspace_optimization
    random_matrix = torch.from_numpy(
        np.random.rand(args.size_per_phoneme, 512)
    ).to(device).float()

    # 3. Parse attack mode
    try:
        mode = AttackMode[args.mode]
    except KeyError:
        print(f"Invalid mode '{args.mode}'. Available modes: {[m.name for m in AttackMode]}")
        return None

    # 4. Parse active objectives
    active_objectives = set()
    for obj_name in args.ACTIVE_OBJECTIVES:
        try:
            active_objectives.add(FitnessObjective[obj_name])
        except KeyError:
            print(f"Warning: '{obj_name}' is not a valid FitnessObjective. Skipping.")

    if not active_objectives:
        print("Error: No valid active_objectives selected.")
        return None

    # Maintain order from objective_order
    active_objectives = [obj for obj in objective_order if obj in active_objectives]

    # 5. Parse thresholds
    thresholds = {}
    if args.thresholds:
        for t in args.thresholds:
            try:
                key_str, val_str = t.split("=")
                obj_enum = FitnessObjective[key_str.strip()]
                thresholds[obj_enum] = float(val_str.strip())
            except Exception as e:
                print(f"Error parsing threshold '{t}': {e}")
                return None

    # 6. Validate batch size
    if args.batch_size > args.pop_size or args.batch_size <= 0:
        args.batch_size = args.pop_size

    # 7. Create and return ConfigData
    return ConfigData(
        text_gt=args.ground_truth_text,
        text_target=args.target_text,
        num_generations=args.num_generations,
        pop_size=args.pop_size,
        loop_count=args.loop_count,
        iv_scalar=args.iv_scalar,
        size_per_phoneme=args.size_per_phoneme,
        batch_size=args.batch_size,
        notify=args.notify,
        mode=mode,
        active_objectives=active_objectives,
        thresholds=thresholds,
        objective_order=objective_order,
        diffusion_steps=5,
        embedding_scale=1,
        subspace_optimization=subspace_optimization,
        random_matrix=random_matrix,
    )


def _load_required_models(device):
    """Load the required models (TTS and ASR)."""
    print("Loading StyleTTS2...")
    tts = StyleTTS2()
    tts.load_models()
    tts.load_checkpoints()
    tts.sample_diffusion()

    print("Loading ASR Model (Whisper)...")
    asr = whisper.load_model("tiny", device=device)

    return tts, asr


def _generate_audio_data(config, tts, device):
    """Generate ground-truth and target audio data."""

    noise = torch.randn(1, 1, 256).to(device)

    # Handle text embeddings based on attack mode
    if config.mode is AttackMode.TARGETED:
        # Text -> Tokens, while adding tokens if necessary
        tokens_gt, tokens_target = addNumbersPattern(
            tts.preprocess_text(config.text_gt),
            tts.preprocess_text(config.text_target),
            [16, 4]
        )
        h_text_gt, h_bert_raw_gt, h_bert_gt, input_lengths, text_mask = tts.extract_embeddings(tokens_gt)
        h_text_target, h_bert_raw_target, h_bert_target, _, _ = tts.extract_embeddings(tokens_target)
    else:
        tokens_gt = tts.preprocess_text(config.text_gt)
        h_text_gt, h_bert_raw_gt, h_bert_gt, input_lengths, text_mask = tts.extract_embeddings(tokens_gt)

        # Random embeddings for untargeted modes
        h_text_target = torch.randn_like(h_text_gt)
        h_text_target /= h_text_target.norm()

        h_bert_raw_target = torch.randn_like(h_bert_raw_gt)
        h_bert_raw_target /= h_bert_raw_target.norm()

        h_bert_target = torch.randn_like(h_bert_gt)
        h_bert_target /= h_bert_target.norm()

    # Generate style vectors
    style_ac_gt, style_pro_gt = tts.compute_style_vector(
        noise, h_bert_raw_gt, config.embedding_scale, config.diffusion_steps
    )
    style_ac_target, style_pro_target = tts.compute_style_vector(
        noise, h_bert_raw_target, config.embedding_scale, config.diffusion_steps
    )

    # Run inference for ground-truth and target
    audio_gt = tts.inference_on_embedding(
        input_lengths, text_mask, h_bert_gt, h_text_gt, style_ac_gt, style_pro_gt
    )
    audio_target = tts.inference_on_embedding(
        input_lengths, text_mask, h_bert_target, h_text_target, style_ac_target, style_pro_target
    )

    return AudioData(
        audio_gt, audio_target,
        h_text_gt, h_text_target,
        h_bert_raw_gt, h_bert_raw_target,
        h_bert_gt, h_bert_target,
        input_lengths, text_mask,
        style_ac_gt, style_pro_gt,
        noise
    )


def load_optimizer(audio_data, config_data):
    """Initialize the NSGA-II optimizer."""

    phoneme_count = audio_data.input_lengths.detach().cpu().item()

    return PymooOptimizer(
        bounds=(0, 1),
        algorithm=NSGA2,
        algo_params={"pop_size": config_data.pop_size},
        num_objectives=len(config_data.active_objectives),
        solution_shape=(phoneme_count, config_data.size_per_phoneme),
    )


def _precompute_audio_embeddings(objective_manager, audio_data, config_data, device):
    """
    Pre-compute audio embeddings that require audio_data.

    This is needed for Wav2Vec objectives that compare against GT audio embedding.
    The embedding is computed once here and stored in embedding_data.
    """
    # Check if any Wav2Vec objectives are active
    wav2vec_objectives = [
        FitnessObjective.WAV2VEC_SIMILAR,
        FitnessObjective.WAV2VEC_DIFFERENT,
        FitnessObjective.WAV2VEC_ASR
    ]

    if not any(obj in config_data.active_objectives for obj in wav2vec_objectives):
        return

    # Get the Wav2Vec objective (any of them, they share the model)
    wav2vec_obj = None
    for obj_enum in wav2vec_objectives:
        wav2vec_obj = objective_manager.get_objective_instance(obj_enum)
        if wav2vec_obj is not None:
            break

    if wav2vec_obj is None:
        return

    # Compute GT audio embedding
    print("[INFO] Pre-computing Wav2Vec GT embedding...")
    with torch.no_grad():
        wav2vec_embedding_gt = torch.mean(
            wav2vec_obj.wav2vec_model(
                **wav2vec_obj.wav2vec_processor(
                    audio_data.audio_gt,
                    return_tensors="pt",
                    sampling_rate=16000
                ).to(device)
            ).last_hidden_state,
            dim=1
        )

    objective_manager.embedding_data.wav2vec_embedding_gt = wav2vec_embedding_gt

    # Compute target audio embedding if in targeted mode
    if config_data.mode is AttackMode.TARGETED:
        wav2vec_embedding_target = torch.mean(
            wav2vec_obj.wav2vec_model(
                **wav2vec_obj.wav2vec_processor(
                    audio_data.audio_target,
                    return_tensors="pt",
                    sampling_rate=16000
                ).to(device)
            ).last_hidden_state,
            dim=1
        )
        objective_manager.embedding_data.wav2vec_embedding_target = wav2vec_embedding_target

    elif config_data.mode is AttackMode.NOISE_UNTARGETED:
        wav2vec_embedding_target = torch.randn_like(wav2vec_embedding_gt)
        wav2vec_embedding_target /= wav2vec_embedding_target.norm()
        objective_manager.embedding_data.wav2vec_embedding_target = wav2vec_embedding_target
