# GenAdvTTS — Setup & Reproduction Guide

## Installation

> **Recommended:** use a virtual environment (`python -m venv .venv`) or conda before installing.

```bash
pip install -r requirements.txt
sudo apt-get install espeak-ng
```

Make sure to copy the 'google_tokens.json' file from [SMACK](https://github.com/WUSTL-CSPL/SMACK) into 'scripts/SMACK'

## Checkpoint

Download the StyleTTS2 LJSpeech checkpoint from [HuggingFace](https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main/Models/LJSpeech):

```bash
wget -O checkpoints/STT2.pth https://huggingface.co/yl4579/StyleTTS2-LJSpeech/blob/main/Models/LJSpeech/epoch_2nd_00100.pth
```

For SMACK checkpoints please donwload and save them under 'checkpoints/' you can find the download link [here](https://github.com/WUSTL-CSPL/SMACK?tab=readme-ov-file#installation)

## Running the Paper Experiments

Each script splits the 100 Harvard sentences across two GPUs and runs them in parallel.

**Adversarial TTS (main experiment):**
```bash
bash scripts/run_adversarial_tts_harvard.sh
```

**Waveform baseline:**
```bash
bash scripts/run_waveform_harvard.sh
```


**SMACK:**
Make sure to create the SMACK conda environment from 'configs/smack.yml'    

```bash
bash scripts/run_smack_harvard.sh
```

**PGD:**
>First setup the PGD environemt using 'configs/setup_pgd_env.sh'

```bash
bash scripts/run_pgd_harvard.sh
```


Some scripts launch two background processes — GPU 0 handles sentences 1–50, GPU 1 handles 51–100 — and wait for both to finish.

## Single-GPU Setup

To run on a single GPU, replace both invocations with one command covering sentences 1–100 and drop the `--gpu` flag:

```bash
python scripts/adversarial_tts_harvard.py \
    --harvard_sentences_start 1 \
    --harvard_sentences_end 100 \
    --loop_count 1 \
    --num_generations 100 \
    --pop_size 100 \
    --batch_size 100 \
    --iv_scalar 0.5 \
    --size_per_phoneme 1 \
    --num_rms_candidates 1 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target
```

The scripts can also be extended for more than two GPUs by adding further background processes with the appropriate `--harvard_sentences_start`, `--harvard_sentences_end`, and `--gpu` values.
