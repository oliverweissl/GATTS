# Creating Adversarial Examples on Black-Box ASR Models via Latent Manipulation on StyleTTS2

### Yanis Wilbrand


## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/Vorgesetzter/StyleTTS2
cd StyleTTS2
```
3. Install python requirements: 
```bash
pip install -r requirements.txt
sudo apt-get install espeak-ng
```
On Windows add:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U
```
4. Download the pretrained StyleTTS 2 on LJSpeech corpus in 24 kHz from [https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main](https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main).

## Inference
1. Run the inference notebook: [adversarial_tts.ipynb](https://github.com/Vorgesetzter/StyleTTS2/blob/main/adversarial_tts.ipynb) 

2. Run the inference colab notebook:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vorgesetzter/StyleTTS2/blob/main/adversarial_tts_colab.ipynb)

3. Run inference in python file: [adversarial_tts.py](https://github.com/Vorgesetzter/StyleTTS2/blob/main/adversarial_tts.py) 

***Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.*** 
