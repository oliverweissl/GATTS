import os
import numpy as np

from utils import levenshteinDistance
from CMUPhoneme.string_similarity import CMU_similarity
from ALINEPhoneme.string_dissimilarity import ALINE_dissimilarity
from NISQA.predict import NISQA_score
from google_ASR import google_ASR
from iflytek_ASR import iflytek_ASR
from whisper_ASR import whisper_ASR
from wav2vec2_ASR import wav2vec2_ASR
from speechbrain_ASR import speechbrain_ASR
from synthesis import audio_synthesis


ASR_DISPATCH = {
    'googleASR': google_ASR,
    'iflytekASR': iflytek_ASR,
    'whisperASR': whisper_ASR,
    'wav2vec2ASR': wav2vec2_ASR,
    'speechbrainASR': speechbrain_ASR,
}


class GradientEstimation:
    def __init__(self, reference_audio, reference_text, target_model, target=None, sigma=0.1, learning_rate=0.01, K=20):
        """
        :param sigma: Scaling factor for noise.
        :param learning_rate: Learning rate for updating the prosody vector.
        :param K: Number of noise vectors used for gradient approximation.
        """
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        if target_model not in ASR_DISPATCH:
            raise ValueError(f"Unsupported target_model '{target_model}'. Choose one of: {', '.join(ASR_DISPATCH)}")

        self.target_model = target_model
        self.target = target
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.K = K

    def _calculate_loss(self, p_i):
        """Evaluate the fitness of a concrete prosody vector."""

        l_emo_numpy = p_i.reshape(-1, 32)
        audio_numpy = audio_synthesis(l_emo_numpy, self.reference_audio, self.reference_text)
        tmp_audio_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SampleDir', 'synthesis.wav')
        
        audio_quality = NISQA_score(tmp_audio_file)

        asr_input = audio_numpy if self.target_model in {'whisperASR', 'wav2vec2ASR', 'speechbrainASR'} else tmp_audio_file
        transcription = ASR_DISPATCH[self.target_model](asr_input)

        if transcription == 'NA':
            loss_levenshtein = 100
            loss_CMU = 0
            loss_ALINE = 10000
        else:
            # Targeted runs maximize similarity to the target; untargeted runs
            # maximize distance from the reference text.
            loss_text = self.reference_text if self.target is None else self.target
            loss_levenshtein = levenshteinDistance(transcription, loss_text) / ((len(transcription) + len(loss_text)) / 2)
            loss_CMU = CMU_similarity(transcription, loss_text)
            loss_ALINE = ALINE_dissimilarity(transcription, loss_text)

        # loss_levenshtein: [0, 1]; loss_CMU: [0, 1]; loss_ALINE: [0, 1000]; audio_quality: [0, 5]
        loss = -10*loss_levenshtein + 0.1*loss_CMU - 0.0001*loss_ALINE + 0.05*audio_quality

        print(f'loss:{loss}, loss_levenshtein: {-10*loss_levenshtein}, loss_CMU: {0.1*loss_CMU}, loss_ALINE: {-0.0001*loss_ALINE}, audio_quality: {+0.05*audio_quality} \n')

        return loss
    
    def _estimate_gradient(self, p_i):
        """      
        :param p_i: The prosody vector at iteration i.
        :return: Estimated gradient.
        """
        gradient = 0
        for k in range(self.K):
            u_k = np.random.normal(0, 1, size=p_i.shape)
            loss = self._calculate_loss(p_i + self.sigma * u_k)
            gradient += loss * u_k
        gradient = gradient / (self.sigma * self.K)
        
        return gradient
    
    def refine_prosody_vector(self, p_i, num_iterations):
        """
        Refines an initially optimized prosody vector p_i through gradient estimation.
        
        :param num_iterations: Number of iterations to run the gradient estimation.
        :return: Refined prosody vector.
        """

        for _ in range(num_iterations):
            gradient = self._estimate_gradient(p_i)
            p_i = p_i + self.learning_rate * np.sign(gradient)
            
        return p_i

# For testing purposes
if __name__ == '__main__':

    reference_audio = './Original_MyVoiceIsThePassword.wav'
    reference_text = "My voice is the password"
    # target_model can be 'googleASR', 'iflytekASR', 'whisperASR', 'wav2vec2ASR', or 'speechbrainASR'
    target_model = 'whisperASR'
    # Run a small number of iterations
    gradient_iterations = 20

    # Initialize the GradientEstimation
    gradient_estimator = GradientEstimation(reference_audio, reference_text, target_model, sigma=0.1, learning_rate=0.01, K=20)

    # Initialize a prosody vector for testing
    exp_p0_tmp = np.exp(np.random.randn(8, 32) * 1)
    softmax_p0_tmp = exp_p0_tmp / np.sum(exp_p0_tmp, axis=-1, keepdims=True)
    p_0 = softmax_p0_tmp * 0.25

    p_refined = gradient_estimator.refine_prosody_vector(p_0, gradient_iterations)