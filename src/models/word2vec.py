import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class Wav2Vec2Model:
    def __init__(self, ):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


    def get_embeddings(self, audios):
        print('Extracting embeddings from Wav2Vec Model')
        audio_embeddings = []
        for audio in audios:
            input_values = self.processor(audio, return_tensors="pt", sampling_rate=16000).input_values
            with torch.no_grad():
                outputs = self.model(input_values)

            latent_space = outputs.last_hidden_state
            word_embeddings = np.array(word_embeddings)
        return word_embeddings

   
    


