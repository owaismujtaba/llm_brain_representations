import torch
import torchaudio
import librosa
import numpy as np
import torchaudio
import pdb
import soundfile as sf
import config as config

class Wav2Vec2Model:
    def __init__(self, ):
        print('Word2Vec')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
        # bundle = torchaudio.pipelines.WAV2VEC2_BASE
        # Build the model and load pretrained weight.
        self.model = self.bundle.get_model().to(device)
        self.layer = config.LAYER

    def get_embeddings(self, audios):
        print('Extracting embeddings from Wav2Vec Model')
        audio_embeddings = []
        for audio in audios:
            with torch.inference_mode():
                audio = torch.tensor(audio)
                sf.write(f'a.wav',audio, samplerate=16000)
                waveform, sample_rate = torchaudio.load(f'a.wav')

                features, _ = self.model.extract_features(waveform)  # , num_layers=1         
                audio_embedding = np.concatenate([layer_embed.numpy(force=True) for layer_embed in features])
                audio_embeddings.append(audio_embedding[config.LAYER])
            
        audio_embeddings = np.array(audio_embeddings)
        print(audio_embeddings.shape)

        return audio_embeddings
        

   
    


