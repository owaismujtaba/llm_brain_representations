import pdb
import numpy as np
import os
from src.dataset.data_reader import DatasetReader
from src.dataset.data_loader import DataLoader
from src.models.chatgpt2 import ChatGPTEmbedder
from src.models.fasttext import FastTextEmbedder

from src.models.trainner import ModelTrainer
from src.models.models import  ElasticNetModel, NeuralNetwork
from src.models.utils import EEGPCAProcessor

from src.utils.visualizations import plot_r2
import config as config

#ax_1 = plot_r2(model='ElasticNet_ChatGPT')
#ax_2 = plot_r2(model='ElasticNet_FastText')
from pathlib import Path
from src.models.word2vec import Wav2Vec2Model

def load_subject_embeddings(subject_folder):
    embeddings = []
    
    for file in os.listdir(subject_folder):
        if file.endswith(".npy"):
            word = os.path.splitext(file)[0]  # Extract word from filename
            embeddings.append(np.load(os.path.join(subject_folder, file)))
    
    return np.array(embeddings)

if config.TRAIN_WAV2VEC_BASED:
    for subject in range(1, 11):

        subjetc_id = f'0{subject}'
        '''
        reader = DatasetReader(
            sub_id=subjetc_id
        )
        data_loader = DataLoader(
            eeg=reader.eeg,
            audio=reader.audio,
            stimulus=reader.stimulus
        )
        
        eeg_trials = data_loader.eeg_trials
        eeg_trials = np.mean(eeg_trials, axis=1)
        audio_trials = data_loader.audio_trials
        word_trials = data_loader.word_labels
        '''
        filepath = r'C:\DeepRESTORE\llm_brain_representations\data\embeddings\embeddings\wav2vec'
        for file in os.listdir(filepath):
            subject_id = file
            folder = Path(filepath, )
            audio_embeddings = 
        pdb.set_trace()
        wav2vec_embedder = Wav2Vec2Model()
        wav2vec_embeddings = wav2vec_embedder.get_embeddings(audio_trials)

        trainer = ModelTrainer(
            model_name='ElasticNet_Wav2Vec',
            subject_id=subjetc_id
        )
        #input_shape = (wav2vec_embeddings.shape[1],)
        #wav2vec_embeddings = 
        pca = EEGPCAProcessor()
        audio_pca = pca.fit_transform(eeg_trials=wav2vec_embeddings)
        output_shape = (eeg_trials.shape[1])
        #model = NeuralNetwork(input_shape=input_shape, output_shape=output_shape)
        model = ElasticNetModel()
        trainer.train_model(
            model=model,
            X=audio_pca, 
            y=eeg_trials,
            trial_labels=word_trials
        )



if config.TRAIN_CHATGPT_BASED:
    for subject in range(1, 11):
        subjetc_id = f'0{subject}'
        reader = DatasetReader(
            sub_id=subjetc_id
        )
        data_loader = DataLoader(
            eeg=reader.eeg,
            audio=reader.audio,
            stimulus=reader.stimulus
        )

        eeg_trials = data_loader.eeg_trials
        eeg_trials = np.mean(eeg_trials, axis=1)
        audio_trials = data_loader.audio_trials
        word_trials = data_loader.word_labels
    
        eeg_pca_processor = EEGPCAProcessor()
        #eeg_pca = eeg_pca_processor.fit_transform(eeg_trials=eeg_trials)
        #eeg_trials = eeg_pca
        
        gpt_embedder = ChatGPTEmbedder()
        gpt_embeddings = gpt_embedder.get_embeddings(word_trials)

        fasttext_embedder = FastTextEmbedder()
        fasttext_embeddings = fasttext_embedder.get_embeddings(word_trials)


        trainer = ModelTrainer(
            model_name='ElasticNet_ChatGPT',
            subject_id=subjetc_id
        )
      
        input_shape = (gpt_embeddings.shape[1],)
        output_shape = (eeg_trials.shape[1])
        #model = NeuralNetwork(input_shape=input_shape, output_shape=output_shape)
        model = ElasticNetModel()
        trainer.train_model(
            model=model,
            X=gpt_embeddings, 
            y=eeg_trials,
            trial_labels=word_trials
        )

        input_shape = (fasttext_embeddings.shape[1],)
        output_shape = (eeg_trials.shape[1])
        #model = NeuralNetwork(input_shape=input_shape, output_shape=output_shape)
        model = ElasticNetModel()
        trainer = ModelTrainer(
            model_name='ElasticNet_FastText',
            subject_id=subjetc_id
        )
        
        trainer.train_model(
            model=model,
            X=fasttext_embeddings, 
            y=eeg_trials,
            trial_labels=word_trials
        )
        
        



