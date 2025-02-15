import pdb
import numpy as np
from src.dataset.data_reader import DatasetReader
from src.dataset.data_loader import DataLoader
from src.models.chatgpt2 import ChatGPTEmbedder
from src.models.fasttext import FastTextEmbedder

from src.models.trainner import ModelTrainer
from src.models.models import RegressionModel, ElasticNetModel


from src.utils.visualizations import plot_r2

plot_r2()



#from src.models.word2vec import Wav2Vec2Model
#from src.analysis.feature_analysis import eeg_and_chatGPT_feature_comptability
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

    gpt_embedder = ChatGPTEmbedder()
    gpt_embeddings = gpt_embedder.get_embeddings(data_loader.word_labels)

    fasttext_embedder = FastTextEmbedder()
    fasttext_embeddings = fasttext_embedder.get_embeddings(data_loader.word_labels)


    trainer = ModelTrainer(
        model_name='ElasticNet_ChatGPT',
        subject_id=subjetc_id
    )

    model = ElasticNetModel()
    trainer.train_model(
        model=model,
        X=gpt_embeddings, 
        y=eeg_trials
    )

    trainer = ModelTrainer(
        model_name='ElasticNet_FastText',
        subject_id=subjetc_id
    )
    model = ElasticNetModel()
    trainer.train_model(
        model=model,
        X=fasttext_embeddings, 
        y=eeg_trials
    )




