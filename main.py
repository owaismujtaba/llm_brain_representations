import pdb

from src.dataset.data_reader import DatasetReader
from src.dataset.data_loader import DataLoader
from src.models.chatgpt2 import ChatGPTEmbedder
from src.models.fasttext import FastTextEmbedder
#from src.models.word2vec import Wav2Vec2Model
from src.analysis.feature_analysis import eeg_and_chatGPT_feature_comptability
reader = DatasetReader()

data_loader = DataLoader(
    eeg=reader.eeg,
    audio=reader.audio,
    stimulus=reader.stimulus
)
eeg_trails = data_loader.eeg_trials
audio_trails = data_loader.audio_trials
word_trials = data_loader.word_labels

gpt_embedder = ChatGPTEmbedder()
gpt_embeddings = gpt_embedder.get_embeddings(data_loader.word_labels)

fasttext_embedder = FastTextEmbedder()
fasttext_embeddings = fasttext_embedder.get_embeddings(data_loader.word_labels)

pdb.set_trace()


