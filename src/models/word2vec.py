import numpy as np
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec

class Wav2Vec2Model:
    def __init__(self, ):
        self.corpus = api.load('text8')
        self.model = Word2Vec(
            self.corpus, vector_size=768, 
            window=5, min_count=5
        )

    def get_embeddings(self, words):
        print('Extracting embeddings from Wav2Vec Model')
        word_embeddings = []
        for word in words:
            embedding = self.model.wv[word]
            word_embeddings.append(embedding)

        word_embeddings = np.array(word_embeddings)
        return word_embeddings

   
    


