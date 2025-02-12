import numpy as np
import fasttext.util
from src.utils.graphics import styled_print



class FastTextEmbedder:
    def __init__(self, ):
        fasttext.util.download_model('nl', if_exists='ignore')  # Downloads Dutch model
        self.model = fasttext.load_model('cc.nl.300.bin')
    def get_embeddings(self, words):
        styled_print("", "Extracting embeddings from FastText Model", "yellow", panel=True)
        word_embeddings = []
        for word in words:
            embedding = self.model.get_word_vector(word)
            word_embeddings.append(embedding)

        word_embeddings = np.array(word_embeddings)
        styled_print("", f"Embeddings shape: {word_embeddings.shape}", "green", panel=False)
        return word_embeddings