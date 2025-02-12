import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model
from src.utils.graphics import styled_print

import pdb

class ChatGPTEmbedder:
    def __init__(self):
        self.model_name = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2Model.from_pretrained(self.model_name)

    def get_embeddings(self, words):
        styled_print("", "Extracting embeddings from ChatGPT2 Model", "yellow", panel=True)
        word_embeddings = []

        for word in words:
            inputs = self.tokenizer(word, add_special_tokens=False, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            last_hidden_state = outputs.last_hidden_state
            last_hidden_state = last_hidden_state.mean(dim=1)
            word_embedding = last_hidden_state
            word_embeddings.append(word_embedding)
        word_embeddings = np.array(word_embeddings)
        word_embeddings = word_embeddings.reshape(
            word_embeddings.shape[0], word_embeddings.shape[2]
        )

        styled_print("", f"Embeddings shape: {word_embeddings.shape}", "green", panel=False)
        return word_embeddings

        