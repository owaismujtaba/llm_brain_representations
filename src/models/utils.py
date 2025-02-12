import numpy as np
import config as config

def segment_audio(audio, window_size=25, overlap=10):
    window_size_samples = int(window_size * config.AUIDO_SR / 1000) 
    overlap_samples = int(overlap * config.AUIDO_SR / 1000) 
    
    step_size = window_size_samples - overlap_samples
    chunks = []
    
    for start in range(0, len(audio) - window_size_samples, step_size):
        end = start + window_size_samples
        chunks.append(audio[start:end])
    
    return np.array(chunks)