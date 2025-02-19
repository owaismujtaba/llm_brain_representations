import numpy as np
import config as config

from sklearn.decomposition import PCA


from sklearn.decomposition import PCA
import numpy as np

from sklearn.decomposition import PCA
import numpy as np

class EEGPCAProcessor:
    def __init__(self, variance_retained=0.90):
        self.variance_retained = variance_retained
        self.pca = None  # A single PCA model for all samples

    def fit_transform(self, eeg_trials):
        """
        Applies PCA on the entire dataset by flattening time and channels, while retaining the given variance.

        Args:
            eeg_trials (numpy.ndarray): EEG trials of shape (N, T, C)

        Returns:
            eeg_pca (numpy.ndarray): Transformed EEG trials in flattened form of shape (N, new_C)
        """
        N, T, C = eeg_trials.shape
        
        # Flatten the entire dataset (N samples, T * C)
        eeg_flat = eeg_trials.reshape(N, -1)  # Shape: (N, T * C)
        
        # Apply PCA across the entire dataset
        self.pca = PCA(n_components=self.variance_retained, svd_solver='full')
        eeg_pca_flat = self.pca.fit_transform(eeg_flat)  # Shape: (N, new_C)
        
        return eeg_pca_flat

    def inverse_transform(self, eeg_pca_flat):
        """
        Inverses the PCA transformation for the entire dataset.

        Args:
            eeg_pca_flat (numpy.ndarray): Transformed EEG trials in flattened form of shape (N, new_C)

        Returns:
            eeg_reconstructed_flat (numpy.ndarray): Reconstructed EEG trials in flattened form of shape (N, T * C)
        """
        # Inverse transform using the single PCA model
        eeg_reconstructed_flat = self.pca.inverse_transform(eeg_pca_flat)  # Shape: (N, T * C)
        
        return eeg_reconstructed_flat





def segment_audio(audio, window_size=25, overlap=10):
    window_size_samples = int(window_size * config.AUIDO_SR / 1000) 
    overlap_samples = int(overlap * config.AUIDO_SR / 1000) 
    
    step_size = window_size_samples - overlap_samples
    chunks = []
    
    for start in range(0, len(audio) - window_size_samples, step_size):
        end = start + window_size_samples
        chunks.append(audio[start:end])
    
    return np.array(chunks)