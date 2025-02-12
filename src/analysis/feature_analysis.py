from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


import pdb

def eeg_and_chatGPT_feature_comptability(eeg_features, chatgpt_features):
    eeg_features = eeg_features.mean(axis=1)
    chatgpt_features = chatgpt_features.reshape(
        chatgpt_features.shape[0], chatgpt_features.shape[2]
    )

    
    pca = PCA(n_components=eeg_features.shape[1])

    chatgpt_features = pca.fit_transform(
        chatgpt_features.reshape(chatgpt_features.shape[0], -1)
    )
    
    pdb.set_trace()
    correlations = [pearsonr(chatgpt_features[i], eeg_features[i])[0] for i in range(eeg_features.shape[0])]
    similarity_matrix = cosine_similarity(chatgpt_features, eeg_features)
    return eeg_features, chatgpt_features
