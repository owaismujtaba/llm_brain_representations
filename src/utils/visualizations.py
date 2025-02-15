import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import config as config
import pdb

def plot_r2(model='ElasticNet_ChatGPT'):

    root_dir = config.TRAINED_DIR

    subject_r2_scores = {}
    
    for subject in sorted(os.listdir(root_dir)):
        subject_path = os.path.join(root_dir, subject, "Mapping", model)
        if os.path.exists(subject_path):
            r2_scores = []
            
            for fold_file in sorted(os.listdir(subject_path)):
                if fold_file.endswith(".npy"):
                    fold_path = os.path.join(subject_path, fold_file)
                    values = np.load(fold_path) 
                    r2_scores.append(values[2])
            
            if r2_scores:
                subject_r2_scores[subject] = r2_scores

    # Plot the boxplots
    plt.figure(figsize=(10, 6))
    plt.boxplot(subject_r2_scores.values(), labels=subject_r2_scores.keys(), patch_artist=True)
    plt.xlabel("Subjects")
    plt.ylabel("R² Score")
    plt.title("R² Score Distribution Across Subjects")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


   