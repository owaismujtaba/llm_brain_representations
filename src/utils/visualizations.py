import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import config as config
import pdb

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import config


def plot_r2(model):
    destination_dir = Path(config.CUR_DIR, 'Results', 'Images')
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
                    r2_scores.append(values[2])  # Extract R² score
            
            if r2_scores:
                subject_r2_scores[subject] = r2_scores

    # Calculate mean R² for each subject
    subject_means = {subject: np.mean(scores) for subject, scores in subject_r2_scores.items()}

    # Print or return the means
    print("Mean R² scores for each subject:")
    for subject, mean_score in subject_means.items():
        print(f"{subject}: {mean_score:.4f}")

    # Convert data into a DataFrame
    subjects = []
    scores = []
    
    for subject, r2_values in subject_r2_scores.items():
        subjects.extend([subject] * len(r2_values))
        scores.extend(r2_values)

    data = pd.DataFrame({"Subjects": subjects, "R²": scores})

    # Plotting
    sns.set_style("whitegrid")

    plt.figure(figsize=(14, 7))

    # Box plot (Only for quartiles and mean)
    ax = sns.boxplot(data=data, x="Subjects", y="R²", palette="coolwarm", showmeans=True, 
                     meanprops={"marker": "s", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 10})

    # Swarm plot (Properly aligned with `dodge=True`)
    sns.stripplot(data=data, x="Subjects", y="R²", color=".25", size=6, alpha=0.7, jitter=True, dodge=True)

    # Customize fonts and colors
    plt.xlabel("Subjects", fontsize=20, fontweight="bold", color="#003366")  
    plt.ylabel("R²", fontsize=20, fontweight="bold", color="#003366")  

    # Change tick font size and color
    plt.xticks(rotation=45, fontsize=18,fontweight="bold",color="black")  
    plt.yticks(fontsize=18,fontweight="bold",  color="black")  

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save plot
    os.makedirs(destination_dir, exist_ok=True)
    filename = Path(destination_dir, f'{model}.pdf')
    plt.savefig(filename, dpi=800, format='pdf', bbox_inches="tight")


    return subject_means  # Return the means if needed



def plot_r2_vs_trials_all_subjects_in_bins(model):
    root_dir = config.TRAINED_DIR
    destination_dir = Path(config.CUR_DIR, 'Results', 'Images')

    plt.figure(figsize=(12, 6))

    # Get color cycle from Matplotlib
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_index = 0  # Track index for cycling through colors

    # Dictionary to track occupied y-positions in each bin to prevent text overlap
    occupied_positions = {}

    # Loop through subjects and plot on the same figure
    for subject in sorted(os.listdir(root_dir)):
        subject_path = os.path.join(root_dir, subject, "Mapping", model)
        all_r2_scores = []

        if os.path.exists(subject_path):
            for fold_file in sorted(os.listdir(subject_path)):
                if fold_file.endswith(".npy"):
                    fold_path = os.path.join(subject_path, fold_file)
                    values = np.load(fold_path)
                    r2_value = values[2]  # Extract R² score
                    all_r2_scores.append(r2_value)

        if not all_r2_scores:
            print(f"No R² scores found for subject {subject}.")
            continue

        # Create bins from 0 to 1 with a 0.1 step
        bins = np.arange(0, 1.1, 0.05)

        # Count occurrences of each R² score in the bins
        binned_data = pd.cut(all_r2_scores, bins=bins)
        trial_counts = pd.value_counts(binned_data, sort=False)

        # Get color from cycle
        line_color = color_cycle[color_index % len(color_cycle)]
        color_index += 1  # Increment color index for next subject

        # Line plot for each subject
        plt.plot(trial_counts.index.astype(str), trial_counts.values, marker='o', linestyle='--', 
                 label=f"Subject {subject}", color=line_color)

        # Annotate each point with the number of trials (excluding zeros)
        for i, (bin_label, count) in enumerate(zip(trial_counts.index.astype(str), trial_counts.values)):
            if count > 0:  # Only annotate non-zero values
                # Determine an available y-position to avoid overlap
                y_position = count
                while (i, y_position) in occupied_positions:  # Shift up if space is occupied
                    y_position += 3

                # Mark this position as occupied
                occupied_positions[(i, y_position)] = True

                #plt.text(i, y_position, str(count), fontsize=12, ha='center', va='bottom', 
                        # color=line_color, fontweight='bold')

    # Plot labels and title
    plt.xlabel("R²", fontsize=18, fontweight="bold", color="#003366")
    plt.ylabel("# Trials", fontsize=18, fontweight="bold", color="#003366")

    plt.xticks(rotation=45, fontsize=18,fontweight="bold",color="black")  
    plt.yticks(fontsize=18,fontweight="bold",  color="black")  

    # Add legend to distinguish subjects
    plt.legend(title="Subjects", fontsize=12, title_fontsize=14)

    plt.tight_layout()

    # Save plot
    os.makedirs(destination_dir, exist_ok=True)
    filename = Path(destination_dir, f'{model}_r2_vs_#trials.pdf')
    plt.savefig(filename, dpi=800, format='pdf', bbox_inches="tight")

    plt.show()




def get_r2_scores(model):
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
                    r2_scores.append(values[3])  
            
            if r2_scores:
                subject_r2_scores[subject] = r2_scores
    
    return subject_r2_scores