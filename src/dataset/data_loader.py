import numpy as np
import config as config
from src.utils.graphics import styled_print
import pdb
class DataLoader:
    def __init__(self, eeg, audio, stimulus):
        styled_print("🚀", "Initializing DatasetLoader...", "yellow", panel=True)

        self.eeg = eeg
        self.audio = audio
        self.stimulus = stimulus
        self._extract_events()
        self._load_trials()


    def _extract_events(self):
        onsets = self.stimulus['onset'].values
        self.eeg_indexs = np.array(self.stimulus['sample'].values)
        audio_indexs = np.array(onsets*config.TARGET_AUDIO_SR)
        self.audio_indexs = audio_indexs.astype(int)
        self.words = self.stimulus['value'].values


    def _load_trials(self):
        styled_print("📊", f"Loading Trials", "magenta")
        eeg_trials, audio_trials, word_labels = [], [], []
        eeg_trial_length = int(config.TRIAL_LENGTH * config.EEG_SR)
        audio_trial_length = int(config.TRIAL_LENGTH * config.TARGET_AUDIO_SR)
        for index in range(self.eeg_indexs.shape[0]):
            eeg_start_index = self.eeg_indexs[index]
            audio_start_index = self.audio_indexs[index]
            eeg_trials.append(
                self.eeg[eeg_start_index:eeg_start_index+eeg_trial_length]
            )
            audio_trials.append(
                self.audio[audio_start_index:audio_start_index+audio_trial_length]
            )
            word_labels.append(self.words[index])

        self.eeg_trials = np.array(eeg_trials)
        self.audio_trials = np.array(audio_trials)
        self.word_labels = np.array(word_labels)
        styled_print("📊", f"EEG trails shape: {self.eeg_trials.shape}", "magenta")
        styled_print("🔊", f"Audio trails shape: {self.audio_trials.shape}", "magenta")
        styled_print("📝", f"words Shape: {self.word_labels.shape}", "magenta")

