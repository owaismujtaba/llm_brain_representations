import numpy as np
import pandas as pd
from pathlib import Path
import scipy
from pynwb import NWBHDF5IO
import warnings

import config as config
from src.utils.graphics import styled_print

warnings.filterwarnings("ignore")
import pdb

class DatasetReader:
    def __init__(self, sub_id='01'):
        styled_print("🚀", "Initializing DatasetReader...", "yellow", panel=True)
        styled_print("🧑‍⚕️", f"Loading Data for Subject: {sub_id}", "cyan")

        self.sub_id = f'sub-{sub_id}'
        self.filename = f'{self.sub_id}_task-wordProduction'
        self.data_dir = Path(
            config.DATA_DIR, self.sub_id, 'ieeg'
        )
        
        styled_print("📂", f"   Looking for file: {self.filename}", "blue")
        self._read_data()
        self.preprocess_audio()
        self.preprocess_eeg()

    @staticmethod
    def hilbert_transform(x):
        return scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)), axis=0)[:len(x)]

    def preprocess_eeg(self):
        styled_print("📂", " Preprocessing EEG...", "blue")
        eeg = scipy.signal.detrend(self.eeg, axis=0)
       
        sos = scipy.signal.iirfilter(4, [70 / (config.EEG_SR / 2), 170 / (config.EEG_SR / 2)], btype='bandpass', output='sos')
        eeg = scipy.signal.sosfiltfilt(sos, eeg, axis=0)

        for freq in [100, 150]:
            sos = scipy.signal.iirfilter(4, [(freq - 2) / (config.EEG_SR / 2), (freq + 2) / (config.EEG_SR / 2)],
                                         btype='bandstop', output='sos')
            eeg = scipy.signal.sosfiltfilt(sos, eeg, axis=0)
        eeg = np.abs(self.hilbert_transform(eeg))
        self.eeg = eeg
    def _read_data(self):
        styled_print("📡", "Reading NWB Data...", "yellow", panel=True)
        
        try:
            filepath = Path(self.data_dir,  f'{self.filename}_ieeg.nwb')
            io = NWBHDF5IO(filepath)
            file = io.read()
            
            self.eeg = file.acquisition['iEEG'].data[:]
            self.audio = file.acquisition['Audio'].data[:]
            self.words = file.acquisition['Stimulus'].data[:]

            filepath = Path(self.data_dir, f'{self.filename}_events.tsv')
            self.stimulus = pd.read_csv(filepath, sep='\t')
            self.stimulus = self.stimulus[self.stimulus['trial_type']=='word']
            
            styled_print("✅", "Data successfully loaded!", "green")
           
        except Exception as e:
            styled_print("❌", f"Error reading data: {e}", "red")

    def preprocess_audio(self):
        styled_print("📂", " Preprocessing Audio...", "blue")

        try:
            styled_print("🔄", " Downsampling audio...", "yellow")
            audio = scipy.signal.decimate(
                self.audio, 
                int(config.AUIDO_SR / config.TARGET_AUDIO_SR)
            )

            styled_print("🎚️", " Normalizing audio...", "yellow")
            audio = np.int16(audio / np.max(np.abs(audio)) * 32767)

            self.audio = audio
            styled_print("✅", " Audio preprocessing complete!", "green")

        except Exception as e:
            styled_print("❌", f" Error in preprocessing: {e}", "red")
