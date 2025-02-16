import os
from pathlib import Path

CUR_DIR = os.getcwd()
DATA_DIR = Path(CUR_DIR, 'data', 'Data')
TRAINED_DIR = Path(CUR_DIR, 'Trained')

AUIDO_SR = 48000
TARGET_AUDIO_SR = 16000
EEG_SR = 1024

TRIAL_LENGTH = 1 # seconds


TRAIN_CHATGPT_BASED = True
TRAIN_FASTTEXT_BASED = False