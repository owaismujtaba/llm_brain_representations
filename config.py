import os
from pathlib import Path

CUR_DIR = os.getcwd()
DATA_DIR = Path(CUR_DIR, 'data', 'Data')


AUIDO_SR = 48000
TARGET_AUDIO_SR = 16000
EEG_SR = 1024

TRIAL_LENGTH = 1 # seconds