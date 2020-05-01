import os
from pathlib import Path


def get_project_root() -> Path:
  """Returns project root folder."""
  return Path(__file__).parent


ROOT_DIR = get_project_root()
DATA_DIR = os.path.join(ROOT_DIR, 'data')

ARCTIC_DIR = os.path.join(DATA_DIR, 'Arctic')
WAVE_DIR = os.path.join(DATA_DIR, 'wav_files')


def create_arctic_directory():
  if not os.path.exists(ARCTIC_DIR):
    os.makedirs(ARCTIC_DIR)


def create_saved_wav_directory():
  if not os.path.exists(WAVE_DIR):
    os.makedirs(WAVE_DIR)