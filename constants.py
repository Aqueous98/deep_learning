import os
from pathlib import Path
from shutil import rmtree


def get_project_root() -> Path:
  """Returns project root folder."""
  return Path(__file__).parent


ROOT_DIR = get_project_root()
DATA_DIR = os.path.join(ROOT_DIR, 'data')

ARCTIC_DIR = os.path.join(DATA_DIR, 'Arctic')
WAVE_DIR = os.path.join(DATA_DIR, 'wav_files')
PP_DATA_DIR = os.path.join(DATA_DIR, 'preprocessed')
NOISE_DIR = os.path.join(DATA_DIR, 'noise')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')


def create_arctic_directory():
  if not os.path.exists(ARCTIC_DIR):
    os.makedirs(ARCTIC_DIR)


def create_saved_wav_directory():
  if not os.path.exists(WAVE_DIR):
    os.makedirs(WAVE_DIR)


def create_preprocessed_dataset_directories():
  try:
    os.makedirs(PP_DATA_DIR)
  except OSError as ose:
    print(ose)

  try:
    os.makedirs(os.path.join(PP_DATA_DIR, "model"))
  except OSError as ose:
    print(ose)

  try:
    os.makedirs(os.path.join(PP_DATA_DIR, "audio"))
  except OSError as ose:
    print(ose)


def clear_logs():
  if os.path.exists(LOGS_DIR):
    rmtree(LOGS_DIR)
