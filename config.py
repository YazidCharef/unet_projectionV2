import torch
import os
from datetime import datetime

# Chemins des données
NC_FILE = "/home/yazid/Documents/stage_cambridge/project_1/Pacific_Pressure_750.nc"
SST_FILE = "data/sst_data_1979_2021.nc"
TYPHOON_POSITIONS_CSV = "data/typhoon_data_reordered.csv"
TYPHOON_PHASES_CSV = "data/typhoon_data_Cyclogenesis_Identification_processed.csv"

INPUT_CHANNELS = 5  # u, v, r, vo, sst
SEQUENCE_LENGTH = 9
TIME_STEP = 6  

CYCLOGENESIS_RADIUS = 200
TYPHOON_RADIUS = 400
CYCLOLYSIS_RADIUS = 200

BATCH_SIZE = 8 * torch.cuda.device_count()  
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count()

MODEL_DIR = "models_trained"
os.makedirs(MODEL_DIR, exist_ok=True)

LOG_EVERY = 100  # 
SAVE_EVERY = 5   

USE_PIXEL_ACCURACY = True
USE_IOU = True
USE_F1 = True

print(f"Configuration initialized:")
print(f"- Using device: {DEVICE}")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Input channels: {INPUT_CHANNELS}")