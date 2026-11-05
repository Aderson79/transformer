import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN = 32
BATCH_SIZE = 16
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
EPOCHS = 20
LR = 1e-4
