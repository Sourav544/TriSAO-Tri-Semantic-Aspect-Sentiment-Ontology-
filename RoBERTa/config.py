import torch

class Config:
    MODEL_NAME = "roberta-base"
    MAX_LENGTH = 192
    NUM_LABELS = 28
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 40
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LAYER_WISE_LR_DECAY = 0.8
    FREEZE_EPOCHS = 0           # freeze backbone this many epochs
    WARMUP_RATIO = 0.1
    ASL_GAMMA_POS = 0.0
    ASL_GAMMA_NEG = 2.5
    ASL_CLIP = 0.02
    CONTRASTIVE_WEIGHT = 0.0   # set >0 to enable supervised contrastive auxiliary loss
    CONTRASTIVE_TAU = 0.3
    SAVE_DIR = "./best_goemotions_model"
    RESULTS_DIR = "./best_results"

    EMOTION_LABELS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]