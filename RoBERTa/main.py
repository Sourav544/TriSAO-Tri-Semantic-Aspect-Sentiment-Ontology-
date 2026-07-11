import warnings
from config import Config
from trainer import Trainer

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.full_train()