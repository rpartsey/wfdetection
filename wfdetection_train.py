import os
from config import Config
from trainer import Trainer

CONFIG_PATH = 'wfdetection/wfdetection_config.yaml'
DEVICE = 'cpu'
STAGE_NUMBER = 1


config = Config.create_from_path(CONFIG_PATH, DEVICE)


trainer = Trainer(config, STAGE_NUMBER)
trainer.train()
