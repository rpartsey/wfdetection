import os
from config import Config
from trainer import Trainer

CONFIG_PATH = 'wfdetection/wfdetection_config.yaml'

DEVICE = os.getenv('DEVICE', 'cuda:0')

if DEVICE is None:
    raise ValueError("please specify the device in OS.ENV using "
                     "`>> DEVICE=cuda:0 python {file you running}` or doing"
                     "`export DEVICE=cuda:0\npython{file you running}`"
                     " so we can easily distribute GPUs")

STAGE_NUMBER = 1


config = Config.create_from_path(CONFIG_PATH, DEVICE)


trainer = Trainer(config, STAGE_NUMBER)
trainer.train()
