import os
from config import Config
from trainer import Trainer
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--config_path',
                        help='config path')

DEVICE = os.getenv('DEVICE', None)
if DEVICE is None:
    raise ValueError("please specify the device in OS.ENV using "
                     "`>> DEVICE=cuda:0 python {file you running}` or doing"
                     "`export DEVICE=cuda:0\npython{file you running}`"
                     " so we can easily distribute GPUs")

# def get_freer_gpu():
#     os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#     memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
#     os.remove('tmp')
#     return np.argmax(memory_available)

args = parser.parse_args()
config_path = args.config_path #'example_config.yaml'


print("Device:", DEVICE)
print("Config:", config_path)
config = Config.create_from_path(config_path, DEVICE)
stage_number = 1

trainer = Trainer(config, stage_number)
trainer.train()
