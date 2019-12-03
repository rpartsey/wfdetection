import torch
import yaml
import os
import copy

from config import Config
from utils.make_submission import main
from models import MODELS
from dataloaders.transformations import create_transform

DEVICE = os.getenv('DEVICE', None)
if DEVICE is None:
    raise ValueError("please specify the device in OS.ENV using "
                     "`>> DEVICE=cuda:0 python {file you running}` or doing"
                     "`export DEVICE=cuda:0\npython{file you running}`"
                     " so we can easily distribute GPUs")


if __name__ == '__main__':
    with open('example_config_make_submission.yaml', 'r') as f:
        config = yaml.load(f)

    models = []
    for m, mconfig in config['models_ensemble'].items():
        model_path = mconfig["path"]
        exp_path = os.path.dirname(model_path)
        model = Config(exp_path, DEVICE)
        model.load_model_from_path(model_path)
        models.append(model)

    dim = config['input_image_size']
    # postprocessing_config = config["postprocessing"]
    main(models, dim, appendix=config['submission_name'], config=config, v=1)
