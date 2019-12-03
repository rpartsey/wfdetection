import os
from config import Config
from gan_trainer import GanTrainer
from gan_config import Config as GanConfig

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

config_path = 'configs/gan_focal_config_sampler_1024tune.yaml' #'example_config.yaml'


print("Device:", DEVICE)
print("Config:", config_path)
config = Config.create_from_path(config_path, DEVICE)
stage_number = 1
gan_config = GanConfig(config)
gan_config.load_model_from_path("experiments/gan_smp_unet_sampler_1024/desc_True_False_0.h5")
trainer = GanTrainer(config, gan_config, stage_number, train_descriminator=False, train_generator=True)
trainer.train()
