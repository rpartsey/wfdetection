from .unet import UNet as old_unet
from .torchhub_unet import UNet as torch_unet
from .smp_unet import SmpUnet
from .r2_unet import R2AttU_Net

MODELS = {
    "old_unet": old_unet,
    "torch_unet": torch_unet,
    "smp_unet": SmpUnet,
    "r2unet": R2AttU_Net,
}
