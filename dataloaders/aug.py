import albumentations as albu


def get_transforms(config):
    size = config['size']
    scope = config['scope']
    crop = config['crop']
    p = config['p']

    augs = {'strong': albu.Compose([albu.HorizontalFlip(),
                                    albu.VerticalFlip(),
                                    albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=20, p=.4),
                                    albu.ElasticTransform(),
                                    albu.OpticalDistortion(),
                                    albu.OneOf([
                                        albu.CLAHE(clip_limit=2),
                                        albu.IAASharpen(),
                                        albu.IAAEmboss(),
                                        albu.RandomBrightnessContrast(),
                                        albu.RandomGamma()
                                    ], p=0.5),
                                    albu.OneOf([
                                        albu.RGBShift(),
                                        albu.HueSaturationValue(),
                                    ], p=0.5),
                                    ]),
            'weak': albu.Compose([albu.HorizontalFlip(),
                                  ]),
            'geometric': albu.OneOf([albu.HorizontalFlip(p=p),
                                     albu.VerticalFlip(p=p),
                                     albu.ShiftScaleRotate(p=p),
                                     albu.Transpose(p=p),
                                     # albu.OpticalDistortion(p=p),
                                     # albu.ElasticTransform(p=p),
                                     ])
            }

    aug_fn = augs[scope]
    crop_fn = {'random': albu.RandomCrop(size[0], size[1], always_apply=True),
               'center': albu.CenterCrop(size[0], size[1], always_apply=True), 'None': None}[crop]
    pad = albu.PadIfNeeded(size[0], size[1])

    pipeline = albu.Compose([aug_fn, crop_fn, pad])

    def process(a, b):
        r = pipeline(image=a, mask=b)
        return r['image'], r['mask']

    return process


def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process


def _resolve_aug_fn(name):
    d = {
        'cutout': albu.Cutout,
        'rgb_shift': albu.RGBShift,
        'hsv_shift': albu.HueSaturationValue,
        'motion_blur': albu.MotionBlur,
        'median_blur': albu.MedianBlur,
        'snow': albu.RandomSnow,
        'shadow': albu.RandomShadow,
        'fog': albu.RandomFog,
        'brightness_contrast': albu.RandomBrightnessContrast,
        'gamma': albu.RandomGamma,
        'sun_flare': albu.RandomSunFlare,
        'sharpen': albu.IAASharpen,
        'jpeg': albu.JpegCompression,
        'gray': albu.ToGray,
        # ToDo: pixelize
        # ToDo: partial gray
    }
    return d[name]


def get_corrupt_function(config):
    augs = []
    for aug_params in config:
        name = aug_params.pop('name')
        cls = _resolve_aug_fn(name)
        prob = aug_params.pop('prob') if 'prob' in aug_params else .5
        augs.append(cls(p=prob, **aug_params))

    augs = albu.OneOf(augs)

    def process(x):
        return augs(image=x)['image']

    return process
