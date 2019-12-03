import torch
import yaml
import os
import copy
import argparse

from tqdm import tqdm, trange
import numpy as np

from config import Config

DEVICE = os.getenv('DEVICE', None)
if DEVICE is None:
    raise ValueError("please specify the device in OS.ENV using "
                     "`>> DEVICE=cuda:0 python {file you running}` or doing"
                     "`export DEVICE=cuda:0\npython{file you running}`"
                     " so we can easily distribute GPUs")


def save_validation(basename, config, flip=False):
    if flip:
        print('CAUTION USING FLIP\n'*10)
    os.makedirs("model_preds", exist_ok=True)
    data, tensor_data = config.load_datasets()
    # validation_data = config.dataloaders(None, data["validation"])
    validation_data = data["validation"]
    test_probability = []
    test_truth = []
    image_ids = []
    config.model.eval()
    t = trange(len(validation_data), desc='Calc metr.')

    for i in t:
        batch = validation_data[i]
        imgs, masks, _, _ = batch
        imgs, masks = imgs.unsqueeze(0), masks.unsqueeze(0)
        imgs, masks = imgs.to(DEVICE), masks.float()
        img_id = validation_data.image_id(i)
        if flip:
            imgs_f = torch.from_numpy(imgs.cpu().numpy()[:, :, :, ::-1].copy()).float().to(DEVICE)

        with torch.no_grad():
            logits = config.model.forward(imgs)
            if logits.shape[1] != 1:
                raise ValueError("Only SIGMOID supported")
            preds = torch.sigmoid(logits)
            if flip:
                preds_f = torch.from_numpy(config.model.forward(imgs_f).sigmoid().cpu().numpy()[:, :, :, ::-1].copy()).to(DEVICE)
                preds = torch.stack([preds, preds_f]).mean(0)

            test_probability.append(preds.detach().cpu().numpy().astype(np.float32))
            test_truth.append(masks.numpy().astype(np.float32))
            image_ids.append(img_id)

    test_truth = np.concatenate(test_truth)
    test_probability = np.concatenate(test_probability)
    image_ids = np.array(image_ids)

    truth_file_path = "model_preds/{}_truth.npy".format(basename)
    probability_file_path = "model_preds/{}_probability.npy".format(basename)
    image_ids_file_path = "model_preds/{}_ids.npy".format(basename)
    np.save(truth_file_path, test_truth)
    np.save(probability_file_path, test_probability)
    np.save(image_ids_file_path, image_ids)
    print("Saved into", truth_file_path)
    print("Saved into", probability_file_path)
    print("Saved into", image_ids_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path',
                        help='model path')
    parser.add_argument('--basename', default="validation",
                        help='model path')

    args = parser.parse_args()
    model_path = args.model_path  # 'example_config.yaml'
    basename = args.basename
    print("Model", model_path)
    print("Basename", basename)

    exp_path = os.path.dirname(model_path)
    model = Config(exp_path, DEVICE)
    model.load_model_from_path(model_path)
    save_validation(basename, model, flip=True)
