import torch
import torch.utils.data
import torchvision

import numpy as np


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        super().__init__(dataset)
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            return int(dataset.mask_exists(idx))

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class BalanceClassSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, length=None):
        super().__init__(dataset)
        self.dataset = dataset
        print("Balanced sampler", length)

        if length is None:
            length = len(self.dataset)

        self.length = length
        self.labels = np.array([self.dataset.mask_exists(idx) for idx in range(len(dataset))])

    def __iter__(self):
        pos_index = np.where(self.labels == 1)[0]
        neg_index = np.where(self.labels == 0)[0]
        half = self.length // 2 + 1
        pos = np.random.choice(pos_index, half, replace=True)
        neg = np.random.choice(neg_index, half, replace=True)
        l = np.stack([pos, neg]).T
        l = l.reshape(-1)
        l = l[:self.length]
        return iter(l)

    def __len__(self):
        return self.length
