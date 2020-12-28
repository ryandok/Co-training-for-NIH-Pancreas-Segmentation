from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import numpy as np
import torch.nn as nn
import torch
import random
import utils
from scipy.ndimage import rotate


class CANDI(Dataset):
    def __init__(self, vol_path, seg_path, transform=None):
        self.vol_path = vol_path
        self.seg_path = seg_path
        self.transform = transform

    def __getitem__(self, index):
        volume_path = self.vol_path[index]
        label_path = self.seg_path[index]

        volume = nib.load(volume_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        name = volume_path.split('/')[-1].split('.')[0]
        sample = {'name': name, 'volume': volume, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.vol_path)


class Rotate(object):
    def __init__(self, rotate_rate):
        self.rotate_rate = rotate_rate

    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']
        if np.random.random() < self.rotate_rate:
            rand_angle = [90, 180, 270, 360]
            np.random.shuffle(rand_angle)
            volume = rotate(volume, angle=rand_angle[0], axes=(1, 0), reshape=False, order=1)
            label = rotate(label, angle=rand_angle[0], axes=(1, 0), reshape=False, order=0)
        sample = {'name': name, 'volume': volume, 'label': label}

        return sample


class Flip(object):
    def __init__(self, flip_rate):
        self.flip_rate = flip_rate

    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']
        if np.random.random() < self.flip_rate:
            volume = volume[:, :, ::-1]
            label = label[:, :, ::-1]
        sample = {'name': name, 'volume': volume, 'label': label}
        return sample


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.005):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(volume.shape[0], volume.shape[1], volume.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        volume = volume + noise
        sample = {'name': name, 'volume': volume, 'label': label}
        return sample


class ToTensor(object):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']
        volume = torch.Tensor(np.expand_dims(volume, axis=0).copy())
        label = torch.Tensor(np.expand_dims(label, axis=0).copy())
        sample = {'name': name, 'volume': volume, 'label': label}
        return sample


if __name__ == "__main__":
    from torchvision import transforms
    from utils import util
    from torch.utils.data import DataLoader
    root_path = '/data1/hra'

    total_volume_path, total_label_path = util.gen_CANDI_data_path(root_path)

    train_volume_path, train_label_path, valid_volume_path, valid_label_path \
        = util.divide_data2train_valid(total_volume_path, total_label_path, 60, 1124)

    dataset = CANDI(train_volume_path, train_volume_path,
                       transform=transforms.Compose([RandomNoise(), Rotate(0.5), Flip(0.5), ToTensor()]))
    def worker_init_fn(worker_id):
        random.seed(worker_id)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for epoch_num in range(2):
        print('epoch_num=%d' % epoch_num)
        for i_batch, batch_sample_i in enumerate(dataloader):
            print(i_batch)
            print(batch_sample_i['name'])