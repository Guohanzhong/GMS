print("start data import")
from tensorflow.python.platform import flags
from imageio import imread
import tensorflow as tf
import io
import lmdb
import logging
import tqdm
import numpy
from PIL import Image
import json
from torch.utils.data import Dataset
import pickle
import os.path as osp
import os
import numpy as np
import time
from PIL import Image
#from scipy.misc import imread, imresize
#from skimage import io 
from skimage.color import rgb2gray
from torchvision.datasets import CIFAR10, MNIST, SVHN, CIFAR100, ImageFolder, LSUNClass, LSUN
from torchvision import transforms
import torch
import torchvision
import pandas as pd
from imageio import imwrite
from absl import flags
import errno
import codecs
from torch.utils import data
import random
print("end data import")


def cutout(mask_color=(0, 0, 0)):
    mask_size_half = FLAGS.cutout_mask_size // 2
    offset = 1 if FLAGS.cutout_mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > FLAGS.cutout_prob:
            return image

        h, w = image.shape[:2]

        if FLAGS.cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + FLAGS.cutout_mask_size
        ymax = ymin + FLAGS.cutout_mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[:, ymin:ymax, xmin:xmax] = np.array(mask_color)[:, None, None]
        return image

    return _cutout


class CelebAHQ(Dataset):

    def __init__(self, cond_idx=1, filter_idx=0):
        self.path = "/datasets01/celebAHQ/081318/imgHQ{:05}.npy"
        self.labels = pd.read_csv("/private/home/yilundu/list_attr_celeba.txt", sep="\s+", skiprows=1)
        self.hq_labels = pd.read_csv("/private/home/yilundu/image_list.txt", sep="\s+")
        self.cond_idx = cond_idx
        self.filter_idx = filter_idx

    def __len__(self):
        return self.hq_labels.shape[0]

    def __getitem__(self, index):
        info = self.hq_labels.iloc[index]
        info = self.labels.iloc[info.orig_idx]

        path = self.path.format(index)
        im = np.load(path)
        im = im[0].transpose((1, 2, 0))
        image_size = 128
        im = imresize(im, (image_size, image_size))
        im = im / 256
        im = im + np.random.uniform(0, 1 / 256., im.shape)

        label = int(info.iloc[self.cond_idx])
        if label == -1:
            label = 0
        label = np.eye(2)[label]

        im_corrupt = np.random.uniform(
            0, 1, size=(image_size, image_size, 3))

        return im_corrupt, im, label


class ImageNet(Dataset):

    def __init__(self, cond_idx=1, filter_idx=0):
        self.path = "/raid/common/imagenet-raw/train"
        #print(os.listdir(self.path))
        self.folders = [osp.join(self.path, d) for d in os.listdir(self.path)]

        self.images = []
        self.labels = []
        for i, folder in enumerate(self.folders):
            im_path = [osp.join(folder, im) for im in os.listdir(folder)]
            self.images.extend(im_path)
            self.labels.extend([i] * len(im_path))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        im = imread(path)
        if len(im.shape) == 2:
            im = np.tile(im[:, :, None], (1, 1, 3))
        else:
            im = im[:, :, :3]
        image_size = 64
        im = numpy.array(Image.fromarray(im).resize((image_size,image_size)))
        #im = imresize(im, (image_size, image_size))
        im = im / 256
        im = im + np.random.uniform(0, 1 / 256., im.shape)
        im = im.transpose((2, 0, 1))
        im = (im-0.5)/0.5
        #im_corrupt = np.random.uniform(0, 1, size=(image_size, image_size, 3))
        label = np.eye(1000)[self.labels[index]]
        return im,label

class LSUNBed(Dataset):

    def __init__(self, cond_idx=1, filter_idx=0):
        self.path = "/home/aiops/allanguo"
        lmdb_path = osp.join(self.path, "bedroom_train_lmdb")

        self.env = lmdb.open(lmdb_path, subdir=osp.isdir(lmdb_path),max_readers=1, readonly=True, lock=False,
                                             readahead=False, meminit=False)
        
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        root_split = lmdb_path.split("/")
        cache_file = os.path.join("/".join(root_split[:-1]), f"_cache_{root_split[-1]}")
        print('the cache file dir is {0}'.format(cache_file))
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            logging.info('Create the file to store')
            with self.env.begin(write=False) as txn:
                self.keys = []
                ans = 0
                for key,_ in txn.cursor():
                    self.keys.append(key)
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        img, target = None, torch.zeros(1)
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = np.array(Image.open(buf).convert('RGB'))
        image_size = 256
        im = numpy.array(Image.fromarray(im).resize((image_size,image_size)))
        #im = imresize(im, (image_size, image_size))
        im = im / 256
        im = im + np.random.uniform(0, 1 / 256., im.shape)
        im = im.transpose((2, 0, 1))
        im = (im-0.5)/0.5
        #im_corrupt = np.random.uniform(0, 1, size=(image_size, image_size, 3))
        return im, target



class CelebA(Dataset):

    def __init__(self, cond_idx=1, filter_idx=0):
        self.path = "/datasets01/CelebA/CelebA/072017/img_align_celeba/"
        self.labels = pd.read_csv("/private/home/yilundu/list_attr_celeba.txt", sep="\s+", skiprows=1)
        self.cond_idx = cond_idx
        self.filter_idx = filter_idx

        if filter_idx != 0:
            mask = (self.labels.to_numpy()[:, self.cond_idx] == filter_idx)
            self.labels = self.labels[mask].reset_index()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):


        if FLAGS.single:
            index = 0

        info = self.labels.iloc[index]
        if self.filter_idx != 0:
            fname = info['index']
        else:
            fname = info.name
        path = osp.join(self.path, fname)
        im = imread(path)
        im = imresize(im, (128, 128))
        image_size = 128
        im = im / 255.

        label = int(info.iloc[self.cond_idx])
        if label == -1:
            label = 0
        label = np.eye(2)[label]

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = np.random.uniform(
                0, 1, size=(image_size, image_size, 3))

        return im_corrupt, im, label

class Cifar10(Dataset):
    def __init__(
            self,
            FLAGS,
            train=True,
            full=False,
            augment=False,
            noise=True,
            rescale=1.0):

        if augment:
            transform_list = [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ]

            transform = transforms.Compose(transform_list)
        else:
            transform = transforms.ToTensor()

        self.full = full
        self.data = CIFAR10(
            "data/cifar10",
            transform=transform,
            train=train,
            download=True)
        self.test_data = CIFAR10(
            "data/cifar10",
            transform=transform,
            train=False,
            download=True)
        self.one_hot_map = np.eye(10)
        self.noise = noise
        self.rescale = rescale
        self.FLAGS = FLAGS

    def __len__(self):

        if self.full:
            return len(self.data) + len(self.test_data)
        else:
            return len(self.data)

    def __getitem__(self, index):
        FLAGS = self.FLAGS
        if self.full:
            if index >= len(self.data):
                im, label = self.test_data[index - len(self.data)]
            else:
                im, label = self.data[index]
        else:
            im, label = self.data[index]

        im = np.transpose(im, (1, 2, 0)).numpy()
        image_size = 32
        label = self.one_hot_map[label]

        im = im * 255 / 256

        im = im * self.rescale + \
            np.random.uniform(0, 1 / 256., im.shape)

        # np.random.seed((index + int(time.time() * 1e7)) % 2**32)

        im_corrupt = np.random.uniform(
            0.0, self.rescale, (image_size, image_size, 3))

        return torch.Tensor(im_corrupt), torch.Tensor(im), label


class Cifar100(Dataset):
    def __init__(self, FLAGS, train=True, augment=False):

        transform = transforms.ToTensor()
        self.one_hot_map = np.eye(100)

        self.data = CIFAR100(
            "/tmp/cifar100",
            transform=transform,
            train=train,
            download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        im, label = self.data[0]

        im = np.transpose(im, (1, 2, 0)).numpy()
        image_size = 32
        label = self.one_hot_map[label]
        im = im * 255 / 256

        im = im + \
            np.random.uniform(0, 1 / 256., im.shape)

        im_corrupt = np.random.uniform(
            0.0, 1.0, (image_size, image_size, 3))

        return im_corrupt, im, label

class Mnist(Dataset):
    def __init__(self, train=True, rescale=1.0):
        self.data = MNIST(
            "data/mnist",
            transform=transforms.ToTensor(),
            download=True, train=train)
        self.labels = np.eye(10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        im, label = self.data[index]
        label = self.labels[label]
        im = im.squeeze()
        im = im.numpy() / 256 * 255 + np.random.uniform(0, 1. / 256, (28, 28))
        im = np.clip(im, 0, 1)
        s = 28
        im_corrupt = np.random.uniform(0, 1, (s, s, 1))
        im = im[:, :, None]

        return torch.Tensor(im_corrupt), torch.Tensor(im), label


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x

if __name__ == "__main__":
    dataset = LSUNBed()
    #data = dataset[100000]
    print(len(dataset))
    #img = data[0]
    #print(img.shape)
    #img = (img+1)/2
    #image = Image.fromarray(np.uint8(img*255)).convert("RGB")
    #image.save("bed.png")

    #dataloader = torch.utils.data.DataLoader(
    #    dataset, 256, shuffle=True,
    #    num_workers=0, drop_last=True)
    #datalooper = infiniteloop(dataloader)
    #print(next(datalooper).size())


    #data = pd.read_csv('sample/likelihood/ddpm1000.csv')
    #print(data)
    #lsun_data = LSUN('bedroom', 'train')
    #torchvision.datasets.LSUN(classes='train', transform=None, target_transform=None)