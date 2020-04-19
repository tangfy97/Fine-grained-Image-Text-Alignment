# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import os.path as osp
import phrasemachine
#在test文件夹里面缺失的文件列表
#File disappeared in test image folder.

class PrecompDataset(data.Dataset):

    def __init__(self, dataset_name, voca):
        self.vocab = voca
        # feature文件路径改成自己的存放的路径
        # 存放文各个文件图像的caption文件集合，改成自己存放路径
        list = os.listdir(os.path.join(dataset_name,"phrase_train"))
        # Captions
        self.captions = []
        for text in list:
            # 同样的改成自己的路径
            with open(os.path.join(dataset_name+"/phrase_train", text), "r") as f:
                 self.captions.append(f.readline())

        # Image features
        #feature文件路径改成自己的存放的路径
        self.images = np.load(os.path.join(dataset_name,"train_phrase.npy"))
        self.length = len(self.captions)

    def __getitem__(self, index):
        image = torch.tensor(self.images[index])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        #tokens = nltk.tokenize.word_tokenize(
        #    str(caption).lower())
        # 改成用phrasemachine 提取phrase
        # use phrasemachine to extract noun phrase instead of tokenizing them.
        tokens = list(phrasemachine.get_phrases(caption)["counts"])
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.tensor(caption)
        return image, target, index

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, vocab)
    print(len(dset.captions))
    print(len(dset.images))
    print(len(dset.vocab))
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              # drop_last=True,
                                              )
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    train_loader = get_precomp_loader(data_name, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(data_name, 'test', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    test_loader = get_precomp_loader(data_name, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
