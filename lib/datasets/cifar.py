#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import os
import sys
import pickle

import torchvision.datasets as datasets

from ..utils.loggers import STDLogger as logger

class CIFAR10(datasets.CIFAR10):

    def __init__(self, root, split='train',
                transform=None, target_transform=None, download=False):
        assert split.lower() in ['train', 'test'], "CIFAR10's split should be one of train/test."
        self.split = split.lower()
        super(CIFAR10, self).__init__(root, self.split == 'train', transform, 
                                        target_transform, download)

    def download(self):
        """
        Override this function to use default logger instead of standard
        ``print'' function for logging
        """
        import tarfile

        root = os.path.expanduser(self.root)
        filename = os.path.basename(self.url) if not self.filename else self.filename
        fpath = os.path.join(root, filename)
        if self._check_integrity():
            logger.debug('Using downloaded and verified file: ' + fpath)
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)

class CIFAR100(CIFAR10):

    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, *args, **kwargs):
        super(CIFAR100, self).__init__(*args, **kwargs)

        # load coarse labels
        downloaded_list = self.train_list if self.train else self.test_list
        self.coarse_labels = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.coarse_labels.extend(entry['coarse_labels'])
        # use 20 super-classes as ground-truth
        self.targets = self.coarse_labels


class tinyimagenet(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, is_valid_file=None):
        super(tinyimagenet, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples