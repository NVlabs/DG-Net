"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torchvision import datasets
import os
import numpy as np
import random

class ReIDFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(ReIDFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets
        self.img_num = len(self.samples)
        print(self.img_num)

    def _get_cam_id(self, path):
        camera_id = []
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        return int(camera_id)-1

    def _get_pos_sample(self, target, index, path):
        pos_index = np.argwhere(self.targets == target)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        if len(pos_index)==0:  # in the query set, only one sample
            return path
        else:
            rand = random.randint(0,len(pos_index)-1)
        return self.samples[pos_index[rand]][0]

    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0,len(neg_index)-1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        pos_path = self._get_pos_sample(target, index, path)
        pos = self.loader(pos_path)

        if self.transform is not None:
            sample = self.transform(sample)
            pos = self.transform(pos)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, pos

