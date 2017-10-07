'''
Dataset comprised of all images in a folder.
Work in progress
'''

import os
import logging
import numpy as np
from PIL import Image
from .util import dataset_home, download, checksum, archive_extract, checkpoint


log = logging.getLogger(__name__)

class DatasetFromDir(object):
    def __init__(self, directory):
        self.name = directory
        filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        self.n_imgs = len(filenames)
        self.directory = directory
        (self.train_idxs, self.val_idxs, self.test_idxs, self.attribute_names,
         self.attributes) = self._load()

    def img(self, idx):
        img_path = os.path.join(self.img_dir, '%.6d.jpg' % (idx+1))
        return np.array(Image.open(img_path))

    def imgs(self):
        for i in range(self.n_imgs):
            yield self.img(i)

    def _load(self):
        with open(self._npz_path, 'rb') as f:
            dic = np.load(f)
            return (dic['train_idxs'], dic['val_idxs'], dic['test_idxs'],
                    dic['attribute_names'][()], dic['attributes'])