import os
import logging
import numpy as np
from PIL import Image
from deeppy.dataset.util import dataset_home, download, checksum, archive_extract, checkpoint


log = logging.getLogger(__name__)


_ALIGNED_IMGS_URL = (
    'https://www.dropbox.com/s/ef5drisi4lb650s/img_align_celeba.zip?dl=1',
    '064a81ef5ee9f49b78016bc83d4838f09d2dbbb2'
)

_PARTITIONS_URL = (
    'https://www.dropbox.com/s/snp6qyulx3lo2xl/list_eval_partition.txt?dl=1',
    '0940bfa1a328323cd3586bfd100c49f9886bba24'
)

_ATTRIBUTES_URL = (
    'https://www.dropbox.com/s/s8bcdztm912zpn2/list_attr_celeba.txt?dl=1',
    'b1715f384deed1b2a33c6abb982441a569c824c2'
)


class CelebA(object):
    '''
    Large-scale CelebFaces Attributes (CelebA) Dataset [1].
    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    References:
    [1]: Ziwei Liu, Ping Luo, Xiaogang Wang and Xiaoou Tang.
         Deep Learning Face Attributes in the Wild. Proceedings of
         International Conference on Computer Vision (ICCV), December, 2015.
    '''

    def __init__(self):
        self.name = 'celeba'
        self.n_imgs = 202599
        self.data_dir = os.path.join(dataset_home, self.name)
        self._npz_path = os.path.join(self.data_dir, self.name+'.npz')
        self.img_dir = os.path.join(self.data_dir, 'img_align_celeba')
        self._install()
        (self.train_idxs, self.val_idxs, self.test_idxs, self.attribute_names,
         self.attributes) = self._load()

    def _download(self, url, sha1):
        log.info('Downloading %s', url)
        filepath = download(url, self.data_dir)
        if sha1 != checksum(filepath):
            raise RuntimeError('Checksum mismatch for %s.' % url)
        return filepath

    def img(self, idx):
        img_path = os.path.join(self.img_dir, '%.6d.jpg' % (idx+1))
        return np.array(Image.open(img_path))

    def imgs(self):
        for i in range(self.n_imgs):
            yield self.img(i)

    def _install(self):
        checkpoint_file = os.path.join(self.data_dir, '__install_check')
        with checkpoint(checkpoint_file) as exists:
            if exists:
                log.info('Skipping install, data already installed')
                return
            url, md5 = _ALIGNED_IMGS_URL
            filepath = self._download(url, md5)
            log.info('Unpacking %s', filepath)
            archive_extract(filepath, self.data_dir)

            url, md5 = _PARTITIONS_URL
            filepath = self._download(url, md5)
            partitions = [[], [], []]
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    img_name, partition = line.strip().split(' ')
                    if int(img_name[:6]) != i + 1:
                        raise ValueError('Parse error.')
                    partition = int(partition)
                    partitions[partition].append(i)
            train_idxs, val_idxs, test_idxs = map(np.array, partitions)

            url, md5 = _ATTRIBUTES_URL
            filepath = self._download(url, md5)
            attributes = []
            with open(filepath, 'r') as f:
                f.readline()
                attribute_names = f.readline().strip().split(' ')
                for i, line in enumerate(f):
                    fields = line.strip().replace('  ', ' ').split(' ')
                    img_name = fields[0]
                    if int(img_name[:6]) != i + 1:
                        raise ValueError('Parse error.')
                    attr_vec = np.array(map(int, fields[1:]))
                    attributes.append(attr_vec)
            attributes = np.array(attributes)

            with open(self._npz_path, 'wb') as f:
                np.savez(f, train_idxs=train_idxs, val_idxs=val_idxs,
                         test_idxs=test_idxs, attribute_names=attribute_names,
                         attributes=attributes)

    def _load(self):
        with open(self._npz_path, 'rb') as f:
            dic = np.load(f)
            return (dic['train_idxs'], dic['val_idxs'], dic['test_idxs'],
                    dic['attribute_names'][()], dic['attributes'])