"""

"""
import torch
import torch.utils.data as data
import cv2
from collections import OrderedDict, defaultdict
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import random
import pandas as pd
import numpy as np
import math
import os
import functools
import time
import mytransforms
import utils
import re

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']


def find_inputs(folder, types=IMG_EXTENSIONS):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                parts = base.split('-')
                assert len(parts) >= 2
                prod_id = int(parts[0])
                img_num = int(parts[1])
                filename = os.path.join(root, rel_filename)
                inputs.append((prod_id, img_num, filename))
    return inputs


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_test_aug(factor):
    if not factor or factor == 1:
        return [
            [False, False, False]]
    elif factor == 4:
        # transpose, v-flip, h-flip
        return [
            [False, False, False],
            [False, False, True],
            [False, True, False],
            [True, True, True]]
    elif factor == 8:
        # return list of all combinations of flips and transpose
        return ((1 & np.arange(0, 8)[:, np.newaxis] // 2**np.arange(2, -1, -1)) > 0).tolist()
    else:
        print('Invalid augmentation factor')
        return [
            [False, False, False]]


def dataset_scan(
        dataset_root,
        metadata_file='prod_metadata.csv',
        category_file='category_map.csv',
        fold=0,
        splits=('train', 'eval')):

    if 'test' in splits:
        split_base = 'test'
        assert len(splits) == 1
        has_labels = False
    else:
        split_base = 'train'  # train and eval share same base
        has_labels = True

    input_root = os.path.join(dataset_root, split_base)
    inputs = find_inputs(input_root, types=['.jpg'])
    if len(inputs) == 0:
        raise (RuntimeError("Found 0 images in : " + input_root))
    inputs_set = {prod_id for prod_id, _, _ in inputs}
    print(len(inputs_set))

    if not os.path.isfile(category_file):
        category_file = os.path.join(dataset_root, category_file)
    category_df = pd.read_csv(category_file)
    category_to_label1 = dict(zip(category_df.category_id, category_df.level1_label))
    category_to_label2 = dict(zip(category_df.category_id, category_df.level2_label))
    category_to_label3 = dict(zip(category_df.category_id, category_df.category_label))

    def _setup_dataset(_df):
        print(len(_df.index))
        filter_inputs = [t for t in inputs if t[0] in _df.index]
        print(len(filter_inputs))
        filtered_targets = dict(zip(_df.index, _df.category_id))
        print(len(filtered_targets))
        return filter_inputs, filtered_targets

    if has_labels:
        if not os.path.isfile(metadata_file):
            metadata_file = os.path.join(input_root, metadata_file)
        target_df = pd.read_csv(metadata_file)
        target_df.set_index(['prod_id'], inplace=True)
        target_df = target_df[target_df.index.isin(inputs_set)]

    output = []
    for s in splits:
        bootstrap = {}
        if has_labels:
            if s == 'train':
                target_df_working = target_df[target_df['cv'] != fold]
            else:
                target_df_working = target_df[target_df['cv'] == fold]
            processed_inputs, processed_targets = _setup_dataset(target_df_working)
            bootstrap['inputs'] = processed_inputs
            bootstrap['targets'] = processed_targets
        else:
            bootstrap['inputs'] = sorted(inputs, key=lambda x: (x[0], x[1]))
        bootstrap['category_to_label1'] = category_to_label1
        bootstrap['category_to_label2'] = category_to_label2
        bootstrap['category_to_label3'] = category_to_label3
        output.append(bootstrap)

    return output[0] if len(output) == 1 else output


class CDiscountDataset(data.Dataset):
    def __init__(
            self,
            root='',
            split='train',
            metadata_file='prod_metadata.csv',
            category_file='category_map.csv',
            is_training=False,
            fold=0,
            img_size=(180, 180),
            normalize='torchvision',
            test_aug=0,
            transform=None,
            bootstrap=None,
            multi_target=0):

        if bootstrap is None:
            assert os.path.exists(root)
            splits = (split,)
            bootstrap = dataset_scan(
                root, metadata_file, category_file, fold, splits=splits)

        self.category_to_label1 = bootstrap['category_to_label1']
        self.category_to_label2 = bootstrap['category_to_label2']
        self.category_to_label3 = bootstrap['category_to_label3']
        self.label3_to_category = dict(zip(
            self.category_to_label3.values(), self.category_to_label3.keys()))
        self.inputs = bootstrap['inputs']
        if 'targets' in bootstrap:
            self.targets = bootstrap['targets']
        else:
            self.targets = None
            assert not is_training

        self.is_training = is_training
        self.img_size = img_size
        self.crop_factor = 1.0 if img_size[0] == 180 else 0.95
        self.multi_target = multi_target

        if not is_training:
            self.test_aug = get_test_aug(test_aug)
        else:
            self.test_aug = []

        if transform is None:
            tfs = [transforms.ToTensor()]
            #if self.is_training:
            #    tfs.append(mytransforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05))
            if normalize == 'le':
                tfs.append(mytransforms.LeNormalize())
            else:
                normalize = transforms.Normalize(
                    mean=[124 / 255, 117 / 255, 104 / 255],
                    std=[1 / (.0167 * 255)] * 3)
                tfs.append(normalize)
            self.transform = transforms.Compose(tfs)

    def _load_input(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _random_crop_and_transform(self, input_img, scale_range=(1.0, 1.0), rot=0.0):
        angle = 0.
        hflip = random.random() < 0.5
        vflip = False  # random.random() < 0.5
        trans = False  # random.random() < 0.25
        do_rotate = (rot > 0 and random.random() < 0.25)
        h, w = input_img.shape[:2]

        # Favour rotation/scale choices that involve cropping within image bounds
        attempts = 0
        while attempts < 3:
            if do_rotate:
                angle = random.uniform(-rot, rot)
            scale = random.uniform(*scale_range)
            crop_w, crop_h = utils.calc_crop_size(self.img_size[0], self.img_size[1], angle, scale)
            if crop_w <= w and crop_h <= h:
                break
            attempts += 1

        if crop_w > w or crop_h > h:
            # We can still handle crops larger than the source, add a border
            angle = 0.0
            border_w = crop_w - w
            border_h = crop_h - h
            input_img = cv2.copyMakeBorder(
                input_img,
                border_h//2, border_h - border_h//2,
                border_w//2, border_w - border_w//2,
                cv2.BORDER_REFLECT_101)
            #print('cropl', crop_w, crop_h, border_w, border_h)
            input_img = np.ascontiguousarray(input_img)  # trying to hunt a pytorch/cuda crash, is was this necessary?
            assert input_img.shape[:2] == (crop_h, crop_w)
        else:
            hd = max(0, h - crop_h)
            wd = max(0, w - crop_w)
            ho = np.random.binomial(hd, 0.5) - math.ceil(hd / 2)
            wo = np.random.binomial(wd, 0.5) - math.ceil(wd / 2)
            cx = w // 2 + wo
            cy = h // 2 + ho
            #print('crops', crop_w, crop_h, cx, cy)
            input_img = utils.crop_center(input_img, cx, cy, crop_w, crop_h)

        #print('hflip: %d, vflip: %d, angle: %f, scale: %f' % (hflip, vflip, angle, scale))
        if angle:
            if trans:
                input_img = cv2.transpose(input_img)
            m_translate = np.identity(3)
            if hflip:
                m_translate[0, 0] *= -1
                m_translate[0, 2] = (self.img_size[0] + crop_w) / 2 - 1
            else:
                m_translate[0, 2] = (self.img_size[0] - crop_w) / 2
            if vflip:
                m_translate[1, 1] *= -1
                m_translate[1, 2] = (self.img_size[1] + crop_h) / 2 - 1
            else:
                m_translate[1, 2] = (self.img_size[1] - crop_h) / 2

            if angle or scale != 1.:
                m_rotate = cv2.getRotationMatrix2D((crop_w / 2, crop_h / 2), angle, scale)
                m_final = np.dot(m_translate, np.vstack([m_rotate, [0, 0, 1]]))
            else:
                m_final = m_translate

            input_img = cv2.warpAffine(
                input_img, m_final[:2, :], self.img_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        else:
            if trans:
                input_img = cv2.transpose(input_img)
            if hflip or vflip:
                if hflip and vflip:
                    c = -1
                else:
                    c = 0 if vflip else 1
                input_img = cv2.flip(input_img, flipCode=c)

            input_img = cv2.resize(input_img, self.img_size,  interpolation=cv2.INTER_CUBIC)

        return input_img

    def _centre_crop_and_transform(self, input_img, scale=1.0, trans=False, vflip=False, hflip=False):
        h, w = input_img.shape[:2]
        cx = w // 2
        cy = h // 2
        crop_w, crop_h = utils.calc_crop_size(self.img_size[0], self.img_size[1], scale=scale)
        input_img = utils.crop_center(input_img, cx, cy, crop_w, crop_h)
        if trans:
            input_img = cv2.transpose(input_img)
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            input_img = cv2.flip(input_img, flipCode=c)
        if scale != 1.0:
            input_img = cv2.resize(input_img, self.img_size, interpolation=cv2.INTER_CUBIC)
        return input_img

    def __getitem__(self, index):
        if not self.is_training and len(self.test_aug) > 1:
            aug_index = index % len(self.test_aug)
            index = index // len(self.test_aug)
        else:
            aug_index = 0

        prod_id, img_num, filename = self.inputs[index]
        input_img = self._load_input(filename)

        if self.targets is not None:
            category_id = self.targets[prod_id]
            target_label = self.category_to_label3[category_id]
            if self.is_training and self.multi_target > 1:
                target_label2 = self.category_to_label2[category_id]
                if self.multi_target == 3:
                    target_label1 = self.category_to_label1[category_id]
                    target_tensor = [target_label, target_label2, target_label1]
                else:
                    target_tensor = [target_label, target_label2]
            else:
                target_tensor = target_label
        else:
            assert not self.is_training
            target_tensor = torch.zeros(1)

        h, w = input_img.shape[:2]
        if self.is_training:
            mid = float(self.img_size[0]) / w
            if self.crop_factor:
                mid /= self.crop_factor
            scale_range = (mid - .05, mid + .05)
            input_img = self._random_crop_and_transform(input_img, scale_range=scale_range, rot=10.0)
            input_tensor = self.transform(input_img)
        else:
            scale = float(self.img_size[0]) / w
            if self.crop_factor:
                scale /= self.crop_factor
            trans, vflip, hflip = False, False, False
            if len(self.test_aug) > 1:
                trans, vflip, hflip = self.test_aug[aug_index]
            input_img = self._centre_crop_and_transform(
                input_img, scale=scale, trans=trans, vflip=vflip, hflip=hflip)
            input_tensor = self.transform(input_img)

        index_tensor = torch.LongTensor([index])
        return input_tensor, target_tensor, index_tensor

    def __len__(self):
        return len(self.inputs) * len(self.test_aug) if self.test_aug else len(self.inputs)

    def get_aug_factor(self):
        return len(self.test_aug)


def test():
    dataset = CDiscountDataset(
        '/data/f/cdiscount/train',
        is_training=True
    )

    stats = []
    ind = np.random.permutation(len(dataset))
    for i in ind[:len(dataset)//4]:
        mean_std = cv2.meanStdDev(dataset[i])
        if i % 100 == 0:
            print(mean_std)
        stats.append(mean_std)
    print(np.mean(stats, axis=0))


if __name__ == '__main__':
    test()
