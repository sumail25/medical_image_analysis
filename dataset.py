import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio


class LiverDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.train_root = r"data/liver/train"
        self.val_root = r"data/liver/val"
        self.test_root = self.val_root
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        assert self.state == "train" or self.state == "val" or self.state == "test"
        if self.state == "train":
            root = self.train_root
        if self.state == "val":
            root = self.val_root
        if self.state == "test":
            root = self.test_root

        pics = []
        masks = []
        n = len(os.listdir(root)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
        for i in range(n):
            img = os.path.join(root, "%03d.png" % i)  # liver is %03d
            mask = os.path.join(root, "%03d_mask.png" % i)
            pics.append(img)
            masks.append(mask)
            # imgs.append((img, mask))
        return pics, masks

    def __getitem__(self, index):
        # x_path, y_path = self.imgs[index]
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = Image.open(x_path)
        origin_y = Image.open(y_path)
        # origin_x = cv2.imread(x_path)
        # origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y, x_path, y_path

    def __len__(self):
        return len(self.pics)


class LungDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r"data/finding-lungs-in-ct-data"
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths, self.test_img_paths = None, None, None
        self.train_mask_paths, self.val_mask_paths, self.test_mask_paths = (
            None,
            None,
            None,
        )
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r"\2d_images\*")
        self.mask_paths = glob(self.root + r"\2d_masks\*")
        (
            self.train_img_paths,
            self.val_img_paths,
            self.train_mask_paths,
            self.val_mask_paths,
        ) = train_test_split(
            self.img_paths, self.mask_paths, test_size=0.2, random_state=41
        )
        self.test_img_paths, self.test_mask_paths = (
            self.val_img_paths,
            self.val_mask_paths,
        )
        assert self.state == "train" or self.state == "val" or self.state == "test"
        if self.state == "train":
            return self.train_img_paths, self.train_mask_paths
        if self.state == "val":
            return self.val_img_paths, self.val_mask_paths
        if self.state == "test":
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        # origin_x = Image.open(x_path)
        # origin_y = Image.open(y_path)
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
        pic = pic.astype("float32") / 255
        mask = mask.astype("float32") / 255
        # if self.aug:
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[:, ::-1, :].copy()
        #         mask = mask[:, ::-1].copy()
        #     if random.uniform(0, 1) > 0.5:
        #         pic = pic[::-1, :, :].copy()
        #         mask = mask[::-1, :].copy()
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def __len__(self):
        return len(self.pics)
