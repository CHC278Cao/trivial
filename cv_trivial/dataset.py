
"""
    Dataset for images including image processing methods. Based on initial data,
    changing the "get_" methods
"""
import os
import pdb
import cv2
import math
import numpy as np

import torch
from torch.utils.data import Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ImgDataset(Dataset):
    def __init__(self, imagefile, data_dir, mask_list = None, label_list=None, train_mode=True):
        """
            Create dataset for image and mask
        :param imagefile: Type: list of str, a list of image files
        :param mask_list: Type: list of list, a list of pixel lists
        :param data_dir: Type: file, the workdir of images
        :param label_list：Type: list of int, the list of labels
        :param train_mode: Type: bool, if True, it outputs "data" and "target", otherwise, it outputs "data"
        """
        self.imagefile = imagefile
        self.data_dir = data_dir
        self.mask_list = mask_list
        self.label_list = label_list

        self.train_mode = train_mode

        assert (os.path.exists(self.data_dir)), "data_dir doesn't exist"

    def __len__(self):
        return len(self.imagefile)

    def __getitem__(self, idx):
        imgfile = os.path.join(self.data_dir, self.imagefile[idx])
        img = torch.tensor(self._get_img(imgfile), dtype=torch.float)
        # pdb.set_trace()

        if self.train_mode:
            mask_list = self.mask_list.iloc[idx, :]
            mask = torch.tensor(self._get_mask(mask_list, img), dtype=torch.int)
            label = torch.tensor(self._get_label(mask_list), dtype=torch.float)
            img, mask = self._process_img_mask(img, mask)

            return {
                "img": img,
                "mask": mask,
                "label": label
            }

        else:
            return {
                "img": img,
            }

    def _get_img(self, imgpath):
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 2)
        img = img.astype(float) / 255.0

        return img

    def _get_mask(self, mask_list, img):
        height, width = img.shape[:2]
        mask = np.zeros((height, width, 4), dtype=np.uint8)

        for idx, pixel in enumerate(mask_list):
            if pixel == "None":
                continue
            signal_mask = np.zeros(height*width, dtype=np.uint8)
            pixel = pixel.split(" ")
            starts, lengths = [np.asarray(x, dtype=int) for x in (pixel[0::2], pixel[1::2])]
            starts -= 1
            ends = starts + lengths
            for lo, hi in zip(starts, ends):
                signal_mask[lo: hi] = 1
            img = signal_mask.reshape(height, width, order='F')
            mask[:, :, idx] = img

        return mask

    def _get_label(self, mask_list):
        label = [0] * 4
        for idx, pixel in enumerate(mask_list):
            if pixel != "None":
                label[idx] = 1
        return label

    def _process_img_mask(self, img, mask):

        if np.random.rand() > 0.5:
            img, mask = self._lr_flip(img, mask)
        if np.random.rand() > 0.5:
            img, mask = self._ud_flip(img, mask)
        if np.random.rand() > 0.5:
            img = self._random_noise(img)
        if np.random.rand() > 0.5:
            img = self._random_salt_pepper_noise(img)
        if np.random.rand() > 0.5:
            img = self._random_log_contast(img)

        crop_rotate = np.random.choice(2)
        if crop_rotate == 0:
            img, mask = self._random_crop_rescale(img, mask, 1124, 224)
        else:
            img, mask = self._random_rotate_rescale(img, mask)

        img = self._normalize_img(img, mean=MEAN, std=STD)

        return img, mask

    def _normalize_img(self, img, mean, std):

        mean = torch.tensor(mean, dtype=torch.float)
        std = torch.tensor(std, dtype=torch.float)
        if len(img.shape) == 3:
            if img.shape[2] == 1:
                mean = torch.mean(mean)
                std = torch.mean(std)
                img = (img - mean) / std

            else:
                assert (img.shape[2] == len(mean) and img.shape[2] == len(std)), \
                    "dimension dones't match"

                channel = img.shape[2]
                for c in range(channel):
                    img[:, :, c] = (img[:, :, c] - mean[c]) / std[c]

        elif len(img.shape) == 2:
            mean = torch.mean(mean)
            std = torch.mean(std)
            img = (img - mean) / std

        return img

    def _lr_flip(self, img, mask):
        img = torch.flip(img, dims=(1, ))
        mask = torch.flip(mask, dims=(1,))

        return img, mask

    def _ud_flip(self, img, mask):
        img = torch.flip(img, dims=(0, ))
        mask = torch.flip(mask, dims=(0, ))

        return img, mask

    def _random_crop(self, img, mask, w, h):

        height, width = img.shape[:2]
        if w > width or h > height:
            raise Exception("crop size is bigger than original image")
        x = np.random.choice(width - w)
        y = np.random.choice(height - h)

        if len(img.shape) == 3:
            img = img[y:y + h, x:x + w, :]
            mask = mask[y:y + h, x:x + w, :]
        else:
            img = img[y:y + h, x:x + w]
            mask = mask[y:y + h, x:x + w]

        return img, mask

    def _random_crop_rescale(self, img, mask, w, h):
        height, width, channel = img.shape[:3]
        crop_img, crop_mask = self._random_crop(img, mask, w, h)
        if (w, h) != (width, height):
            crop_img, crop_mask = crop_img.numpy(), crop_mask.float().numpy()
            img = cv2.resize(crop_img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(crop_mask, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
            if channel == 1:
                img, mask = torch.from_numpy(img).unsqueeze(dim=2), torch.from_numpy(mask).int()
            else:
                img, mask = torch.from_numpy(img), torch.from_numpy(mask).int()

        return img, mask

    def _get_matrix(self, rotation, shear, h_zoom, w_zoom, h_shift, w_shift):
        rotation = math.pi * rotation / 180
        shear = math.pi * shear / 180

        # rotation matrix
        c1 = torch.cos(rotation)
        s1 = torch.sin(shear)
        one = torch.ones(size=(1,))
        zero = torch.zeros(size=(1,))
        rotation_matrix = torch.reshape(torch.cat([c1, s1, zero, -s1, c1, zero, zero, zero, one]), (3, 3))

        # shear matrix
        c2 = torch.cos(shear)
        s2 = torch.sin(shear)
        shear_matrix = torch.reshape(torch.cat([one, s2, zero, zero, c2, zero, zero, zero, one]), (3, 3))

        # zoom matrix
        zoom_matrix = torch.reshape(torch.cat([one / h_zoom, zero, zero, zero, one / w_zoom, zero, zero, zero, one]), (3, 3))

        # shift matrix
        shift_matrix = torch.reshape(torch.cat([one, zero, h_shift, zero, one, w_shift, zero, zero, one]), (3, 3))

        matrix = torch.mm(torch.mm(rotation_matrix, shear_matrix), torch.mm(zoom_matrix, shift_matrix))
        return matrix

    def _random_rotate_rescale(self, img, mask):
        height, width = img.shape[:2]
        rotation = torch.randint(low=-15, high=15, size=(1,), dtype=torch.float)
        shear = torch.randint(low=-10, high=10, size=(1,), dtype=torch.float)
        h_zoom = 1.0 + torch.normal(mean=0., std=1., size=(1,)) / 10
        w_zoom = 1.0 + torch.normal(mean=0., std=1., size=(1,)) / 10
        h_shift = 1.0 + torch.normal(mean=0., std=1., size=(1,)) / 10
        w_shift= 1.0 + torch.normal(mean=0., std=1., size=(1,)) / 10

        matrix = self._get_matrix(rotation, shear, h_zoom, w_zoom, h_shift, w_shift)
        x = torch.from_numpy(np.repeat(np.arange(height//2, -height//2, -1), width)).long()
        y = torch.from_numpy(np.tile(np.arange(width // 2, -width // 2, -1), [height])).long()
        z = torch.ones(size = (height*width, ), dtype=torch.long)
        idx = torch.stack([x, y, z])

        idx2 = torch.mm(matrix, idx.float()).long()
        idx2[0, :] = torch.clamp(idx2[0, :], -height//2, height//2-1)
        idx2[1, :] = torch.clamp(idx2[1, :], -width//2, width//2-1)

        idx3 = torch.stack([height//2 - 1 - idx2[0, :], width//2 - 1 - idx2[1, :]])
        img = img[idx3[0, :], idx3[1, :]].reshape(height, width, img.shape[2])
        mask = mask[idx3[0, :], idx3[1, :]].reshape(height, width, mask.shape[2])

        return img, mask

    def _random_log_contast(self, img, gain=[0.70, 1.30]):
        gain = np.random.uniform(gain[0], gain[1], size=1)[0]
        inverse = np.random.choice(2, 1)

        if inverse == 0:
            img = torch.mul(torch.log(torch.add(img, 1.)), gain)
        else:
            img = torch.mul((torch.add(torch.pow(2., img), -1.)), gain)

        img = torch.clamp(img, 0., 1.)

        return img

    def _random_noise(self, img, noise = 0.4):
        height, width = img.shape[:2]
        img = img + torch.mul(torch.normal(-1, 1, size=(height, width, 1)), noise)
        img = torch.clamp(img, 0., 1.)

        return img

    def _random_contast(self, img):
        beta = 0.1
        alpha = torch.normal(0.5, 2.0, size=(1,))
        img = img * alpha + beta
        img = torch.clamp(img, 0., 1.)
        return img

    def _random_salt_pepper_noise(self, img, noise = 0.0005):
        height, width = img.shape[:2]
        num_salt = int(noise * width * height)
        # Salt mode
        yx = [np.random.randint(0, d - 1, num_salt) for d in img.shape[:2]]
        if img.shape[2] == 1:
            img[tuple(yx)] = 1.0
        elif img.shape[2] > 1:
            img[tuple(yx)] = torch.tensor([1.0] * img.shape[2], dtype=torch.float)

        # Pepper mode
        yx = [np.random.randint(0, d - 1, num_salt) for d in img.shape[:2]]
        if img.shape[2] == 1:
            img[tuple(yx)] = 0.0
        elif img.shape[2] > 1:
            img[tuple(yx)] = torch.tensor([1.0] * img.shape[2], dtype=torch.float)
        img = torch.clamp(img, 0., 1.)

        return img

    def _random_salt_pepper_line(self, img, noise=0.0005, length=10):
        height, width = img.shape[:2]
        num_salt = int(noise * width * height)
        # Salt mode
        y0x0 = np.array([np.random.randint(0, d - 1, num_salt) for d in img.shape[:2]]).T
        y1x1 = y0x0 + np.random.choice(2 * length, size=(num_salt, 2)) - length
        img = img.numpy()
        for (y0, x0), (y1, x1) in zip(y0x0, y1x1):
            if img.shape[2] == 1:
                cv2.line(img, (x0, y0), (x1, y1), 1.0, 1)
            elif img.shape[2] == 3:
                cv2.line(img, (x0, y0), (x1, y1), (1.0, 1.0, 1.0), 1)
        # Pepper mode

        y0x0 = np.array([np.random.randint(0, d - 1, num_salt) for d in img.shape[:2]]).T
        y1x1 = y0x0 + np.random.choice(2 * length, size=(num_salt, 2)) - length
        for (y0, x0), (y1, x1) in zip(y0x0, y1x1):
            if img.shape[2] == 1:
                cv2.line(img, (x0, y0), (x1, y1), 0.0, 1)
            elif img.shape[2] == 3:
                cv2.line(img, (x0, y0), (x1, y1), (0, 0, 0), 1)
        img = torch.tensor(img)
        img = torch.clamp(img, 0., 1.)

        return img

    def _random_cutout(self, img, mask):
        height, width = img.shape[:2]

        u0 = np.random.choice(2)
        u1 = np.random.choice(width)

        if u0 == 0:
            x0, x1 = 0, u1
        if u0 == 1:
            x0, x1 = u1, width
        img[:, x0:x1] = 0.0
        mask[:, x0:x1] = 0.0

        return img, mask



