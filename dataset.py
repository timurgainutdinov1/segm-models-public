import cv2
from torch.utils.data import Dataset
from pytorch_toolbelt.utils.torch_utils import image_to_tensor
from glob import glob
import albumentations as A
from augmentations import get_augmentations
import os
import numpy as np
from tqdm import tqdm
from typing import List
import sys

# def new_print(x, *args):
#     __builtins__.print(x, *args, flush=True)
#
# print = new_print

file_exts = ('*.tif', '*.png', '*.jpg')


def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of found objects
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img


class TilesDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int, max_mask_val=0, class_list=[255],
                 augmentation=None, add_dirs=None, apply_crop=True, use_masks=True, images_list_path=None):
        """
        :param data_dir: Путь к набору данных
        :param image_size: Целевой размер изображений в наборе
        :param augmentation: str 'hard' Какие аугментации применять
        :param max_mask_val: Максимальное значение класса на масках, например 255, 1, или 10, если не 0 то скрипт будет
                             обучаться для таких классов: self.class_list = list(range(1, max_mask_val+1))
        :param class_list: List[int] Список классов  которые использовать для обучения, по умолчанию:255
        :param add_dirs: List ['nir, kv'] - папки с дополнительными каналами
        """
        print(f'Creating dataset from {data_dir}')
        image_size = int(image_size)
        max_mask_val = int(max_mask_val)

        self.data_dir = data_dir
        self.name = os.path.basename(os.path.dirname(data_dir))
        # TODO читать in_channels из файла и сделать cv::imreadmulti из images ,если нужно читать многоканальные
        # изображения, сейчас в images должны лежать трехканальные изображения
        self.use_masks = use_masks
        self.in_channels = 3
        self.add_dirs = add_dirs
        if add_dirs is not None:
            self.in_channels = self.in_channels + len(self.add_dirs)

        self.class_list = class_list
        if max_mask_val > 0:
            self.class_list = list(range(1, max_mask_val+1))
        # else:
            # self.class_list = [int(x) for x in class_list.split()]
        print(f"Class list {self.class_list}")
        self.num_classes = len(self.class_list)
        # если многоклассовая классификация то добавляем класс background (на один класс больше)
        self.num_classes = 1 if self.num_classes == 1 else self.num_classes + 1
        print(f"Num classes to train with background: {self.num_classes}")

        # Читаем список изображений с диска
        if self.data_dir is not None:
            self.image_set = []    # TODO переименовать в images_list
            for ext in file_exts:
                self.image_set.extend(glob(os.path.join(data_dir, 'images', ext)))
        elif images_list_path is not None:
            with open(images_list_path) as file:
                self.image_set = [line.rstrip() for line in file]
        else:
            print("Please provide a dataset path -d or train images list -lt ", file=sys.stderr)
            exit()

        self.mean_var = {'mean': [], 'std': []}
        self.read_mean_vals(data_dir)
        print(f'Mean vals {self.mean_var}')

        if apply_crop:
            image_transform = [
                A.Resize(image_size, image_size),
                A.CenterCrop(image_size, image_size),
                A.Normalize(self.mean_var['mean'], self.mean_var['std'], 1),
            ]
        else:
            image_transform = [
                A.Normalize(self.mean_var['mean'], self.mean_var['std'], 1),
            ]
        self.transform = A.Compose(get_augmentations(augmentation, self.in_channels == 3) + image_transform)
        # нужен для записи примеров с аугментацией на диск
        self.test_transform = A.Compose(get_augmentations(augmentation, self.in_channels == 3))

    def read_image(self, path):
        '''
        Дополнительные каналы считываются как одноканальные изображения
        '''
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # читаем изображение как оно есть, а не в 8 бит
        if image is None:
            raise IOError(f"Cannot read {path}")

        if self.add_dirs and len(self.add_dirs) > 0:
            filename = os.path.basename(path)
            for i_dir in range(len(self.add_dirs)):
                path = os.path.join(self.data_dir, self.add_dirs[i_dir], filename)
                channel = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # читаем изображение как оно есть
                if channel is None:
                    raise IOError(f"Cannot read {path}")
                image = image.astype(np.float32)
                if len(channel.shape) > 2:
                    channel = channel[:, :, 0]  # оставляем только один канал
                channel = channel.astype(np.float32)
                image = cv2.merge([image, channel])
        return image

    def read_mask_for_img(self, img_path: str):
        """
        Читает маску с разметкой для изображения из img_path
        """
        filename = os.path.basename(img_path)
        mask_path = os.path.join(self.data_dir, 'gt', filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # print(filename, '\n')
        if mask is None:
            raise IOError("Cannot read " + mask_path)

        # extract certain classes from mask (e.g. cars)
        masks = [np.where(mask == v, 1, 0) for v in self.class_list]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        return mask

    def write_means(self, path: str):
        mean = [[] for i in range(self.in_channels)]
        var = [[] for i in range(self.in_channels)]
        for i in tqdm(range(len(self.image_set))):
            # calc rgb mean
            img = self.read_image(self.image_set[i])  # изображение в формате opencv - B G R + доп каналы
            for ch in range(img.shape[2]):
                m = np.array(img[:, :, ch]).mean()
                mean[ch].append(m)
                v = img[:, :, ch] - m
                v = np.mean(v**2)
                var[ch].append(v)
        for ch in range(len(mean)):
            mean[ch] = np.mean(np.array(mean[ch]))
            var[ch] = np.sqrt(np.mean(np.array(var[ch])))
        print(f'Calculated mean {mean} calculated std {var}')
        with open(path,"w") as file:
            file.write(' '.join(str(x) for x in mean))
            file.write('\n')
            file.write(' '.join(str(x) for x in var))
        return {'mean': mean, 'std': var}

    def read_mean_vals(self, path: str):
        path = os.path.join(os.path.dirname(path), 'mean_vals.txt')
        if not os.path.isfile(path):
            print(f'No means vals found. Calculating mean vals over dataset to {path}')
            self.mean_var = self.write_means(path)
        else:
            self.mean_var = {}
            with open(path) as file:
                self.mean_var['mean'] = [float(x) for x in file.readline().split()]
                self.mean_var['std'] = [float(x) for x in file.readline().split()]
        self.mean_var['mean'] = self.mean_var['mean'][0:self.in_channels]
        self.mean_var['std'] = self.mean_var['std'][0:self.in_channels]

    def __len__(self):
        return len(self.image_set)

    def __getitem__(self, index):
        image = self.read_image(self.image_set[index])
        if self.use_masks:
            mask = self.read_mask_for_img(self.image_set[index])
            # print(f'input img shape is {image.shape} res_mask shape is {mask.shape} \n', flush=True)
            data = self.transform(image=image, mask=mask)
            # return torch.from_numpy(data['image']), torch.from_numpy(data['mask'])
            res_img = image_to_tensor(data['image'])
            res_mask = image_to_tensor(data['mask'])
            # print(f'res_img shape is {res_img.shape} res_mask shape is {res_mask.shape} \n', flush=True)
            return res_img, res_mask
        else:
            data = self.transform(image=image)
            return image_to_tensor(data['image']), None

    def test_augment(self, save_dir, n_samples):
        """
        Сохраняет в save_path n_samples примеров из набора данных с применением аугментации
        :param save_dir: куда сохранить примеры картинок
        :param n_samples: сколько примеров сохранить
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        orig_transform = self.transform
        self.transform = self.test_transform

        for i in range(n_samples):
            idx = np.random.randint(0, len(self))
            image, mask = self[idx]
            cv2.imwrite(os.path.join(save_dir, f'{idx}_image.tif'), image.numpy())
            cv2.imwrite(os.path.join(save_dir, f'{idx}_mask.tif'), 255*mask.numpy())

        self.transform = orig_transform




