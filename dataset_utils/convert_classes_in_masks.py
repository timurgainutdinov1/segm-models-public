from pathlib import Path
import os
from tqdm import tqdm
import glob
import cv2
import numpy as np


def swap_colors(image, color_mapping):
    image_copy = np.copy(image)
    for old_color, new_color in color_mapping:
        image_copy[image == old_color] = new_color
    return image_copy


def main():
    """
    Функция для конвертации цветов в масках
    Запустить скрипт на кластере можно с помощью одной из этих команд:
    sbatch --mem=32000 -t 0:30:00 --wrap="python convert_classes_in_masks.py"
    srun --mem=32000 -t 0:30:00 python convert_classes_in_masks.py
    """
    # LandCover.ai
    source_path = Path(r'D:\Vector_data\RG3\water_dataset\gt_white_bg')
    res_path = Path(r'D:\Vector_data\RG3\water_dataset\gt_1_2')  # лучше сохранять в новую папку чтобы убедиться, что всё ок
    files_ext = '*.tif'
    # old_new_colors = [[3, 1], [4, 3], [1, 4]]
    old_new_colors = [[255, 1], [0, 2]]
    #old_new_colors = [[255, 64], [64, 255], [192, 128], [128, 192]]

    # DeeepGlobe
    # source_path = Path(r'/misc/home6/m_imm_freedata/Segmentation/DeepGlobe_Land/DeepGlobe512/val/gt')
    # res_path = Path(r'/misc/home6/m_imm_freedata/Segmentation/DeepGlobe_Land/DeepGlobe512/val/gt2')  # лучше сохранять в новую папку чтобы убедиться, что всё ок
    # files_ext = '*.tif'
    # # old_new_colors = [[0, 1], [255, 0]]  # заменяем 0 на 1, 255 на 0
    # old_new_colors = [[1, 64], [4, 128], [5, 40], [6, 150], [7, 100], [8, 250]]

    # MSFLoods
    # source_path = Path(r'/misc/home6/m_imm_freedata/Segmentation/floods_detection/MS_Floods_512/val/gt')
    # res_path = Path(r'/misc/home6/m_imm_freedata/Segmentation/floods_detection/MS_Floods_512/val/gt2')  # лучше сохранять в новую папку чтобы убедиться, что всё ок
    # files_ext = '*.tif'
    # # old_new_colors = [[0, 1], [255, 0]]  # заменяем 0 на 1, 255 на 0
    # old_new_colors = [[1, 64]]

    exist_ok = False
    os.makedirs(res_path, exist_ok=exist_ok)

    files = glob.glob(str(source_path / files_ext))
    for f in tqdm(files):
        mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        basename = os.path.basename(f)
        new_mask = swap_colors(mask, old_new_colors)
        cv2.imwrite(str(res_path / basename), new_mask)


if __name__ == '__main__':
    main()
