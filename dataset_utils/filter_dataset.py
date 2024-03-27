from pathlib import Path
import cv2
import os
from tqdm import tqdm
import glob
import shutil


def main():
    """
    Функция для фильтрации набора данных. Генерируется новый набор где на всех тайлах присутствует целевой класс
    """
    # thres = (512*512) * 0.09  # 9%
    thres = 1  # сколько пикселей целевого класса должно быть на изображении
    dataset_path = Path(r'D:\Vector_data\RG3\water_dataset')
    res_path = Path(r'D:\Vector_data\RG3\water_dataset_filt')
    label = 0  # какой класс ищем на изображениях

    folders = ['gt', 'images', 'nir']  # папка с масками должна быть первая в списке, предполагается что они 1-канальные
    files_ext = '*.tif'

    exist_ok=False
    os.makedirs(res_path, exist_ok=exist_ok)

    for folder in folders:
        os.makedirs(res_path / folder, exist_ok=exist_ok)

    files = glob.glob(str(dataset_path / folders[0] / files_ext))

    for f in tqdm(files):
        mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        mask = (mask == label).astype('uint8')
        pixel_count = cv2.countNonZero(mask)
        if pixel_count > thres:
            basename = os.path.basename(f)
            for folder in folders:
                shutil.copy2(dataset_path / folder / basename, res_path / folder / basename, follow_symlinks=True)


if __name__ == '__main__':
    main()
