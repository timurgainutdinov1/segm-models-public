from pathlib import Path
import os
from tqdm import tqdm
import glob
import shutil


def main():
    """
    Берет изображения из папки images (source_path) и копирует из набора (dataset_path) для этих изображений другие каналы
    """
    source_path = Path(r'D:\Vector_data\RG3\water_frags\val\images')
    dataset_path = Path(r'D:\Vector_data\RG3\water_frags')
    res_path = Path(r'D:\Vector_data\RG3\water_frags\val')

    folders = ['gt', 'nir', 'rgb_orig', 'ndvi']
    files_ext = '*.tif'

    exist_ok = True
    for folder in folders:
        os.makedirs(res_path / folder, exist_ok=exist_ok)

    files = glob.glob(str(source_path / files_ext))

    for f in tqdm(files):
        basename = os.path.basename(f)
        for folder in folders:
            shutil.copy2(dataset_path / folder / basename, res_path / folder / basename, follow_symlinks=True)


if __name__ == '__main__':
    main()
