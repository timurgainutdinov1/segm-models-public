from pathlib import Path
import cv2
import os
from tqdm import tqdm


def main():
    """
    Функция для нарезания многоканального снимка на фрагменты tile_size с шагом shift
    """
    tile_size = 512
    shift = tile_size  # Можно задать значение меньше тогда тайлы будут нарезаться со сдвигом 256, а не 512
    path = Path(r'D:\Vector_data\RG3\T1_water')  # Путь к папке где лежат каналы снимка
    output_dir = Path(r'D:\Vector_data\RG3\water_dataset')  # Путь куда сохранить результат
    img_name = 'rg3'

    # images = ['rgb-lab.png', 'NI.png', 'water.png']
    # dirs = ['images', 'nir', 'gt']
    # # читаем изображение как оно есть, флаг '-1' без конвертации в 8 бит
    # flags = [None, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_GRAYSCALE]

    images = ['ndvi.png']
    dirs = ['ndvi']
    # читаем изображение как оно есть, флаг '-1' без конвертации в 8 бит
    flags = [cv2.IMREAD_GRAYSCALE]

    exist_ok = True
    os.makedirs(output_dir, exist_ok=exist_ok)
    for dir in dirs:
        os.makedirs(output_dir / dir, exist_ok=exist_ok)

    for idx in range(len(images)):
        print(str(path / images[idx]))
        image = cv2.imread(str(path / images[idx]), flags[idx])
        ch_count = len(image.shape)
        h, w = image.shape[0], image.shape[1]
        i = n = 0
        with tqdm(total=(w // tile_size) * (h // tile_size)) as pbar:
            while i + tile_size <= h:
                j = 0
                while j + tile_size <= w:
                    n += 1
                    p = str((output_dir / dirs[idx] / f'{img_name}_{n}.tif').resolve())
                    if ch_count == 2:
                        cv2.imwrite(p, image[i:i + tile_size, j:j + tile_size])
                    else:
                        cv2.imwrite(p, image[i:i + tile_size, j:j + tile_size, :])
                    j += shift
                    pbar.update(1)
                i += shift


if __name__ == '__main__':
    main()
