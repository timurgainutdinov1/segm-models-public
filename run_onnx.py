import onnxruntime
import numpy as np
from PIL import Image
import operator
from tqdm import tqdm
import os
from argparse import ArgumentParser
#import requests

# TODO Добавить корректную обработку тайлов меньше 512*512 пикселей
# Растр -> Извлечение -> Обрезать растр по обхвату
Image.MAX_IMAGE_PIXELS = None
DEBUG = True
url = ''


def normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def run_network(img, session):
    # compute ONNX Runtime output prediction
    ort_inputs = {session.get_inputs()[0].name: img}
    return session.run(None, ort_inputs)[0]


# def download_model(url, model_path):
#     response = requests.get(url, stream=True)
#
#     with open(model_path, "wb") as handle:
#         for data in tqdm(response.iter_content()):
#             handle.write(data)


def main():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", help="file path for processing")
    parser.add_argument("-o", "--out", help="path for output")
    args = parser.parse_args()

    print(f'File path for processing {args.file}')

    img_path = args.file
    output_path = args.out
    dir_name, filename = os.path.split(img_path)
    filename, ext = os.path.splitext(filename)
    postfix = '_filtered'

    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, 'onnx/buildings_b6.onnx')

    # print("Downloading model weights...")
    # download_model(url, model_path)
    ort_session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    tile_size = 512
    step = tile_size

    #img = Image.open("/misc/home1/u0304/Catalyst-Inria-Segmentation-Example/test/rgbX2.tif")
    img = Image.open(img_path)
    img = np.asarray(img)
    img = normalize(img)

    img = np.moveaxis(img, -1, 0)
    _, w, h = img.shape

    #img = img[:, 0:512, 0:512]
    img = img.reshape((1,) + img.shape)
    print(f'img shape is {img.shape}')
    # print(f'img dtype is {img.dtype}')

    res_shape = tuple(map(operator.add, img.shape, (0, 0, tile_size - w % tile_size + tile_size // 2,
                                                    tile_size - h % tile_size + tile_size // 2)))
    res = np.zeros(res_shape, dtype=np.float32)
    res[:,:,0:w,0:h] = img
    img = res

    w_new = res_shape[2]
    h_new = res_shape[3]
    res = np.zeros((1, w_new, h_new), dtype=np.float32)
    i = j = 0
    # res =
    # print(f'prom shape is {res.shape}')

    # Первый проход
    #total = (w // tile_size + 1) * (h // tile_size + 1)
    with tqdm(total=(w // step + 1) * (h // step + 1)) as pbar:
        while i + tile_size <= w_new:
            j = 0
            while j + tile_size <= h_new:
                #print(f'i is {i} j is {j}')
                frag = img[:, :, i:i+tile_size, j:j+tile_size]

                out = run_network(frag, ort_session)

                res[:, i:i+tile_size, j:j+tile_size] = out
                j += step
                pbar.update(1)
            i += step

    # # Второй проход со сдвигом tile_size || 2
    # # total = (w // tile_size + 1) * (h // tile_size + 1)
    # i = tile_size // 2
    # with tqdm(total=(w_new // step) * (h_new // step)) as pbar:
    #     while i + tile_size <= w_new:
    #         j = tile_size // 2
    #         while j + tile_size <= h_new:
    #             print(f'i is {i} j is {j}')
    #             frag = img[:, :, i:i + tile_size, j:j + tile_size]
    #
    #             out = run_network(frag, ort_session)
    #             res[:, i:i + tile_size, j:j + tile_size] *= 0.5
    #             #print(f'out.shape {out.shape} j res.shape {res.shape}')
    #             res[:, i:i + tile_size, j:j + tile_size] += 0.5 * out[0, :, :, :]
    #             j += step
    #             pbar.update(1)
    #         i += step

    max = np.max(res)
    min = np.min(res)

    img = res.reshape((w_new, h_new))
    img = img[0:w, 0:h]
    img = 255.0*(img - min)/(max-min)
    print(f'result shape is {img.shape}')
    # print(f'max val is {max} min val is {min}')
    img = Image.fromarray(np.uint8(img))
    img.save(output_path)

    if not DEBUG:
        # скопируем данные по extent и проекции из исходного файла в целевой файл
        print(f'stage fix extent')
        import subprocess
        import sys
        dirname = os.path.dirname(__file__)
        subprocess.check_call([sys.executable, os.path.join(dirname,'gdalcopyproj.py'), img_path, output_path])


if __name__ == "__main__":
    main()

