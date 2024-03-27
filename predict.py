import argparse
import torch
import cv2
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_toolbelt.utils.torch_utils import image_to_tensor
# from torchsummary import summary
from model import get_model
from torch import nn
import sys
from glob import glob
import os

file_exts = ('*.tif', '*.png', '*.jpg', '*.tiff')
_gt_dir_name = 'gt'

def read_image(filename):
    image = cv2.imread(filename)
    if image is None:
        raise IOError("Cannot read " + filename)
    return image


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


def write_plots_and_visualize(path_to_res, visualize=False, **images):
    """Функция для визуализации данных, располагает изображения в ряд"""
    n = len(images)
    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(images.items()):
        if image is not None:
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)

    plt.savefig(path_to_res, bbox_inches='tight')

    if visualize:
        plt.show()


def make_prediction(model, image, tile_size: int, step: int, device):
    w, h, z = image.shape
    print(f'image shape is{image.shape}')
    # делаем размер выхода кратное размеру тайла с округлением вверх
    res_shape = (math.ceil(float(w) / tile_size) * tile_size, math.ceil(float(h) / tile_size) * tile_size, z)
    print(f'res shape is{res_shape}')
    w_new = res_shape[0]
    h_new = res_shape[1]
    res = np.zeros(res_shape, dtype=np.float32)
    res[0:w, 0:h, :] = image
    image = res

    image = normalize(image)
    tensor = image_to_tensor(image)
    tensor = tensor.reshape((1,) + tensor.shape)
    tensor.to(device)

    res = np.zeros((1, w_new, h_new), dtype=np.float32)
    i = j = 0

    with tqdm(total=(w_new // step) * (h_new // step)) as pbar:
        while i + tile_size <= w_new:
            j = 0
            while j + tile_size <= h_new:
                # print(f'i is {i} j is {j}')
                frag = tensor[:, :, i:i + tile_size, j:j + tile_size]
                # print(frag.shape)
                out = model.forward(frag)
                out = out.detach().numpy()
                print(f'model returned tensor {out.shape}')
                res[:, i:i + tile_size, j:j + tile_size] = out
                j += step
                pbar.update(1)
            i += step

    res = res.reshape((w_new, h_new))
    res = res[0:w, 0:h]

    max_val = np.max(res)
    min_val = np.min(res)

    res = 255.0 * (res - min_val) / (max_val - min_val)
    res = np.uint8(res)
    return res


def main():
    parser = argparse.ArgumentParser(description='Run model in evaluation mode')
    parser.add_argument("-check", '--check-path', type=str, help="checkpoint path")
    parser.add_argument("-cd", "--check-dir", type=str, help="path where checkpoint is placed,"
                                                             "script will find a checkpoint with .pth extension")
    # parser.add_argument("-t", '--thres', type=float, default=0.5, help="threshold for calculating metrics")
    # отличие --dataset-dir и --data-dir в том что у dataset есть разметка и подпапки images и gt,
    # а data-dir - просто папка
    parser.add_argument("-d", "--dataset-dir", type=str, help="path where dataset is placed")
    parser.add_argument("-dir", "--data-dir", type=str, help="path where dir is placed")
    parser.add_argument("-i", "--img-path", type=str, help="path of an image to segment")
    parser.add_argument("-o", "--output-dir", type=str, help="path where to write output,"
                                                             "if isn't provided output=checkpoint's dir")
    parser.add_argument('--vis', action='store_true', help="visualize plot with result on one image")
    parser.add_argument('--no-plots', action='store_true', help="don't write plots to disk")
    parser.add_argument('--com', '--calc-only-metrics', action='store_true',
                        help="calc only metrics without saving imgs with predictions")
    # parser.add_argument('--uim', '--use-imagenet-means', action='store_true',
    #                     help="use imagenet mean and std for normalizing images")

    parser.add_argument("--tile-size", default=512, help="default=512")
    parser.add_argument('--cpu', action='store_true', help="use cpu for training")
    parser.add_argument("--device", type=str, default="cuda", help="default=cuda, or cpu")
    # parser.add_argument("-w", "--workers", type=int, default=6, help="default=6")

    args = parser.parse_args()
    print("Passed arguments: ", args)

    # args.img_path = r'D:\Vector_data\RG3\frag1_new.tif'
    # args.output_dir = r'D:\Vector_data\Water'
    # -i D:\Vector_data\RG3\frag1_new.tif

    if args.cpu:
        args.device = 'cpu'

    b_visualize = True if args.vis else False
    b_write_plots = False if args.no_plots else True
    b_calc_only_metrics = True if args.vis else False

    # Определяем путь до чекпоинта
    if args.check_path is None:
        if args.check_dir is not None:
            # print(f'Look for a checkpoint in {args.check_dir}')
            check_path = glob(os.path.join(args.check_dir, '*.pth'))
            if len(check_path) == 0:
                print(f"Can't find a checkpoint in {args.check_dir}\n"
                      f"Please provide a checkpoint path (-c or --check-path) "
                      "or a checkpoint's dir path (--cd or --check-dir)", file=sys.stderr)
            else:
                args.check_path = check_path[0]
        else:
            print("Please provide a checkpoint path (-c or --check-path) "
                  "or a checkpoint's dir path (--cd or --check-dir)", file=sys.stderr)

    # Определяем где лежат изображения
    _b_calc_metrics = False
    _output_dir_name = None  # если пользователь не указал output_dir создадим папку с этим именем в папке с чекпоинтом
                            #  для сохранения предсказаний
    _files_to_process = []
    if args.img_path is not None:  # Сегментируем одно изображение
        _files_to_process.append(args.img_path)
    elif args.dataset_dir is not None:  # Сегментируем все изображения из подпапки images,если есть gt папка считаем IoU
        for ext in file_exts:
            _files_to_process.extend(glob(os.path.join(args.dataset_dir, 'images', ext)))
        # для набора данных возьмем его имя
        _output_dir_name = os.path.split(os.path.dirname(args.dataset_dir))[-1] + '_' + os.path.split(args.dataset_dir)[-1]
        _b_calc_metrics = os.path.isdir(os.path.join(args.dataset_dir, 'gt'))  # проверяем, что есть папка с разметкой
        if not _b_calc_metrics:
            print(f"Can't find 'gt' dir {os.path.join(args.dataset_dir, 'gt')}. THE SCRIPT CAN'T CALCULATE METRICS!!!",
                  file=sys.stderr)
    elif args.data_dir is not None:  # Сегментируем все изображения из этой папки
        for ext in file_exts:
            _files_to_process.extend(glob(os.path.join(args.data_dir, ext)))
        _output_dir_name = os.path.split(os.path.dirname(args.data_dir))[-1]
    else:
        print("Please provide a dataset path -d | dir path -dir | or image path -i", file=sys.stderr)
        exit()

    # Определяем путь куда сохранять результаты. Сохраняем в папку, которую указал пользователь или папку чекпоинта.
    if args.output_dir is not None:
        _output_path = args.output_dir
    else:
        if args.check_path is None:
            _output_path = args.check_dir
            if _output_dir_name is not None:
                _output_path = os.path.join(_output_path, _output_dir_name)
        else:
            _output_path = os.path.dirname(args.check_path)
            if _output_dir_name is not None:
                _output_path = os.path.join(_output_path, _output_dir_name)

    _output_plots_path = os.path.join(_output_path, 'plots')
    exist_ok = True
    if _output_dir_name is not None:  # Создаем папку куда сохраняем предсказания
        os.makedirs(_output_path, exist_ok=exist_ok)
    os.makedirs(_output_plots_path, exist_ok=exist_ok)  # Создаем папку куда сохраним графики

    _tile_size = args.tile_size
    _device = args.device
    _step = _tile_size  # шаг с которым идем по исходной картинке для вырезания тайлов

    checkpoint = torch.load(args.check_path)
    model_name = checkpoint['model_name']
    encoder_name = checkpoint['encoder_name']
    encoder_weights = checkpoint['encoder_weights']
    activation = checkpoint['activation']

    # encoder_weights = 'imagenet'
    _model = get_model(model_name=model_name, encoder_name=encoder_name,
                      encoder_weights=encoder_weights, activation=activation)

    # для вывода архитектуры сети, но будет понятнее, если просто посмотреть исходный код модели
    # _model.to('cuda')
    # # print(_model)
    # input = torch.tensor((3, 512, 512))
    # summary(_model, (3, 512, 512))
    # exit()

    _model = nn.DataParallel(_model)
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model = _model.module.to(_device)  # убираем nn.DataParallel т.к. с ним не считается на cpu
    _model.eval()
    
    for file_path in _files_to_process:
        print(f'Process {file_path} \n')
        try:
            image = read_image(file_path)
        except Exception as e:
            print(e, file=sys.stderr)
            continue

        res = make_prediction(_model, image, _tile_size, _step, _device)
        filename = os.path.basename(file_path)

        # # считаем метрики
        # if _b_calc_metrics:  # Если считаем метрики, значит есть папка gt с разметкой
        #     try:
        #         gt_image = read_image(os.path.join(args.dataset_dir, _gt_dir_name, filename))
        #     except Exception as e:
        #         print(e, file=sys.stderr)
        #         print("Can't calculate metrics for that image", file=sys.stderr)
        #     print(f'gt image shape is {gt_image.shape}')

        if not b_calc_only_metrics:
            print(f'Save {os.path.join(_output_path, filename)}')
            cv2.imwrite(os.path.join(_output_path, filename), res)

            if b_write_plots:
                write_plots_and_visualize(  # (path_to_res, visualize=False, **images)
                    os.path.join(_output_plots_path, filename),
                    visualize=b_visualize,
                    image=image,  # image=image_copy.permute(1, 2, 0),
                    predicted=res,
                    mask=image if False else None
                )


if __name__ == '__main__':
    main()
