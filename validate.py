import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# from torchsummary import summary
from model import get_model
from torch import nn
import sys
from glob import glob
import os
from dataset import TilesDataset
from segmentation_models_pytorch.utils.functional import iou, accuracy
import math
import statistics

'''
Скрипт написан неоптимально, т.к. обрабатывает каждую картинку отдельно и не объединяет их в батчи, для более быстрой
обработки на видеокарте
'''

WORK_FOR_OLD_CHECKS = False  # Сделать False если вы обучали сеть на этой версии проекта
# В коде зафиксирован номер класса воды - 64, т.к. старые чекпоинты не содержат labels,
# а также некоторые другие переменные.

_gt_dir_name = 'gt'


def read_image(filename):
    image = cv2.imread(filename)
    if image is None:
        raise IOError("Cannot read " + filename)
    return image


# def normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0):
#     mean = np.array(mean, dtype=np.float32)
#     mean *= max_pixel_value
#
#     std = np.array(std, dtype=np.float32)
#     std *= max_pixel_value
#
#     denominator = np.reciprocal(std, dtype=np.float32)
#
#     img = img.astype(np.float32)
#     img -= mean
#     img *= denominator
#     return img


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


def convert_torch_to_8_bit(tensor):
    b, c, w, h = tensor.shape
    res = tensor.cpu().detach().numpy()
    res = res.reshape((w, h))
    max_val = np.max(res)
    min_val = np.min(res)

    res = 255.0 * (res - min_val) / (max_val - min_val)
    res = np.uint8(res)
    return res


# Вычисление IoU для segmentation_models_pytorch версии 0.3.3
# if mask is not None:
#     tp, fp, fn, tn = smp.metrics.get_stats(res, mask, mode='binary', threshold=thres)
#     iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
#     # f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
#     # f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
#     accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
#     # recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
#     metrics = {'iou': float(iou_score), 'acc': float(accuracy)}


def make_prediction(model, image, tile_size: int, step: int, device, thres: float = 0.5,
                    mask=None, is_dataset: bool = True):
    """Функция для получения выхода сети"""

    c, w, h = image.shape
    if is_dataset:
        image = image.reshape((1,) + image.shape)
        image = image.to(device)
        if mask is not None:
            mask = mask.reshape((1,) + mask.shape)
            mask = mask.to(device)
        res = model.forward(image)
        raw = res.clone()
        res_low = torch.where(res < thres, res, torch.tensor(0, dtype=res.dtype).to(device))
        res = torch.where(res >= thres, 1, 0)
        metrics = None
        if mask is not None:
            iou_val = iou(res, mask, eps=1e-7)
            acc_val = accuracy(res, mask)
            metrics = {'iou': float(iou_val), 'acc': float(acc_val)}
            for key in metrics:
                if math.isnan(metrics[key]):
                    metrics[key] = 0

        res = convert_torch_to_8_bit(res)
        res_low = convert_torch_to_8_bit(res_low)
        raw = convert_torch_to_8_bit(raw)

        rgb = np.dstack((res, res, res))
        rgb_raw = np.dstack((res_low, res_low, res_low))
        binary = np.copy(rgb)
        rgb[res == 255] = (0, 0, 255)
        plot = rgb + rgb_raw
        # rgb = rgb_raw
        # rgb = np.add(rgb,rgb_raw)
        # cv2.imshow("image", rgb_raw)
        # cv2.waitKey(0)
        # mask = convert_torch_to_8_bit(mask)
        # cv2.imwrite(r'C:\RG3\checkpoints\res.tif', res)
        # cv2.imwrite(r'C:\RG3\checkpoints\mask.tif', mask)
        return {'plot': plot, 'binary': binary, 'raw': raw}, metrics

    else:
        print(f'image shape is{image.shape}')
        # делаем размер выхода кратное размеру тайла с округлением вверх
        res_shape = (math.ceil(float(w) / tile_size) * tile_size, math.ceil(float(h) / tile_size) * tile_size, c)
        print(f'res shape is{res_shape}')
        w_new = res_shape[0]
        h_new = res_shape[1]
        res = torch.zeros(res_shape, dtype=torch.float32)
        res[0:w, 0:h, :] = image
        image = res

        # image = normalize(image)
        # tensor = image_to_tensor(image)
        # tensor = tensor.reshape((1,) + tensor.shape)
        # tensor.to(device)

        image = image.reshape((1,) + image.shape)
        image.to(device)

        res = np.zeros((1, w_new, h_new), dtype=np.float32)
        i = j = 0

        with tqdm(total=(w_new // step) * (h_new // step)) as pbar:
            while i + tile_size <= w_new:
                j = 0
                while j + tile_size <= h_new:
                    # print(f'i is {i} j is {j}')
                    frag = image[:, :, i:i + tile_size, j:j + tile_size]
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
    parser.add_argument("-c", '--check-path', type=str, help="checkpoint path")
    parser.add_argument("-cd", "--check-dir", type=str, help="path where checkpoint is placed,"
                                                             "script will find a checkpoint with .pth extension")
    parser.add_argument("-t", '--thres', type=float, default=0.5, help="threshold for calculating metrics")
    # отличие --dataset-dir и --var-dataset в том что у dataset все картинки одинакового размера равному размеру тайла
    parser.add_argument("-d", "--dataset-dir", type=str, help="path where dataset is placed."
                                                              "The size of images will be RESIZED to tile-size")
    # parser.add_argument("-vd", "--var-dataset", type=str, help="Path to dataset dir with 'images' and 'gt' folders where"
    #                                                            "images can be bigger than tile size and can have "
    #                                                            "different size; ")
    # parser.add_argument("-i", "--img-path", type=str, help="path of an image to segment")
    parser.add_argument("-o", "--output-dir", type=str, help="path where to write output, if isn't provided "
                                                             "output=checkpoint's dir")
    parser.add_argument('--vis', action='store_true', help="visualize plot with result on one image")
    parser.add_argument('--plots', action='store_true', help="write img|output|mask in one plot")
    parser.add_argument('--binary', action='store_true', help="write binary output to disk")
    parser.add_argument('--raw', action='store_true', help="write raw output to disk")
    parser.add_argument('--com', '--calc-only-metrics', action='store_true',
                        help="calc only metrics without saving imgs with predictions")
    # parser.add_argument('--uim', '--use-imagenet-means', action='store_true',
    #                     help="use imagenet mean and std for normalizing images")
    # parser.add_argument('--ucm', '--use-checkpoint-means', action='store_true',
    #                     help="use mean and std from checkpoint for normalizing images")
    parser.add_argument("--tile-size", default=512, help="default=512")
    parser.add_argument('--cpu', action='store_true', help="use cpu for training")
    parser.add_argument("--device", type=str, default="cuda", help="default=cuda, or cpu")
    # parser.add_argument("-w", "--workers", type=int, default=6, help="default=6")

    args = parser.parse_args()
    if args.cpu:
        args.device = 'cpu'

    print("Passed arguments: ", str(args).replace(',',',\n'))

    b_visualize = True if args.vis else False
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
    _b_calc_metrics = False  # Если есть разметка, то посчитаем метрики
    _b_is_dataset = False  # True если все картинки будут уменьшены до tile_size / одинакового размера
    _output_dir_name = None  # если пользователь не указал output_dir создадим папку с этим именем в папке с чекпоинтом
    # для сохранения предсказаний
    _dataset_path = None
    # _files_to_process = []
    # if args.img_path is not None:  # Сегментируем одно изображение
    #     _files_to_process.append(args.img_path)
    if args.dataset_dir is None and args.var_dataset is None:
        # print("Please provide a dataset path -d | dir path -dir | or image path -i", file=sys.stderr)
        print("Please provide a dataset path -d | dir path -dir", file=sys.stderr)
        exit()
    else:
        if args.dataset_dir is not None:  # Сегментируем все изображения из подпапки images,если есть gt папка считаем IoU
            _b_is_dataset = True
            gt_path = os.path.join(args.dataset_dir, _gt_dir_name)
            # для набора данных возьмем его имя
            _output_dir_name = os.path.split(os.path.dirname(args.dataset_dir))[-1] + '_' + \
                               os.path.split(args.dataset_dir)[-1] + f"_thres_{args.thres}"
        # if args.var_dataset is not None:  # Сегментируем все изображения из этой папки,если рядом есть gt папка считаем IoU
        #     _dataset_path = args.var_dataset
        #     gt_path = os.path.join(os.path.dirname(os.path.normpath(args.var_dataset)), _gt_dir_name)
        #     _output_dir_name = os.path.split(os.path.dirname(args.var_dataset))[-1]
        _dataset_path = os.path.join(args.dataset_dir)
        # for ext in file_exts:
        #     _files_to_process.extend(glob(os.path.join(images_path, ext)))
        _b_calc_metrics = os.path.isdir(gt_path)  # проверяем, что есть папка с разметкой
        if not _b_calc_metrics:
            print(f"Can't find 'gt' {gt_path}. THE SCRIPT CAN'T CALCULATE METRICS!!!",
                  file=sys.stderr)

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
    _output_raw_path = os.path.join(_output_path, 'raw')
    exist_ok = True
    if _output_dir_name is not None:  # Создаем папку куда сохраняем предсказания
        os.makedirs(_output_path, exist_ok=exist_ok)
    os.makedirs(_output_plots_path, exist_ok=exist_ok)  # Создаем папку куда сохраним графики
    os.makedirs(_output_raw_path, exist_ok=exist_ok)  # Создаем папку куда сохраним графики

    checkpoint = torch.load(args.check_path, map_location=torch.device(args.device))
    model_name = checkpoint['model_name']
    encoder_name = checkpoint['encoder_name']
    encoder_weights = checkpoint['encoder_weights']
    activation = checkpoint['activation']
    in_channels = 3
    if WORK_FOR_OLD_CHECKS:
        print('SCRIPT USES hardcoded class_list = [64] and encoder_weights. Line 293 of validate.py')
        class_list = [64]
        add_dirs = None
        encoder_weights = 'imagenet'
    else:
        class_list = checkpoint['class_list']
        add_dirs = checkpoint['add_dirs']
        if add_dirs is not None:
            in_channels += len(add_dirs)
        print(f'Loaded values from the checkpoint: class_list={class_list} add_dirs={add_dirs} '
              f'in_channels={in_channels}')

    if _b_is_dataset:
        valid_set = TilesDataset(_dataset_path,
                                 args.tile_size,
                                 class_list=class_list,
                                 add_dirs=add_dirs,
                                 use_masks=_b_calc_metrics,
                                 )
    else:  # Если в папке все картинки разного размера, передаем apply_crop=False чтобы картинки не обрезались
        valid_set = TilesDataset(_dataset_path,
                                 args.tile_size,
                                 class_list=class_list,
                                 add_dirs=add_dirs,
                                 use_masks=_b_calc_metrics,
                                 apply_crop=False
                                 )

    _tile_size = args.tile_size
    _device = args.device
    _step = _tile_size  # шаг с которым идем по исходной картинке для вырезания тайлов

    _model = get_model(model_name=model_name, encoder_name=encoder_name,
                       encoder_weights=encoder_weights, activation=activation, in_channels=in_channels)

    # для вывода архитектуры сети, но будет понятнее, если просто посмотреть исходный код модели
    # _model.to('cuda')
    # # print(_model)
    # input = torch.tensor((3, 512, 512))
    # summary(_model, (3, 512, 512))
    # exit()
    new_version = False
    if 'device' in checkpoint:  # значит это новая версия проекта, и в чекпоинте есть ключ 'device'
        new_version = True

    # Если модель обучалась на cuda, она обернута в nn.DataParallel
    if (new_version and checkpoint['device'] == 'cuda') or not new_version:
        _model = nn.DataParallel(_model)
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model = _model.module.to(_device)  # убираем nn.DataParallel т.к. с ним не считается на cpu
    else: # Если обучалась без cuda, то была сохранена без nn.DataParallel
        _model.load_state_dict(checkpoint['model_state_dict'])

    _model.eval()

    with open(os.path.join(_output_path, "_metrics.csv"), "a") as evalfile:
        _metrics_names = 'file; iou; accuracy \n'  # если изменили порядок вычисления метрик в make_prediction,
        # нужно изменить эту переменную
        evalfile.write(_metrics_names)  # печатаем шапку .csv файла

        mean_metrics = [[], []]
        for i in range(len(valid_set)):
            image, mask = valid_set[i]
            res, metrics = make_prediction(_model, image, _tile_size, _step, _device, thres=args.thres, mask=mask)
            image_path = valid_set.image_set[i]
            image = read_image(image_path)
            if mask is not None:
                mask = valid_set.read_mask_for_img(image_path)
            # print(f'res shape is {res.shape}')
            # print(f'image shape is {image.shape}')
            # print(f'mask shape is {mask.shape}')
            filename = os.path.basename(valid_set.image_set[i])

            # сохраняем в файл для русского экселя
            evalfile.write(f"{filename}; " + f"{metrics['iou']:.4f}; {metrics['acc']:.4f}\n".replace('.', ','))
            i = 0
            for val in metrics.values():
                mean_metrics[i].append(val)
                i += 1

            if not b_calc_only_metrics:  # Сохраним бинарные картинки
                if args.binary:
                    print(f'Save {os.path.join(_output_path, filename)}')
                    cv2.imwrite(os.path.join(_output_path, filename), cv2.cvtColor(res['binary'], cv2.COLOR_RGB2BGR))

                if args.raw: # Сохраним выход сети без пороговой обработки
                    print(f'Save {os.path.join(_output_raw_path, filename)}')
                    cv2.imwrite(os.path.join(_output_raw_path, filename), res['raw'])

                if args.plots:  # сохраняем снимок/выход сети/ разметку в одну картинку в папку plots
                    name, ext = filename.split('.')
                    plot_name = name + f"_iou_{metrics['iou']:.2f}_acc_{metrics['acc']:.2f}." + ext
                    write_plots_and_visualize(  # (path_to_res, visualize=False, **images)
                        os.path.join(_output_plots_path, plot_name),
                        visualize=b_visualize,
                        image=image,
                        predicted=res['plot'],
                        mask=mask
                    )
        for i in range(len(mean_metrics)):
            mean_metrics[i] = statistics.mean(mean_metrics[i])
        evalfile.write(f"{'mean_vals'}; " + f"{mean_metrics[0]:.4f}; {mean_metrics[1]:.4f}\n".replace('.', ','))
        print(f"{'Mean vals over dataset are'}: " + f"iou {mean_metrics[0]:.4f}, acc {mean_metrics[1]:.4f}\n".replace('.', ','))


if __name__ == '__main__':
    main()
