import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import os
import argparse
from pytorch_toolbelt.utils.random import set_manual_seed
from dataset import TilesDataset
from trainer import SegmentationTrainer


def main():
    parser = argparse.ArgumentParser(description='Train model for segmentation task'
                                                 'Usage: python train.py -d D:\Vector_data\Water\test_dataset')
    # Model
    parser.add_argument("-m", "--model", type=str, default="deeplabv3+", help="model name, default deeplabv3+")
    parser.add_argument("--encoder", type=str, default="efficientnet-b0", help="default efficientnet-b0")
    parser.add_argument("--image-size", default=512, help="default=512")
    # в папке logs создается папка с именем эксперимента, если такой эксперимент уже был,
    # то будет загружаться чекпоинт и обучение будет продолжаться, если имя не задано оно сгенерируется автоматически:
    # exp_name = model_name + ' ' + encoder_name + ' ' + str(datetime.date(datetime.now()))
    parser.add_argument("--exp-name", type=str, default="", help="name of the experiment")
    # Masks
    parser.add_argument("--class-list", type=str, default="255",
                        help="classes from mask to train network, example: '1 2 5'")  # числа разделенные пробелами
    # использовать только если маска содержит метки от 1 до max-mask-val, чтобы не перечислять их все в class-list
    parser.add_argument("--max-mask-val", type=int, default=0,
                        help="set this value only if your mask contain labels from 1 to max-mask-val")
    # когда не хотим обучать модель выделять все классы с маски а только конкретные, класс 0 - фон указывать не нужно
    # по умолчанию '255' - означает что на маске два значения - 0 и 255
    # размер батча должен быть четным числом, а число примеров в наборе также четным
    parser.add_argument("-b", "--batch", type=int, default=6, help="batch size default=6, must be an even number")
    parser.add_argument("-lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="default=50")
    parser.add_argument("-a", "--augmentation", type=str, default="hard", help="default=hard, medium light safe")
    parser.add_argument("-w", "--workers", type=int, default=1, help="default=6")
    # Dirs
    parser.add_argument("-d", "--dataset", type=str, required=True, help="path where dataset is placed")
    # Если хотим взять данные из нескольких наборов, нужно собрать в .txt файл список файлов из всех папок images,
    # которые нас интересуют.
    # parser.add_argument("-lt", "--train-img-list", type=str, help="List of images from datasets")
    parser.add_argument("--add-dirs", type=str, default=None,
                        help="List of additional channels for input. Example: 'nir kv' additional images should be \
                             1-channel")
    # заполнить нулем rgb каналы во время обучения
    # parser.add_argument('--ignore-rgb', action='store_true', help="fill 0 rgb channels during training")
    # в папке logs создаются папки с экспериментами
    parser.add_argument("-log", "--log-dir", type=str, default="./logs/",
                        help="default='./logs/' path where logs of experiments are placed")
    parser.add_argument("-val", "--val-dir", type=str, default=None,
                        help="path for val dataset when train and val sets placed in different folders")
    # Если хотим проверять качество сразу на нескольких val наборах, нужно указать путь к .txt файлу,
    # где в отдельных строках пути к наборам
    parser.add_argument("--add-val-dirs", type=str, default=None,
                        help="path to txt file with paths to additional val dirs")
    parser.add_argument("--adv-freq", type=int, default=1, help="how often start evaluation on add-val-dirs")
    # не запускать валидацию на папке val, считать только miou на папках из списка add-val
    parser.add_argument('--use-only-add-val', action='store_true', help="validate only on add-val dirs")
    parser.add_argument('--cpu', action='store_true', help="use cpu for training")
    parser.add_argument("--device", type=str, default="cuda", help="default=cuda, or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed default=42")
    args = parser.parse_args()

    # Фиксируем генерацию случайных чисел для повторяемости результатов
    set_manual_seed(args.seed)  # fix seed in nympy and pytorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.cpu:
        args.device = 'cpu'

    if args.add_dirs is not None:
        args.add_dirs = args.add_dirs.split()

    # test configuration: -d D:\Vector_data\Water\test_dataset --cpu --log-dir=../logs -e 1 -b 2 -w 1

    # TODO считать среднее по каналам
    #  1. проверить возможность мультиклассовой сегментации
    #  4. Добавить возможность обучаться на данных с переменным числом каналов
    #  V 5. проверять модель сразу на нескольких val наборах
    #  6. добавить HSV аугментацию
    #  7. добавить наложение теней от облаков

    class_list = [int(x) for x in args.class_list.split()]
    train_set = TilesDataset(os.path.join(args.dataset, 'train') if args.dataset is not None else None,
                             args.image_size,
                             augmentation=args.augmentation,
                             max_mask_val=args.max_mask_val,
                             class_list=class_list,
                             add_dirs=args.add_dirs,
                             # images_list_path=args.train_img_list
                             )

    assert len(train_set) % args.batch != 1, (f"Train dataset len is {len(train_set)} batch_size is {args.batch}. "
                                              f"len(train_set) % batch_size == 1. Please use a different batch size."
                                              f"The training code cannot process a batch with size 1.")

    # train_set.test_augment(r'D:\Vector_data\Water\test_dataset\train_test_aug', 10)
    valid_set = None
    add_valid_set_list = []

    if not args.use_only_add_val:
        if args.val_dir is not None:
            # Используем папку val из args.val_dir
            valid_set = TilesDataset(args.val_dir,
                                     args.image_size,
                                     max_mask_val=args.max_mask_val,
                                     class_list=class_list,
                                     add_dirs=args.add_dirs,
                                     )
        else:
            # Используем папку val из набора данных
            valid_set = TilesDataset(os.path.join(args.dataset, 'val'),
                                     args.image_size,
                                     max_mask_val=args.max_mask_val,
                                     class_list=class_list,
                                     add_dirs=args.add_dirs,
                                     )
    # Проверяем модель сразу на нескольких val наборах
    if args.add_val_dirs is not None:
        with open(args.add_val_dirs) as file:
            lines = [line.rstrip() for line in file]
        for line in lines:
            print(f'Read additional val dataset from {line}')
            add_valid_set_list.append(
                TilesDataset(line,
                             args.image_size,
                             max_mask_val=args.max_mask_val,
                             class_list=class_list,
                             add_dirs=args.add_dirs,
                             )
            )

    # valid_set.test_augment(r'D:\Vector_data\Water\test_dataset\valid_test_aug', 10)

    print('Параметры переданные в скрипт')
    print("Passed arguments: ", str(args).replace(',', ',\n'))
    # print("Train session    :", args.exp_name)
    # print("  Model          :", args.model)
    # print("  Encoder        :", args.encoder)
    # print("  Max class label       :", args.max_mask_val)
    # print("  Class list to train        :", class_list)
    # print("  Dataset dir       :", args.dataset)
    # print("  Additional data dirs       :", args.add_dirs)
    # print("  Log dir        :", args.log_dir)
    # print("  Additional val dir        :", args.val_dir)
    # print("  Augmentations  :", args.augmentation)
    # # print("  FP16 mode      :", fp16)
    # print("  Epochs         :", args.epochs)
    # print("  Batch size     :", args.batch)
    # print("  Learning rate     :", args.lr)
    # print("  Image size     :", args.image_size)
    # print("  Workers        :", args.workers)
    # print("  Device        :", args.device)
    print("  Train size     :", "dataset", len(train_set))
    print("  Val size     :", "dataset", len(valid_set) if valid_set is not None else 0)
    for i in range(len(add_valid_set_list)):
        print(f"  Additional val set {i}     :", "dataset", len(valid_set))
    # print("Optimizer        :", optimizer_name)

    trainer = SegmentationTrainer(
        train_set,
        valid_set,
        add_valid_set_list,
        use_only_add_val=bool(args.use_only_add_val),
        add_val_freq=int(args.adv_freq),
        model_name=args.model,
        exp_name=args.exp_name,
        log_dir=args.log_dir,
        epochs_count=int(args.epochs),
        learning_rate=float(args.lr),
        train_batch_size=int(args.batch),
        valid_batch_size=int(args.batch),
        train_workers_count=int(args.workers),
        valid_workers_count=int(args.workers),
        encoder_name=args.encoder,
        device=args.device
    )

    trainer.start_training()


if __name__ == '__main__':
    main()
