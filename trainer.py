import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_toolbelt.utils import count_parameters
import os
from datetime import datetime
from model import get_model
# from torchsummary import summary

class SegmentationTrainer:
    def __init__(
            self,
            train_set,
            valid_set,
            valid_set_list,
            use_only_add_val=False,  # Если не False значит в valid_set_list на 0 позиции val set из набора
                                     # Если True значит в этом списке только доп. валидационные наборы
                                     # и по val набору ничего не считаем
            add_val_freq=1,
            exp_name='',
            log_dir='../logs/',
            model_name='unet',
            encoder_name='efficientnet-b0',
            encoder_weights='imagenet',
            activation='sigmoid',
            device='cuda',
            epochs_count=50,
            learning_rate=0.001,
            train_batch_size=2,
            valid_batch_size=2,
            train_workers_count=1,
            valid_workers_count=1,
    ):
        self.encoder_name = encoder_name
        self.model_name = model_name
        self.encoder_weights = encoder_weights
        self.activation = activation
        self.device = device
        self.epochs_count = epochs_count
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.train_set = train_set
        self.use_only_add_val = use_only_add_val
        self.add_val_freq = add_val_freq

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        if not exp_name:
            self.exp_name = model_name + ' ' + encoder_name + ' ' + str(datetime.date(datetime.now()))
        print(f'Num classes to train: {train_set.num_classes}')
        self._model = get_model(model_name=model_name,
                                encoder_name=encoder_name,
                                encoder_weights=encoder_weights,
                                activation=activation,
                                classes=train_set.num_classes,
                                in_channels=train_set.in_channels
                                )

        print("  Parameters     :", count_parameters(self._model))
        if self.device == 'cpu':
            self._model.to(self.device)  # убираем nn.DataParallel т.к. с ним не считается на cpu
        else:
            self._model = nn.DataParallel(self._model)

        # summary(self._model, (3, 128, 128), device='cpu')

        self._train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_workers_count,
            pin_memory=self.device
        )
        # print(f'Shape of batch {train_set[0]}')
        num_elements = 0
        self.valid_loader = {  # основной val loader
                'set': DataLoader(
                    valid_set,
                    batch_size=valid_batch_size,
                    shuffle=False,
                    num_workers=valid_workers_count,
                    pin_memory=self.device
                ),
                'name': valid_set.name,
                'weight': len(valid_set)
            }
        num_elements += len(valid_set)

        self.valid_loader_list = []
        for val_set in valid_set_list:
            self.valid_loader_list.append({
                'set': DataLoader(
                    val_set,
                    batch_size=valid_batch_size,
                    shuffle=False,
                    num_workers=valid_workers_count,
                    pin_memory=self.device
                ),
                'name': val_set.name,
                'weight': len(val_set)
            })
            num_elements += len(val_set)
        for loader in self.valid_loader_list:
            loader['weight'] /= num_elements

        self.valid_loader['weight'] /= num_elements

        self._loss = smp_utils.losses.DiceLoss()
        self._metrics = [
            smp_utils.metrics.IoU(threshold=0.5),
        ]

        self._optimizer = torch.optim.Adam([
            dict(params=self._model.parameters(), lr=learning_rate),
        ])

        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, self.epochs_count)

        self._train_epoch = smp_utils.train.TrainEpoch(
            self._model,
            loss=self._loss,
            metrics=self._metrics,
            optimizer=self._optimizer,
            device=self.device,
            verbose=True,
        )

        self._valid_epoch = smp_utils.train.ValidEpoch(
            self._model,
            loss=self._loss,
            metrics=self._metrics,
            device=self.device,
            verbose=True,
        )

    def start_training(self):
        print(f'Запуск обучения модели с энкодером {self.encoder_name}')
        logs_path = os.path.join(self.log_dir, self.exp_name)
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)

        checkpoint_name = f'{logs_path}/{self.model_name}_{self.encoder_name}_best_model.pth'
        writer = SummaryWriter(logs_path)

        if self.valid_loader_list is not None:
            layout = {
                "Validation": {
                    "Loss": ["Multiline", [loader['name']
                                           + ' loss' for loader in self.valid_loader_list + [self.valid_loader]]],
                    "IOU": ["Multiline", [loader['name']
                                          + ' iou' for loader in self.valid_loader_list + [self.valid_loader]]],
                },
            }
            writer.add_custom_scalars(layout)

        max_score = 0
        if os.path.exists(checkpoint_name):
            print(f'Load checkpoint from {checkpoint_name}')
            checkpoint = torch.load(checkpoint_name)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch'] + 1
            self._loss = checkpoint['loss']  # TODO нужно ли? берет ту же функцию потерь, которая была в чекпоинте
            max_score = checkpoint['max_score']
        else:
            epoch = 1

        for i in range(epoch, epoch + self.epochs_count):
            print('\nEpoch: {}'.format(i))
            mean_val_iou = 0

            train_logs = self._train_epoch.run(self._train_loader)
            writer.add_scalar('Accuracy/train', train_logs['iou_score'], i)
            writer.add_scalar('Loss/train', train_logs['dice_loss'], i)

            # Валидация
            if not self.use_only_add_val:  # если в списке есть основной val набор считаем iou по нему
                valid_logs = self._valid_epoch.run(self.valid_loader['set'])
                writer.add_scalar(self.valid_loader['name'] + ' iou', valid_logs['iou_score'], i)
                writer.add_scalar(self.valid_loader['name'] + ' loss', valid_logs['dice_loss'], i)
                val_iou = valid_logs['iou_score']
                if self.valid_loader_list is not None:
                    mean_val_iou += valid_logs['iou_score'] * self.valid_loader['weight']

            # Считаем mean_val_iou по нескольким наборам
            if self.valid_loader_list is not None:
                validate_now = (i % self.add_val_freq) == 0
                if validate_now:
                    for loader in self.valid_loader_list:
                        valid_logs = self._valid_epoch.run(loader['set'])
                        writer.add_scalar(loader['name'] + ' iou', valid_logs['iou_score'], i)
                        writer.add_scalar(loader['name'] + ' loss', valid_logs['dice_loss'], i)
                        mean_val_iou += valid_logs['iou_score'] * loader['weight']

            # считаем либо только по основному набору либо только по доп. наборам
            iou_value = mean_val_iou if self.use_only_add_val else val_iou

            with open(os.path.join(logs_path, "_iou_per_epoch_val.csv"), "a") as evalfile:
                # сохраняем в файл для русского экселя
                evalfile.write(f"{i}; {iou_value:.4f}\n".replace('.', ','))

            if max_score < iou_value:
                max_score = iou_value
                torch.save({
                    'epoch': i,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'loss': self._loss,
                    'model_name': self.model_name,
                    'encoder_name': self.encoder_name,
                    'encoder_weights': self.encoder_weights,
                    'activation': self.activation,
                    'max_score': max_score,
                    'mean_var': self.train_set.mean_var,
                    'class_list': self.train_set.class_list,
                    'add_dirs': self.train_set.add_dirs,
                    'device': self._device,
                }, checkpoint_name)  # checkpoint_name + '_iou_{:.2f}_epoch_{}.pth'.format(self._max_score, i))
                print(f'Model saved at {checkpoint_name}')

            self._scheduler.step()
            print(self._scheduler.get_last_lr())
        writer.close()
