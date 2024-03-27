#!/bin/sh

source segm_models/bin/activate

CPU_TRAIN=false
BATCH=8
EPOH=100
ENCODER="efficientnet-b0"
AUGM="hard"
DSET_NAME="deepglobe"
MODEL="deeplabv3+"
# на какой целевой класс сейчас обучаем
CLASS="water"
CLASS_LIST="64"
IMAGE_SIZE="512"
# лог обучения сохранится в файл ./logs/$ENCODER"_batch_"$BATCH"_%j" где j - номер задачи на кластере
# имя эксперимента задается ниже в этом файле exp-name=$ENCODER'_bsize_'$BATCH'_'$AUGM
# имя эксперимента используется для создания папки, где лежит сохраненная модель
# если запустить эксперимент с тем же именем, то обучение продолжится из чекопоинта
# обученная модель сохранится по следующему пути: './logs/'$CLASS'_'$DSET_NAME/exp-name

# Не трогать, переменные для переключения между cpu и gpu
PYTHON_CPU=""
if [ $CPU_TRAIN == true ]
then
  PYTHON_CPU="--cpu"
fi

python3.9 train.py \
--dataset='/misc/home6/m_imm_freedata/Segmentation/DeepGlobe_Land/DeepGlobe512' \
--batch=$BATCH \
--model=$MODEL \
--encoder=$ENCODER \
--augmentation=$AUGM \
--exp-name=$ENCODER'_bsize_'$BATCH'_'$AUGM \
-log './logs/'$CLASS'_'$DSET_NAME \
--workers=18 \
--epochs=$EPOH \
--class-list=$CLASS_LIST \
--image-size=$IMAGE_SIZE \
$PYTHON_CPU \


# --workers 1
# смотрите какое num_workers рекоменуют вам в warnings в логах обучения