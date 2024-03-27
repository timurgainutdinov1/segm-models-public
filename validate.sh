#!/bin/sh

source segm_models/bin/activate

CPU_TRAIN=true
CHECK_DIR="/misc/home1/u0304/segm-models/logs/water_deepglobe/efficientnet-b0_bsize_8_hard"
DATASET_PATH="/misc/home6/m_imm_freedata/Segmentation/RG3/val"
# Наборы лежат тут /misc/home6/m_imm_freedata/Segmentation

# Не трогать, переменные для переключения между cpu и gpu
PYTHON_CPU=""
if [ $CPU_TRAIN == true ]
then
  PYTHON_CPU="--cpu"
fi

# ПРИМЕЧАНИЕ: после слова wrap вставлять все строки в одинарных кавычках
python3.9 validate.py \
-cd $CHECK_DIR \
-d $DATASET_PATH \
$PYTHON_CPU \

# --workers 1
# смотрите какое num_workers рекоменуют вам в warnings в логах обучения

