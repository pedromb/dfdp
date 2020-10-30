#!/bin/sh

## Train best resnext and efficientnet models (based on resnet34 results)

# Best ACC
python train.py -m efficientnet-b7 -i 1 -f 10
python train.py -m efficientnet-b7 -i 1 -f 30
python train.py -m efficientnet-b7 -i 1 -f 30 -da low
python train.py -m efficientnet-b7 -i 2 -f 10
python train.py -m efficientnet-b7 -i 2 -f 30
python train.py -m efficientnet-b7 -i 2 -f 30 -da low
python train.py -m efficientnet-b7 -i 3 -f 10
python train.py -m efficientnet-b7 -i 3 -f 30
python train.py -m efficientnet-b7 -i 3 -f 30 -da low

python train.py -m se_resnext101_32x4d -i 1 -f 10
python train.py -m se_resnext101_32x4d -i 1 -f 30
python train.py -m se_resnext101_32x4d -i 1 -f 30 -da low
python train.py -m se_resnext101_32x4d -i 2 -f 10
python train.py -m se_resnext101_32x4d -i 2 -f 30
python train.py -m se_resnext101_32x4d -i 2 -f 30 -da low
python train.py -m se_resnext101_32x4d -i 3 -f 10
python train.py -m se_resnext101_32x4d -i 3 -f 30
python train.py -m se_resnext101_32x4d -i 3 -f 30 -da low


# Best Real
python train.py -m se_resnext101_32x4d -i 1 -f 0 -eq True -da high
python train.py -m se_resnext101_32x4d -i 1 -f 10 -eq True -da high
python train.py -m se_resnext101_32x4d -i 1 -f 10 -da high
python train.py -m se_resnext101_32x4d -i 2 -f 0 -eq True -da high
python train.py -m se_resnext101_32x4d -i 2 -f 10 -eq True -da high
python train.py -m se_resnext101_32x4d -i 2 -f 10 -da high
python train.py -m se_resnext101_32x4d -i 3 -f 0 -eq True -da high
python train.py -m se_resnext101_32x4d -i 3 -f 10 -eq True -da high
python train.py -m se_resnext101_32x4d -i 3 -f 10 -da high

python train.py -m efficientnet-b7 -i 1 -f 0 -eq True -da high
python train.py -m efficientnet-b7 -i 1 -f 10 -eq True -da high
python train.py -m efficientnet-b7 -i 1 -f 10 -da high
python train.py -m efficientnet-b7 -i 2 -f 0 -eq True -da high
python train.py -m efficientnet-b7 -i 2 -f 10 -eq True -da high
python train.py -m efficientnet-b7 -i 2 -f 10 -da high
python train.py -m efficientnet-b7 -i 3 -f 0 -eq True -da high
python train.py -m efficientnet-b7 -i 3 -f 10 -eq True -da high
python train.py -m efficientnet-b7 -i 3 -f 10 -da high

# Best Fake
python train.py -m se_resnext101_32x4d -i 1 -f 0
python train.py -m se_resnext101_32x4d -i 1 -f 50
python train.py -m se_resnext101_32x4d -i 2 -f 0
python train.py -m se_resnext101_32x4d -i 2 -f 50
python train.py -m se_resnext101_32x4d -i 3 -f 0
python train.py -m se_resnext101_32x4d -i 3 -f 50

python train.py -m efficientnet-b7 -i 1 -f 0
python train.py -m efficientnet-b7 -i 1 -f 50
python train.py -m efficientnet-b7 -i 2 -f 0
python train.py -m efficientnet-b7 -i 2 -f 50
python train.py -m efficientnet-b7 -i 3 -f 0
python train.py -m efficientnet-b7 -i 3 -f 50

## Evaluate models

python evaluate.py -n se_resnext101_32x4d_0_None_histogram_not_equalized -if /data/models/se_resnext101_32x4d_0_None_histogram_not_equalized
python evaluate.py -n se_resnext101_32x4d_0_high_histogram_equalized -if /data/models/se_resnext101_32x4d_0_high_histogram_equalized
python evaluate.py -n se_resnext101_32x4d_10_None_histogram_not_equalized -if /data/models/se_resnext101_32x4d_10_None_histogram_not_equalized
python evaluate.py -n se_resnext101_32x4d_10_high_histogram_equalized -if /data/models/se_resnext101_32x4d_10_high_histogram_equalized
python evaluate.py -n se_resnext101_32x4d_10_high_histogram_not_equalized -if /data/models/se_resnext101_32x4d_10_high_histogram_not_equalized
python evaluate.py -n se_resnext101_32x4d_30_None_histogram_not_equalized -if /data/models/se_resnext101_32x4d_30_None_histogram_not_equalized
python evaluate.py -n se_resnext101_32x4d_30_low_histogram_not_equalized -if /data/models/se_resnext101_32x4d_30_low_histogram_not_equalized
python evaluate.py -n se_resnext101_32x4d_50_None_histogram_not_equalized -if /data/models/se_resnext101_32x4d_50_None_histogram_not_equalized

python evaluate.py -n efficientnet-b7_0_None_histogram_not_equalized -if /data/models/efficientnet-b7_0_None_histogram_not_equalized
python evaluate.py -n efficientnet-b7_0_high_histogram_equalized -if /data/models/efficientnet-b7_0_high_histogram_equalized
python evaluate.py -n efficientnet-b7_10_None_histogram_not_equalized -if /data/models/efficientnet-b7_10_None_histogram_not_equalized
python evaluate.py -n efficientnet-b7_10_high_histogram_equalized -if /data/models/efficientnet-b7_10_high_histogram_equalized
python evaluate.py -n efficientnet-b7_10_high_histogram_not_equalized -if /data/models/efficientnet-b7_10_high_histogram_not_equalized
python evaluate.py -n efficientnet-b7_30_None_histogram_not_equalized -if /data/models/efficientnet-b7_30_None_histogram_not_equalized
python evaluate.py -n efficientnet-b7_30_low_histogram_not_equalized -if /data/models/efficientnet-b7_30_low_histogram_not_equalized
python evaluate.py -n efficientnet-b7_50_None_histogram_not_equalized -if /data/models/efficientnet-b7_50_None_histogram_not_equalized

## Create ensembles
python create_ensembles_resnext_efficientnet.py

## Evaluate ensembles
python evaluate.py -n resnext_best_acc -if /data/models/resnext_best_acc
python evaluate.py -n resnext_best_fake -if /data/models/resnext_best_fake
python evaluate.py -n resnext_best_real -if /data/models/resnext_best_real
python evaluate.py -n resnext_best_comb -if /data/models/resnext_best_comb

python evaluate.py -n efficientnet_best_acc -if /data/models/efficientnet_best_acc
python evaluate.py -n efficientnet_best_real -if /data/models/efficientnet_best_real
python evaluate.py -n efficientnet_best_comb -if /data/models/efficientnet_best_comb
