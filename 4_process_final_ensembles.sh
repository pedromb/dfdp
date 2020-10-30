#!/bin/sh

## Create final ensembles
python create_ensembles_all.py

## Evaluate final ensembles
python evaluate.py -n resnet_best -if /data/models/resnet_best
python evaluate.py -n resnext_best -if /data/models/resnext_best
python evaluate.py -n efficientnet_best -if /data/models/efficientnet_best
python evaluate.py -n all_best -if /data/models/all_best
python evaluate.py -n all_best_acc -if /data/models/all_best_acc
python evaluate.py -n all_best_real -if /data/models/all_best_real

