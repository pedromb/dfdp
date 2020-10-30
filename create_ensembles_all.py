
import torch.nn as nn
import argparse
import os

import shutil
from data import *
from models import *
from config import MODELS_FOLDER

def copy_files(folder_src, folder_dest):
    src_files = os.listdir(folder_src)
    for file_name in src_files:
        full_file_name = os.path.join(folder_src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, folder_dest)


def copy_folders(folders_src, folder_dist):
    for folder in folders_src: copy_files(folder, folder_dist)

if __name__ == '__main__':


    all_best_acc = MODELS_FOLDER + "all_best_acc/"
    all_best_real = MODELS_FOLDER + "all_best_real/"
    all_best_fake = MODELS_FOLDER + "all_best_fake/"

    resnet = MODELS_FOLDER + "resnet_best/"
    resnext = MODELS_FOLDER + "resnext_best/"
    efficientnet = MODELS_FOLDER + "efficientnet_best/"
    all_best = MODELS_FOLDER + "all_best/"

    new_folders = [
        all_best_acc, all_best_real,
        resnet, resnext, efficientnet, all_best
    ]

    for f in new_folders:
        if not os.path.exists(f):
            os.mkdir(f)

    all_best_acc_folders = [
        MODELS_FOLDER + "resnet34_10_None_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_10_None_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_30_low_histogram_not_equalized/"
    ] 

    all_best_real_folders = [
        MODELS_FOLDER + "efficientnet-b7_10_high_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_0_high_histogram_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_10_high_histogram_equalized/"
    ] 

    resnet_folders = [
        MODELS_FOLDER + "resnet34_0_None_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_0_high_histogram_equalized/",
        MODELS_FOLDER + "resnet34_10_None_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_10_high_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_10_high_histogram_equalized/",
        MODELS_FOLDER + "resnet34_30_None_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_30_low_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_50_None_histogram_not_equalized/"
    ] 

    resnext_folders = [
        MODELS_FOLDER + "se_resnext101_32x4d_0_None_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_0_high_histogram_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_10_None_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_10_high_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_10_high_histogram_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_30_None_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_30_low_histogram_not_equalized/"
    ] 

    efficientnet_folders = [
        MODELS_FOLDER + "efficientnet-b7_30_low_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_30_None_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_50_None_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_10_high_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_10_high_histogram_equalized/",
        MODELS_FOLDER + "efficientnet-b7_0_high_histogram_equalized/"
    ] 

    all_best_folders = [
        MODELS_FOLDER + "resnet34_0_None_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_0_high_histogram_equalized/",
        MODELS_FOLDER + "resnet34_10_None_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_10_high_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_10_high_histogram_equalized/",
        MODELS_FOLDER + "resnet34_30_None_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_30_low_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_50_None_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_0_None_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_0_high_histogram_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_10_None_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_10_high_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_10_high_histogram_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_30_None_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_30_low_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_30_low_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_30_None_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_50_None_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_10_high_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_10_high_histogram_equalized/",
        MODELS_FOLDER + "efficientnet-b7_0_high_histogram_equalized/"

    ] 

    copy_folders(all_best_acc_folders, all_best_acc)
    copy_folders(all_best_real_folders, all_best_real)
    copy_folders(resnet_folders, resnet)
    copy_folders(resnext_folders, resnext)
    copy_folders(efficientnet_folders, efficientnet)
    copy_folders(all_best_folders, all_best)