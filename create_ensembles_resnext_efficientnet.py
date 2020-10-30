
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

    best_acc_folder_resnext = MODELS_FOLDER + "resnext_best_acc/"
    best_real_folder_resnext = MODELS_FOLDER + "resnext_best_real/"
    best_fake_folder_resnext = MODELS_FOLDER + "resnext_best_fake/"
    best_comb_folder_resnext = MODELS_FOLDER + "resnext_best_comb/"

    best_acc_folder_efficientnet = MODELS_FOLDER + "efficientnet_best_acc/"
    best_real_folder_efficientnet = MODELS_FOLDER + "efficientnet_best_real/"
    best_comb_folder_efficientnet = MODELS_FOLDER + "efficientnet_best_comb/"

    new_folders = [
        best_acc_folder_resnext, best_real_folder_resnext, best_fake_folder_resnext, best_comb_folder_resnext,
        best_acc_folder_efficientnet, best_real_folder_efficientnet, best_comb_folder_efficientnet
    ]

    for f in new_folders:
        if not os.path.exists(f):
            os.mkdir(f)

    best_acc_folders_resnext = [
        MODELS_FOLDER + "se_resnext101_32x4d_10_None_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_10_high_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_30_None_histogram_not_equalized/",
    ] 

    best_real_folders_resnext = [
        MODELS_FOLDER + "se_resnext101_32x4d_10_high_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_10_high_histogram_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_0_high_histogram_equalized/",
    ] 

    best_fake_folders_resnext = [
        MODELS_FOLDER + "se_resnext101_32x4d_10_None_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_0_None_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_30_low_histogram_not_equalized/",
    ]

    best_comb_folders_resnext = [
        MODELS_FOLDER + "se_resnext101_32x4d_10_None_histogram_not_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_10_high_histogram_equalized/",
        MODELS_FOLDER + "se_resnext101_32x4d_0_None_histogram_not_equalized/",
    ]

    best_acc_folders_efficientnet = [
        MODELS_FOLDER + "efficientnet-b7_30_low_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_30_None_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_50_None_histogram_not_equalized/",
    ] 

    best_real_folders_efficientnet = [
        MODELS_FOLDER + "efficientnet-b7_10_high_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_10_high_histogram_equalized/",
        MODELS_FOLDER + "efficientnet-b7_0_high_histogram_equalized/",
    ] 

    best_comb_folders_efficientnet = [
        MODELS_FOLDER + "efficientnet-b7_30_low_histogram_not_equalized/",
        MODELS_FOLDER + "efficientnet-b7_10_high_histogram_not_equalized/"
    ]


    copy_folders(best_acc_folders_resnext, best_acc_folder_resnext)
    copy_folders(best_real_folders_resnext, best_real_folder_resnext)
    copy_folders(best_fake_folders_resnext, best_fake_folder_resnext)
    copy_folders(best_comb_folders_resnext, best_comb_folder_resnext)

    copy_folders(best_acc_folders_efficientnet, best_acc_folder_efficientnet)
    copy_folders(best_real_folders_efficientnet, best_real_folder_efficientnet)
    copy_folders(best_comb_folders_efficientnet, best_comb_folder_efficientnet)