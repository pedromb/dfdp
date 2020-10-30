
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

    best_acc_folder = MODELS_FOLDER + "resnet34_best_acc/"
    best_real_folder = MODELS_FOLDER + "resnet34_best_real/"
    best_fake_folder = MODELS_FOLDER + "resnet34_best_fake/"
    best_comb_folder = MODELS_FOLDER + "resnet34_best_comb/"
    new_folders = [best_acc_folder, best_real_folder, best_fake_folder, best_comb_folder]
    
    for f in new_folders:
        if not os.path.exists(f):
            os.mkdir(f)

    best_acc_folders = [
        MODELS_FOLDER + "resnet34_10_None_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_30_None_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_30_low_histogram_not_equalized/",
    ] 

    best_real_folders = [
        MODELS_FOLDER + "resnet34_0_high_histogram_equalized/",
        MODELS_FOLDER + "resnet34_10_high_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_10_high_histogram_equalized/",
    ] 

    best_fake_folders = [
        MODELS_FOLDER + "resnet34_10_None_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_50_None_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_0_None_histogram_not_equalized/",
    ]

    best_comb_folders = [
        MODELS_FOLDER + "resnet34_10_None_histogram_not_equalized/",
        MODELS_FOLDER + "resnet34_0_high_histogram_equalized/",
        MODELS_FOLDER + "resnet34_50_None_histogram_not_equalized/",
    ]

    copy_folders(best_acc_folders, best_acc_folder)
    copy_folders(best_real_folders, best_real_folder)
    copy_folders(best_fake_folders, best_fake_folder)
    copy_folders(best_comb_folders, best_comb_folder)