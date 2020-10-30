'''
'''

import pandas as pd
import numpy as np
import h5py

from tqdm import tqdm

from config import PROCESSED_FACES_MARGIN_0, PROCESSED_FACES_MARGIN_10, PROCESSED_FACES_MARGIN_30, PROCESSED_FACES_MARGIN_50


def balance_dataset(dataset):
    idxs_by_mm = {}
    balanced_idxs_by_mm = {}
    for key in dataset:
        mm = dataset[key].attrs["manipulation_method"]
        if mm in idxs_by_mm:
            idxs_by_mm[mm].append(key)
        else:
            idxs_by_mm[mm] = [key]
    n_real = len(idxs_by_mm["real"])
    n_fake = n_real // 4
    balanced_idxs_by_mm["real"] = idxs_by_mm["real"]
    for key in idxs_by_mm.keys():
        if key != "real":
            selected_idxs = np.random.choice(idxs_by_mm[key], n_fake)
            balanced_idxs_by_mm[key] = selected_idxs
    return balanced_idxs_by_mm

if __name__ == "__main__":

    for i in tqdm([PROCESSED_FACES_MARGIN_0, PROCESSED_FACES_MARGIN_10, PROCESSED_FACES_MARGIN_30, PROCESSED_FACES_MARGIN_50]):

        val = h5py.File(i + "val.h5py", "r")
        test = h5py.File(i + "test.h5py", "r")

        balanced_val = h5py.File(i + "val_balanced.h5py", "a", swmr=True)
        balanced_test = h5py.File(i + "test_balanced.h5py", "a", swmr=True)

        balanced_val_idxs = balance_dataset(val)
        balanced_test_idxs = balance_dataset(test)

        val_frame = 0
        test_frame = 0

        for mm in balanced_val_idxs.keys():
            mm_idxs = balanced_val_idxs[mm]
            for idx in mm_idxs:
                frame = val[idx]
                data = frame[:]
                dataset = balanced_val.create_dataset(str(val_frame), data.shape, data=data, dtype=h5py.h5t.STD_U8BE)
                dataset.attrs["label"] = frame.attrs['label']
                dataset.attrs["manipulation_method"] = frame.attrs['manipulation_method']
                dataset.attrs["video_id"] = frame.attrs['video_id']
                val_frame += 1

        for mm in balanced_test_idxs.keys():
            mm_idxs = balanced_test_idxs[mm]
            for idx in mm_idxs:
                frame = test[idx]
                data = frame[:]
                dataset = balanced_test.create_dataset(str(test_frame), data.shape, data=data, dtype=h5py.h5t.STD_U8BE)
                dataset.attrs["label"] = frame.attrs['label']
                dataset.attrs["manipulation_method"] = frame.attrs['manipulation_method']
                dataset.attrs["video_id"] = frame.attrs['video_id']
                test_frame += 1

        balanced_val.close()
        balanced_test.close()

