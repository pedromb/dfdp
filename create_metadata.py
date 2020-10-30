import json
import pandas as pd
import numpy as np

from config import * 

if __name__ == "__main__":


    train = json.load(open('{}train.json'.format(SPLITS_FOLDER)))
    test = json.load(open('{}test.json'.format(SPLITS_FOLDER)))
    val = json.load(open('{}val.json'.format(SPLITS_FOLDER)))

    train = np.concatenate((np.flip(np.array(train), axis=1), np.array(train)))
    test = np.concatenate((np.flip(np.array(test), axis=1), np.array(test)))
    val = np.concatenate((np.flip(np.array(val), axis=1), np.array(val)))

    train_videos = [i[0] for i in train]
    test_videos = [i[0] for i in test]
    val_videos = [i[0] for i in val]
    
    all_videos = [*train_videos, *val_videos, *test_videos]
    all_videos = [i + '.mp4' for i in all_videos]
    
    split = [*["train"]*len(train_videos), *["val"]*len(val_videos), *["test"]*len(test_videos)]
    metadata = pd.DataFrame({"video":all_videos, "split":split}).sort_values("video")
    
    metadata.to_csv("{}".format(METADATA_FILE), index=False)