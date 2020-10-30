'''

To accelerate train and test we pre extract the face coordinates that will be used.
We retrieve 100 frames from each video for training, 100 for validation and 100 for test
We will use c40 quality for validation and test and c23 and c40 for training.
However to speed up computation we detect the faces on the original sequences using c23 quality and extend to other qualitites and to the manipulated sequences.
We only keep the largest found face for simplicity.
'''

import pandas as pd
import numpy as np
import torch

from config import * 
from data import *


if __name__ == "__main__":

    PROCESS_FRAMES_CONFIG = {
        "n_frames": N_FRAMES,
        "clip_hist_percent": 1,
        "verbose": False,
    }

    FACE_DETECTION_CONFIG = {
        "batch_size": BATCH_SIZE,
        "verbose": False,
        "return_scores": False,
        "face_margin": 0
    }

    process_frames = ProcessFrames(**PROCESS_FRAMES_CONFIG)
    face_detection = FaceDetection(**FACE_DETECTION_CONFIG)
    face_coordinates = FacesCoordinates(process_frames, face_detection)

    metadata = pd.read_csv(METADATA_FILE)
    video_files = ["{}c23/videos/{}".format(RAW_DATA_FOLDER_REAL, row.video)  for _, row in metadata.iterrows()]

    faces_coords = face_coordinates(video_files)
    faces_coords_filtered = {}
    for v_file, f_coords in faces_coords.items():
        frames = list(f_coords.keys())
        new_faces_coords = {}
        for f in frames:
            try:
                faces = f_coords[f]
                faces_len = [(f, (f[1]-f[0])*(f[3] - f[2])) for f in faces]
                face_to_keep = sorted(faces_len, key=lambda x: x[1])[0][0]
                new_faces_coords[f] = face_to_keep
            except:
                pass
        file_id = v_file.split('/')[-1].split('.')[0]
        faces_coords_filtered[file_id] = new_faces_coords

    np.save('{}'.format(FACES_COORDINATES), faces_coords_filtered)