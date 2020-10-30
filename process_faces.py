'''
Save faces into h5py files.
This files will later be used for training, validation and test
'''


from data import *
from config import *

from tqdm import tqdm
import numpy as np
import argparse
import h5py
import os
import cv2

from multiprocessing import Pool


def convert_faces(faces):
    first_face = faces[0]
    h, w = first_face.shape[:2]
    if h > 224 or w > 224:
        h, w = 224, 224
    faces = np.array([cv2.resize(f, (h, w)) for f in faces])
    return faces

def run_fast_scandir(dir, ext):
    subfolders, files = [], []
    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)
    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def save_h5py_file(faces, label, video_id, manipulation_method, file_to_save):
    h5py_file = h5py.File(file_to_save, "a", swmr=True)
    faces_data = convert_faces(faces)
    dataset = h5py_file.create_dataset("frames", faces_data.shape, data=faces_data, dtype=h5py.h5t.STD_U8BE)
    dataset.attrs["label"] = label
    dataset.attrs["manipulation_method"] = manipulation_method
    dataset.attrs["video_id"] = video_id
    h5py_file.close()

def process_single_entry(filename):
    label = all_files[filename]['label']
    file_id = all_files[filename]['file_id']
    manipulation_method = all_files[filename]['manipulation_method']
    f_name = all_files[filename]['h5py_filename']
    if file_id in faces_coords:
        frames_coordinates = faces_coords[file_id]

        new_coordinates = {}
        frames_to_collect_idxs = list(frames_coordinates.keys())[:N_FRAMES]
        for i in frames_to_collect_idxs:
            start_x, end_x, start_y, end_y = frames_coordinates[i]
            size_x = int((end_x - start_x)*FACE_MARGIN)
            size_y = int((end_y - start_y)*FACE_MARGIN)
            start_x = start_x - size_x//2 if start_x >= size_x//2 else 0
            start_y = start_y - size_y//2 if start_y >= size_y//2 else 0
            end_x += size_x//2
            end_y += size_y//2
            if start_y < 0: start_y = 0
            if start_x < 0: start_x = 0
            new_coordinates[i] = (start_x, end_x, start_y, end_y)

        faces = retrieve_faces(filename, new_coordinates, frames_to_collect_idxs)
        save_h5py_file(faces, label, file_id, manipulation_method, f_name)

if __name__ == '__main__':


    PROCESS_FRAMES_CONFIG = {
        "n_frames": N_FRAMES,
        "clip_hist_percent": 0,
        "verbose": False,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--face_margin", type=int, default=0, help="The face margin to use when extracting faces")
    parser.add_argument("-c", "--cpu_cores", type=int, default=4, help="Number of cores to use to process the data")
    args = parser.parse_args()
    face_margin = args.face_margin
    cpu_cores = args.cpu_cores
    processed_data_folder = PROCESSED_DATA_FOLDER[face_margin]

    FACE_MARGIN = face_margin/100
    metadata = pd.read_csv(METADATA_FILE)

    manip_map = {'Deepfakes':'df', 'Face2Face':'f2f', 'FaceSwap': 'fs', 'NeuralTextures':'nt'}
    _, real_files = run_fast_scandir(RAW_DATA_FOLDER_REAL, [".mp4"])
    _, fake_files = run_fast_scandir(RAW_DATA_FOLDER_FAKES, [".mp4"])
    all_files = {}

    for r in real_files:
        file_id =  r.split('/')[-1].split('.')[0]
        metadata_file_name = file_id + '.mp4'
        compression = r.split('/')[-3]
        split = metadata.loc[metadata.video == metadata_file_name].split.values[0]
        if (split == "train") or (split == "val" and compression == "c40") or (split == "test" and compression == "c40"):
            f_name = processed_data_folder + '{}_real_{}.h5py'.format(file_id, compression)
            all_files[r] = {
                'label':0, 
                'file_id': file_id, 
                'manipulation_method': 'real', 
                'split': split,
                'h5py_filename': f_name
            }

    for f in fake_files:
        file_id =  f.split('/')[-1].split('.')[0].split('_')[0]
        metadata_file_name = file_id + '.mp4'
        compression = f.split('/')[-3]
        split = metadata.loc[metadata.video == metadata_file_name].split.values[0]
        manipulation_method = manip_map[f.split('/')[-4]]
        if (split == "train") or (split == "val" and compression == "c40") or (split == "test" and compression == "c40"):
            f_name = processed_data_folder + '{}_{}_{}.h5py'.format(file_id, manipulation_method, compression)
            all_files[f] = {
                'label':1, 
                'file_id': file_id, 
                'manipulation_method': manipulation_method, 
                'split': split, 
                'h5py_filename': f_name
            }

    faces_coords = np.load(FACES_COORDINATES, allow_pickle=True)[()]
    process_frames = ProcessFrames(**PROCESS_FRAMES_CONFIG)
    retrieve_faces = RetrieveFaces(process_frames)


    with Pool(cpu_cores) as p:
        _ = list(tqdm(p.imap(process_single_entry, list(all_files.keys())), total=len(list(all_files.keys()))))

    frames_to_manipulation_method_train = {}
    train_h5py = h5py.File(processed_data_folder + "train.h5py", "a", swmr=True)
    val_h5py = h5py.File(processed_data_folder + "val.h5py", "a", swmr=True)
    test_h5py = h5py.File(processed_data_folder + "test.h5py", "a", swmr=True)

    train_frame = 0
    val_frame = 0
    test_frame = 0

    for key, entry in tqdm(all_files.items()):
        try:
            h5py_filename = entry["h5py_filename"]
            split = entry["split"]

            aux_file = h5py.File(h5py_filename, "r")
            data = aux_file["frames"]
            label = data.attrs["label"]
            manipulation_method = data.attrs["manipulation_method"]
            video_id = data.attrs["video_id"]
        
            for frame in data:
                if split == "train":
                    dataset = train_h5py.create_dataset(str(train_frame), frame.shape, data=frame, dtype=h5py.h5t.STD_U8BE)
                    frames_to_manipulation_method_train[str(train_frame)] = manipulation_method
                    train_frame += 1
                elif split == "val":
                    dataset = val_h5py.create_dataset(str(val_frame), frame.shape, data=frame, dtype=h5py.h5t.STD_U8BE)
                    val_frame += 1
                elif split == "test":
                    dataset = test_h5py.create_dataset(str(test_frame), frame.shape, data=frame, dtype=h5py.h5t.STD_U8BE)
                    test_frame += 1
                    
                dataset.attrs["label"] = label
                dataset.attrs["manipulation_method"] = manipulation_method
                dataset.attrs["video_id"] = video_id

            aux_file.close()
            os.remove(h5py_filename)
        except:
            pass
                        


    train_h5py.close()
    val_h5py.close()
    test_h5py.close()
    np.save('{}frames_to_mm_train.npy'.format(processed_data_folder), frames_to_manipulation_method_train)