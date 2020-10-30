import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import h5py
import numpy as np
import pandas as pd

from facenet_pytorch import MTCNN
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms

from albumentations import (
    ImageCompression, Blur, GaussNoise, HueSaturationValue, 
    RandomBrightnessContrast, RandomGamma, Compose, OneOf,
    MotionBlur, MedianBlur, IAASharpen, IAAEmboss, IAAAdditiveGaussianNoise,
    ShiftScaleRotate, RandomCrop, ToGray, ToSepia, Cutout, CenterCrop,
    OpticalDistortion, GridDistortion, Downscale
)

class ProcessFrames():
    
    def __init__(
        self, 
        n_frames = 300,
        clip_hist_percent = 1,
        verbose = True,
    ):
        self.n_frames = n_frames
        self.clip_hist_percent = clip_hist_percent
        self.verbose = verbose

    def __automatic_brightness_and_contrast(self, image, clip_hist_percent):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray],[0],None,[256],[0,256])
            hist_size = len(hist)
            accumulator = []
            accumulator.append(float(hist[0]))
            for index in range(1, hist_size):
                accumulator.append(accumulator[index -1] + float(hist[index]))
            maximum = accumulator[-1]
            clip_hist_percent *= (maximum/100.0)
            clip_hist_percent /= 2.0
            minimum_gray = 0
            while accumulator[minimum_gray] < clip_hist_percent:
                minimum_gray += 1
            maximum_gray = hist_size -1
            while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
                maximum_gray -= 1

            alpha = 255 / (maximum_gray - minimum_gray)
            beta = -minimum_gray * alpha
            auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            return auto_result
        except:
            return image

    def __capture_frames(self, video_file, frames_to_collect):
        vidcap = cv2.VideoCapture(video_file)
        vid_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        jump = (vid_length + 1) // self.n_frames
        if jump == 0:
            jump = 1
        all_frames = []
        flag = True
        count = 0
        secondary_count = 0
        frames_captured = []
        if self.verbose:
            pbar = tqdm(range(vid_length), desc = "Capturing video frames...")
        else:
            pbar = range(vid_length)
        for i in pbar:
            ret = vidcap.grab()
            retrieve_frame = False
            if frames_to_collect is not None and i in frames_to_collect:
                retrieve_frame = True
            elif frames_to_collect is None and i % jump == 0:
                retrieve_frame = True
            if retrieve_frame:
                flag, frame = vidcap.retrieve()
                if flag:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    if self.clip_hist_percent > 0:
                        img_bgr = self.__automatic_brightness_and_contrast(img_bgr, self.clip_hist_percent)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    all_frames.append(img_rgb)
                    frames_captured.append(i)
                    count += 1
            if self.verbose:
                pbar.update(1)
                secondary_count += 1
            if count == self.n_frames:
                left = vid_length - secondary_count
                if self.verbose and left > 0:
                    pbar.update(left)
                break
        all_frames = all_frames[:min(count, self.n_frames)]
        frames_captured = frames_captured[:min(count, self.n_frames)]
        return all_frames, frames_captured

    def __call__(self, video_file, frames_to_collect=None):
        return self.__capture_frames(video_file, frames_to_collect)

class FaceDetection():
    
    def __init__(
        self, 
        batch_size = 10,
        thresholds = [(0.9,[0.6, 0.7, 0.7])],
        verbose = True,
        return_scores = False,
        face_margin = 0
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.thresholds = thresholds
        self.batch_size = batch_size
        self.verbose = verbose
        self.return_scores = return_scores
        self.face_margin = face_margin

    def __process_face_coords(self, box, h_scale, w_scale, h, w):
        scale_x = h_scale / h
        scale_y = w_scale / w
        start_y, start_x, end_y, end_x = box
        start_x, end_x = int(start_x*scale_x), int(end_x*scale_x)
        start_y, end_y = int(start_y*scale_y), int(end_y*scale_y)
        size_x = int((end_x - start_x)*self.face_margin)
        size_y = int((end_y - start_y)*self.face_margin)
        start_x = start_x - size_x//2 if start_x >= size_x//2 else 0
        start_y = start_y - size_y//2 if start_y >= size_y//2 else 0
        end_x += size_x//2
        end_y += size_y//2
        return start_x, end_x, start_y, end_y

    def __detect_faces(self, frames, frames_idxs):
        video_faces_coords = {}
        if self.verbose:
            pbar = tqdm(range(0, len(frames), self.batch_size), desc="Detecting faces...")
        else:
            pbar = range(0, len(frames), self.batch_size)
        for idx in pbar:
            data = frames[idx:idx+self.batch_size]
            f_idxs = frames_idxs[idx:idx+self.batch_size]
            h_scale, w_scale = data[0].shape[:2]
            found = False
            for th, thresholds in self.thresholds:
                self.mtcnn = MTCNN(margin=0, keep_all=True, thresholds=thresholds, post_process=False, device=self.device)
                for h, w in [(480, 480), (640, 640), (1280, 1280)]:
                    new_data = [Image.fromarray(cv2.resize(i, (h, w))) for i in data]
                    faces = self.mtcnn.detect(new_data)
                    faces_coords = []
                    n_empty = 0
                    for i in range(len(faces[0])):
                        filtered_results = []
                        if faces[0][i] is not None:
                            faces_on_frame = [self.__process_face_coords(list(i), h_scale, w_scale, h, w) for i in list(faces[0][i].astype(int))]
                            faces_confidence = faces[1][i]
                            results = list(zip(faces_on_frame, faces_confidence))
                            for i in results:
                                if i[1] > th:
                                    if self.return_scores:
                                        filtered_results.append(i)
                                    else:
                                        filtered_results.append(i[0])
                        faces_coords.append(filtered_results)
                        if len(filtered_results) == 0:
                            n_empty += 1
                    if n_empty / self.batch_size < 0.2:
                        found = True
                        break
                if found:
                    break
            
            for j, i in enumerate(f_idxs):
                video_faces_coords[i] = faces_coords[j]

        return video_faces_coords
        
    def __call__(self, frames, frames_idxs):
        faces = self.__detect_faces(frames, frames_idxs)      
        return faces

class FacesCoordinates():

    def __init__(
        self,
        process_frames,
        face_detect
    ):
        self.process_frames = process_frames
        self.face_detect = face_detect            

    def __call__(self, video_files):
        final_coord = {}
        for idx, v_file in enumerate(tqdm(video_files)):
            frames, frames_idxs = self.process_frames(v_file)
            faces_coord = self.face_detect(frames, frames_idxs)
            final_coord[v_file] = faces_coord
        return final_coord

class RetrieveFaces():

    def __init__(self, process_frames, face_detect=None):
        self.process_frames = process_frames
        self.face_detect = face_detect

    def __get_faces_from_coords_single(self, face_coords, frames):
        faces = []
        for frame_idx, frame in frames.items():
            start_x, end_x, start_y, end_y = face_coords[frame_idx]
            faces.append(frame[start_x:end_x, start_y:end_y])
        return faces
    
    def __get_faces_from_coords_multiple(self, face_coords, frames):
        faces = []
        for frame_idx, frame in frames.items():
            faces_and_scores = face_coords[frame_idx]
            frame_faces = []
            for coords, score in faces_and_scores:
                start_x, end_x, start_y, end_y = coords
                face = frame[start_x:end_x, start_y:end_y]
                frame_faces.append((face, score))
            faces.append(frame_faces)
        return faces

    def __call__(self, video_file, faces_coords = None, frames_to_collect = None, single=True):
        frames, frames_idxs = self.process_frames(video_file, frames_to_collect)
        if faces_coords is None and self.face_detect is not None:
            faces_coords = self.face_detect(frames, frames_idxs)
        frames = {frames_idxs[i]:frames[i] for i in range(len(frames))}
        if single:
            faces = self.__get_faces_from_coords_single(faces_coords, frames)
        else:
            faces = self.__get_faces_from_coords_multiple(faces_coords, frames)
        return faces

class DataAugmentation():

    def __init__(self, level=None):
        self.level = level
        
    def __call__(self, img):
        if self.level is None:
            return img
        else:
            augmentations_possible = []
            if self.level in ['low', 'medium', 'high']:
                augmentations_possible.append(
                    OneOf([
                        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                        Downscale(scale_min=0.7, scale_max=0.99, p=0.5)
                    ], p=0.5)
                )

            if self.level in ['medium', 'high']:
                augmentations_possible.append(
                    OneOf([
                        IAASharpen(p=.3),
                        IAAEmboss(p=.3),
                        RandomBrightnessContrast(p=.3), 
                        RandomGamma(p=.3),
                    ], p=0.5)
                )
            if self.level == "high":
                augmentations_possible.append(
                    OneOf([
                        MotionBlur(blur_limit=7, p=.3),
                        MedianBlur(blur_limit=7, p=.3),
                        Blur(blur_limit=7, p=.3)
                    ], p=0.5)
                )
                augmentations_possible.append(
                    OneOf([
                        IAAAdditiveGaussianNoise(p=.3),
                        GaussNoise(p=.3),
                    ], p=0.5)
                )
            aug = Compose(augmentations_possible, p=1)
            img = aug(image=img)["image"]
            return img

class DataTransformation():

    def __init__(
        self, 
        resize=(224, 224), 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225],
        equalize_img = True,
    ):
        self.resize = resize
        self.mean = mean
        self.std = std
        self.equalize_img = equalize_img

    def automatic_brightness_and_contrast(self, image, clip_hist_percent):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        try:
            hist = cv2.calcHist([gray], [0] ,None, [256], [0,256])
            hist_size = len(hist)
            accumulator = []
            accumulator.append(float(hist[0]))
            for index in range(1, hist_size):
                accumulator.append(accumulator[index -1] + float(hist[index]))
            maximum = accumulator[-1]
            clip_hist_percent *= (maximum/100.0)
            clip_hist_percent /= 2.0
            minimum_gray = 0
            while accumulator[minimum_gray] < clip_hist_percent:
                minimum_gray += 1
            maximum_gray = hist_size -1
            while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
                maximum_gray -= 1

            alpha = 255 / (maximum_gray - minimum_gray)
            beta = -minimum_gray * alpha
        except:
            alpha = 2
            beta = -40
        auto_result = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
        auto_result = cv2.cvtColor(auto_result, cv2.COLOR_BGR2RGB)
        return auto_result

    def __call__(self, X):
        if not np.all(X == 0):
            if self.equalize_img:
                X = self.automatic_brightness_and_contrast(X, 1.0)
            X = X / 255
            if self.resize is not None:
                X = cv2.resize(X, self.resize)
            if self.mean is not None and self.std is not None:
                normalize = transforms.Normalize(
                    mean = self.mean,
                    std = self.std
                )
                X = torch.from_numpy(X.T)
                X = normalize(X).float()
                X = X.numpy().T
        elif self.resize is not None:
            X = cv2.resize(X, self.resize)
        return X.T.astype("float32")

class FFDataset(data.Dataset):

    def __init__(
        self,
        split = "train",
        h5py_file = None,
        frame_to_mm_train = None,
        data_transformation = None,
        data_augmentation = None

    ):
        self.split = split
        self.h5py_file = h5py_file
        self.data_transformation = data_transformation
        self.data_augmentation = data_augmentation
        self.frame_to_mm_train = frame_to_mm_train

        if self.split != "train":
            h5py_file = h5py.File(self.h5py_file, "r")
            self.len = len(h5py_file.keys())
        else:
            self.frame_to_mm_train = np.load(self.frame_to_mm_train, allow_pickle=True)[()]
            self.len = len(self.frame_to_mm_train)

    def __len__(self):
        return self.len
    
    def __getitem_train(self, index):
        h5py_file = h5py.File(self.h5py_file, "r")
        data = h5py_file[str(index)]
        X = data[:]
        y = data.attrs["label"]
        if self.data_augmentation is not None:
            X = self.data_augmentation(X)
        X = self.data_transformation(X)
        return np.array(X), y

    def __getitem_val(self, index):
        h5py_file = h5py.File(self.h5py_file, "r")
        data = h5py_file[str(index)]
        X = data[:]
        y = data.attrs["label"]
        X = self.data_transformation(X)
        return np.array(X), y

    def __getitem_test(self, index):
        
        h5py_file = h5py.File(self.h5py_file, "r")
        frame = str(index)
        dataset = h5py_file[frame]
        label = dataset.attrs["label"]
        mm = dataset.attrs["manipulation_method"]
        X = self.data_transformation(dataset[:])
        return X, label, mm, float(frame)

    def __getitem__(self, index):
        if self.split == "train":
            X, y = self.__getitem_train(index)
        elif self.split == "val":
            X, y = self.__getitem_val(index)
        elif self.split == "test":
            X, y, mm, frame = self.__getitem_test(index)
            return X, float(y), mm, frame
        return X, float(y)

class FFDatasetBenchmark(data.Dataset):

    def __init__(
        self,
        folder = "data/faceforensics_benchmark_images",
        data_transformation = None,
        face_detector = None
    ):
        self.data_transformation = data_transformation
        self.face_detector = face_detector
        self.image_files = [folder + "/{}".format(i) for i in os.listdir(folder) if 'png' in i]

    def __len__(self):
        return len(self.image_files)
    
    def __get_faces_from_coords_multiple(self, img, faces_coords):
        faces = []
        faces_coords = faces_coords[0]
        for coords in faces_coords:
            start_x, end_x, start_y, end_y = coords
            face = img[start_x:end_x, start_y:end_y]
            faces.append(face)
        return faces
    
    def __getitem__(self, index):
        img_file = self.image_files[index]
        img_file_name = img_file.split("/")[-1]
        img = np.array(Image.open(img_file))
        faces_coords = self.face_detector([img], [0])
        faces = self.__get_faces_from_coords_multiple(img, faces_coords)
        faces = np.array([self.data_transformation(i) for i in faces])
        return faces, img_file_name

class DataSampler(data.Sampler):


    def __init__(self, dataset, sample_size=32000):
        self.sample_size = sample_size
        self.mm_to_frames = self.__indexes_to_mm(dataset.frame_to_mm_train)

    def __indexes_to_mm(self, frames_to_mm):
        mm_to_frames = {}
        for key, item in frames_to_mm.items():
            if item != "real": item = "fake"
            if item in mm_to_frames:
                mm_to_frames[item].append(int(key))
            else:
                mm_to_frames[item] = [int(key)]
        return mm_to_frames


    def __iter__(self):
        indexes_to_return = []
        mm_size = self.sample_size // 2
        for m in self.mm_to_frames:
            choices = np.random.choice(self.mm_to_frames[m], mm_size, replace=False)
            indexes_to_return.extend(choices)
        np.random.shuffle(indexes_to_return)
        return iter(indexes_to_return)

    def __len__(self):
        return self.sample_size