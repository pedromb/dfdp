import os
import cv2
import h5py
import math
import json
import torch
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from torchvision.models.resnet import *
import pretrainedmodels
from efficientnet_pytorch import EfficientNet

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F


class CNNModel(nn.Module):

    def __init__(self, backbone="resnet34", pretrained_backbone=False):
        super(CNNModel, self).__init__()

        if backbone == "resnet34" or backbone == 0:
            self.backbone = resnet34(pretrained=pretrained_backbone)
            self.backbone.fc = nn.Identity()
            self.output_size = 512
        elif backbone == "se_resnext50_32x4d" or backbone == 2:
            pretrained_resnext = 'imagenet' if pretrained_backbone else None
            self.backbone = pretrainedmodels.se_resnext50_32x4d(num_classes=1000, pretrained=pretrained_resnext)
            self.backbone.last_linear = nn.Identity()
            self.output_size = 2048
        elif backbone == "se_resnext101_32x4d" or backbone == 3:
            pretrained_resnext = 'imagenet' if pretrained_backbone else None
            self.backbone = pretrainedmodels.se_resnext101_32x4d(num_classes=1000, pretrained=pretrained_resnext)
            self.backbone.last_linear = nn.Identity()
            self.output_size = 2048
        elif backbone == "efficientnet-b7" or backbone == 4:
            if pretrained_backbone:
                self.backbone = EfficientNet.from_pretrained('efficientnet-b7')
            else:
                self.backbone = EfficientNet.from_name('efficientnet-b7')
            self.backbone._fc = nn.Identity()
            self.output_size = 2560
        else:
            self.backbone = backbone
            self.output_size = self.backbone.output_size

        self.classifier = nn.Sequential(
            nn.Linear(self.output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(128, 32),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(32, 1)    
        )

    def forward(self, x):
        features = self.backbone(x)
        y = self.classifier(features)
        return y

class ModelWrapper():
    
    def __init__(
        self, 
        model = None, 
        optimizer=None,
        loss_function=None,
        train_generator=None, 
        scheduler = None,
        val_generator = None, 
        load_model=None,
        save_checkpoint=None,
        early_stopping_rounds = None,
        stacked_models = None
    ):
        self.device_name = "cuda:{}".format(self.__get_freer_gpu()) if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.save_checkpoint = save_checkpoint
        self.load_model = load_model
        self.early_stopping_rounds = early_stopping_rounds
        self.stacked_models = stacked_models
        if self.load_model is not None:
            self.__load_model()
        else:
            self.epoch = -1
            self.best_acc = float("-Inf")
            self.load_model = self.save_checkpoint
            self.model.to(self.device)

    
    def __load_model(self):
        checkpoint = torch.load(self.load_model, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_acc = checkpoint['acc']
        self.model.to(self.device)
        print()
        print("Model loaded, best acc = {}".format(self.best_acc))
        print()
    
    def __get_freer_gpu(self):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        os.remove("tmp")
        return np.argmax(memory_available)

    def __make_train_step(self):
        def train_step(x, y):
            self.model.train()
            yhat = self.model(x)
            yhat = yhat.view(yhat.size(0))
            loss = self.loss_function(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            return loss.item()
        return train_step
    
    def __save_checkpoint(self, new_best_acc):
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "acc": new_best_acc
        }, self.save_checkpoint)
    
    def train(self, epochs):
        train_step = self.__make_train_step()
        control_early_stopping = self.early_stopping_rounds
        for epoch in range(self.epoch+1, self.epoch+epochs+1):
            losses = []
            total = len(self.train_generator)
            if self.val_generator is not None:
                total += len(self.val_generator)
            pbar = tqdm(total=total, desc = "Epoch {}".format(epoch))
            found_one = False
            for i, data in enumerate(self.train_generator):
                x, y = data
                y = y.to(self.device)
                x = x.to(self.device)
                loss = train_step(x, y)
                losses.append(loss)
                pbar.update(1)
                if i == len(self.train_generator) - 1:
                    train_loss = np.mean(losses)
                    acc = 0.0000
                    if self.val_generator is not None:
                        acc = self.evaluate(self.val_generator, pbar)
                    if (acc > self.best_acc) or (self.val_generator is None):
                        message = self.__get_desc_message(epoch, train_loss, acc)
                        pbar.set_description(message)
                        if self.save_checkpoint is not None:
                            self.__save_checkpoint(acc)
                        if self.early_stopping_rounds is not None:
                            control_early_stopping = self.early_stopping_rounds
                        self.best_acc = acc
                    elif self.early_stopping_rounds is not None:
                        control_early_stopping -= 1
            self.epoch += 1
            pbar.close(); del pbar
            if control_early_stopping <= 0:
                self.__load_model()
                break
                        
    def evaluate(self, data_generator, pbar=None, use_tqdm=False):
        with torch.no_grad():
            if use_tqdm:
                data_generator = tqdm(data_generator, total=len(data_generator))
            self.model.eval()
            y_labels = []
            y_preds = []
            for x, y in data_generator:
                x = x.to(self.device)
                yhat = self.model(x)
                yhat = yhat.view(yhat.size(0))
                yhat = torch.sigmoid(yhat)
                yhat = yhat.detach().cpu().numpy()
                ylabel = y.detach().cpu().numpy()
                y_preds.extend(yhat)
                y_labels.extend(ylabel)
                if pbar is not None:
                    pbar.update(1)
            y_preds = np.clip(y_preds, 0.000001, 0.999999)
            y_preds_acc = np.array([1 if i > 0.5 else 0 for i in y_preds])
            acc = np.around(np.mean(y_labels == y_preds_acc), 4)
            return acc
    
    def predict(self, data_generator):
        with torch.no_grad():
            frames = []
            labels = []
            predictions = []
            mms = []
            self.model.eval()
            for x, y, mm, frame in tqdm(data_generator):  
            
                x = x.to(self.device)
                yhat = self.model(x)
                yhat = yhat.view(yhat.size(0))
                yhat = torch.sigmoid(yhat)
                yhat = yhat.cpu().numpy()
                y = y.cpu().numpy()
                frame = frame.cpu().numpy()

                labels.extend(y)
                mms.extend(mm)
                predictions.extend(yhat)
                frames.extend(frame)


        preds = pd.DataFrame({
            "manipulation_method": mms,
            "labels": labels,
            "predictions": predictions,
            "frame": frames
        })

        return preds
    
    def predict_benchmark(self, data_generator):
        with torch.no_grad():
            filenames = []
            predictions = []
            self.model.eval()
            for x, f_name in tqdm(data_generator):
                x = x.to(self.device)
                x = x.squeeze(0)
                yhat = self.model(x)
                yhat = yhat.view(yhat.size(0))
                yhat = torch.sigmoid(yhat)
                yhat = yhat.cpu().numpy()
                pred = np.max(yhat)
                f_name = f_name[0]
                predictions.append(pred)
                filenames.append(f_name)

        preds = pd.DataFrame({
            "filename": filenames,
            "predictions": predictions,
        })

        return preds    

    def __get_desc_message(self, epoch, train_loss, val_acc):
        message = "Epoch {0: <3} - train_loss {1:.4f} | val_acc {2:.4f}".format(epoch, train_loss, val_acc)
        return message
 