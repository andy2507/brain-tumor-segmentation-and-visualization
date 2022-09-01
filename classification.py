import nibabel as nib
import monai
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AddChanneld, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd, EnsureTyped, EnsureType


class PredictUsingDensenet():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "/home/andrea/Notebooks/FYP/monai/classification/best_metric_model_classification3d_dict.pth"
        self.model = self.get_model()
        self.post_pred = Compose([EnsureType(), Activations(softmax=True)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
        self.root_dir = "/home/andrea/Notebooks/FYP/frontend_html_css"
        # sample filename BraTS19_2013_10_1

    def get_model(self):
        if self.device!="cuda":
            model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=3)
            model.load_state_dict(torch.load(self.model_path,map_location=torch.device('cpu')))
        else:
            model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=3).to(self.device)
            model.load_state_dict(torch.load(self.model_path))
        return model
    
    def predict(self,filename):
        img = nib.load(os.path.join(self.root_dir,filename,filename+"_flair.nii.gz")).get_fdata()
        img = np.expand_dims(img,axis=0)
        img = np.expand_dims(img,axis=0)
        img = torch.from_numpy(img).to(self.device).float()
        outputs = self.model(img)
        outputs = self.post_pred(decollate_batch(outputs)[0]).cpu().numpy()
        return self.get_class(outputs)
            
    def get_class(self,x):
        if np.argmax(x)==2:
            return "Glioblastoma"
        if np.argmax(x)==1:
            return "Oligodendroglioma"
        else:
            return "Astrocytoma"

    