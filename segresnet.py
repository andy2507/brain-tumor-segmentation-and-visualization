# imports
import os
from pyexpat import model
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
)
from monai.utils import set_determinism
import nibabel as nib
import sys
import torch

class PredictWithSegResNet():
    '''
    Predict using SegResNet
    '''
    def __init__(self,file_name):
        self.ckpt_path = '/home/andrea/Notebooks/FYP/Flask_file/best_metric_model.pth'
        self.root_directory = os.getcwd()
        self.filename = file_name
        self.result_dest = os.path.join(self.root_directory,self.filename,'results',"prediction_segmentation.nii.gz")
        if torch.cuda.is_available():  
            self.device = "cuda:0" 
        else:  
            self.device = "cpu" 
        self.model = self.get_model(self.ckpt_path)
        self.img = self.get_img()
        self.img = self.normalise(self.img)
        self.VAL_AMP = True
        self.post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])  

    def get_model(self,model_path):
        '''
        Load the SegResNet model with the pretrained weights
        '''
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        ).to(self.device)
        if self.device=='cuda:0':
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        return model

    def get_img(self):
        '''
        Returns concatenation of all four modalities
        '''
        flair_img = np.expand_dims(nib.load(os.path.join(self.root_directory,self.filename,self.filename+'_flair.nii.gz')).get_fdata(),axis=0)
        t1_img = np.expand_dims(nib.load(os.path.join(self.root_directory,self.filename,self.filename+'_t1.nii.gz')).get_fdata(),axis=0)
        t1ce_img = np.expand_dims(nib.load(os.path.join(self.root_directory,self.filename,self.filename+'_t1ce.nii.gz')).get_fdata(),axis=0)
        t2_img = np.expand_dims(nib.load(os.path.join(self.root_directory,self.filename,self.filename+'_t2.nii.gz')).get_fdata(),axis=0)

        flair_img = np.rot90(flair_img,1,axes=(1,2))
        t1_img = np.rot90(t1_img,1,axes=(1,2))
        t1ce_img = np.rot90(t1ce_img,1,axes=(1,2))
        t2_img = np.rot90(t2_img,1,axes=(1,2))

        img = np.concatenate((flair_img,t1_img,t1ce_img,t2_img))
        return img
    
    def normalise(self,img):
        '''
        Max Normalize the concatenated modalities
        '''
        for i in range(img.shape[0]):
            arr = img[i,:,:,:]
            arr = arr[arr!=0]
            img[i,:,:,:][img[i,:,:,:]!=0] = (img[i,:,:,:][img[i,:,:,:]!=0]-np.mean(arr))/np.std(arr)
        return img
    
    def get_y_pred(self):
        '''
        To split the prediction into enhancing, non-enhancing and core tumor
        -> blue: non-enhancing tumor (1)
        -> yellow: edema (2)
        -> green: enhancing tumor (4)
        '''
        y_pred_prob = self.predict(self.img)
        y_pred_prob = y_pred_prob.cpu().numpy()
        y_pred=np.zeros((1,240,240,155))

        y_pred[0]=np.where(y_pred_prob[1]==1,2,y_pred[0])  
        y_pred[0]=np.where(y_pred_prob[0]==1,4,y_pred[0])
        y_pred[0]=np.where(y_pred_prob[2]==1,1,y_pred[0])
        y_pred = np.rot90(y_pred,3,axes=(1,2))


        return y_pred
    
    def predict(self,arr):
        '''
        To get the prediction from the model 
        '''
        if arr.shape!=(1,4,240,240,155):
            arr = np.expand_dims(arr,axis=0)
        roi_size = (128, 128, 64)
        sw_batch_size = 4
        input = {}
        input['image']=torch.from_numpy(arr).float()
        output = self.inference(input['image'])
        output = self.post_trans(output[0])

        return output
    
    def inference(self,input):
        '''
        Helper fn to get prediction
        '''
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=self.model,
                overlap=0.5,
            )

        if self.VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)
        
    def get_seg(self):
        '''
        Returns the ground truth
        '''
        seg = nib.load(os.path.join(self.root_directory,self.filename,self.filename+'_seg.nii.gz')).get_fdata()
        seg = np.expand_dims(seg,axis=0)
        seg = np.rot90(seg,1,axes=(1,2))
        return seg

    def show_segmentation_on_image(self,image, label):
        '''
        Returns segmentation on image
        '''
        ones = np.argwhere(label == 1)
        twos = np.argwhere(label == 2)
        fours = np.argwhere(label == 4)

        image = image/image.max()

        image = np.expand_dims(image,axis=-1)
        label = np.expand_dims(label,axis=-1)
        image = np.concatenate((image,image,image),axis=-1)
        label = np.concatenate((label,label,label),axis=-1)
        red_multiplier = [1, 0.2, 0.2]
        green_multiplier = [0.35,0.75,0.25]
        blue_multiplier = [0,0.5,1.]
        yellow_multiplier = [1,1,0.25]
        brown_miltiplier = [40./255, 26./255, 13./255]

        for i in range(len(ones)):
            image[ones[i][0]][ones[i][1]] = blue_multiplier
        for i in range(len(twos)):
            image[twos[i][0]][twos[i][1]] = yellow_multiplier 
        for i in range(len(fours)):
            image[fours[i][0]][fours[i][1]] = green_multiplier

        return image
  
    def show_slice(self,flair_img,y_pred,seg):
        '''
        Plots the segmentation on the slice
        '''
        pred_img = self.show_segmentation_on_image(flair_img[0,:,:,70],y_pred[0,:,:,70])
        seg_img = self.show_segmentation_on_image(flair_img[0,:,:,70],seg[0,:,:,70])
        # os.remove('/home/andrea/Notebooks/FYP/Flask_file/static/img/pred_img.jpg')
        # os.remove('/home/andrea/Notebooks/FYP/Flask_file/static/img/seg_img.jpg')
        # plt.imsave('/home/andrea/Notebooks/FYP/Flask_file/static/img/pred_img.jpg',pred_img)
        # plt.imsave('/home/andrea/Notebooks/FYP/Flask_file/static/img/seg_img.jpg',seg_img)


    def get_flair(self):
        '''
        Returns Flair image
        '''
        flair_img = np.expand_dims(nib.load(os.path.join(self.root_directory,self.filename,self.filename+'_flair.nii.gz')).get_fdata(),axis=0)
        return flair_img
    
    def save_nifti(self,y_pred):
        '''
        Saves the predicted file as a .nii file
        '''
        affine = [[-1,-0,-0,0],[-0,-1,-0,239],[0,0,1,0],[0,0,0,1]]
        y_pred_img = nib.Nifti1Image(y_pred[0], affine)
        nib.save(y_pred_img, self.result_dest)
    
    def return_dest(self):
        return self.result_dest


    