import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob
import shutil
import sysconfig
from PIL import Image
from pyexpat import model
import shutil
import tempfile
import time
import nibabel as nib
import sys
import torch

class YoloModel():
  '''
  Class to load pretrained Yolo model 
  '''
  def __init__(self):
    self.ckpt_path = '/home/andrea/Notebooks/FYP/monai/best.pt' 
    self.img_size  = 240
    self.conf      = 0.25
    self.iou       = 0.5
    self.augment   = True
  
  def load_model(self):
    device="cpu"
    model = torch.hub.load('ultralytics/yolov5',
                           'custom',path=self.ckpt_path,force_reload=True, device="cpu")
                          #  path=self.ckpt_path,
                          #  device=torch.device("cpu"),force_reload=True)  
    model.conf = self.conf  # NMS confidence threshold
    model.iou  = self.iou  # NMS IoU threshold
    model.classes = None   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1  # maximum number of detections per image
    return model

class PredictionWithYolo():
  '''
  Class to Predict with YOLO and save results into a .csv file
  '''
  def __init__(self,filename):
    self.root_dir = "/home/andrea/Notebooks/FYP/frontend_html_css"
    self.filename = filename
    self.file_flair = os.path.join(self.root_dir,self.filename,self.filename+'_flair.nii.gz')
    self.output_folder_slices = f'{self.root_dir}/{self.filename}'
    self.output_csv = os.path.join(self.root_dir,self.filename,'results','prediction_yolo_bboxes.csv')
    self.yolomodel = YoloModel()
    self.model = self.yolomodel.load_model()
    if 'temp' not in os.listdir(self.root_dir):
        os.mkdir(os.path.join(self.root_dir,'temp'))
    self.temp_dir = os.path.join(self.root_dir,'temp')
    self.result_dest = os.path.join(self.filename,'results','prediction_yolo.nii.gz')

  def slices(self):
    '''
    Saving slices of the .nii files into the folder
    '''
    os.chdir('med2image')
    os.system(f'med2image -i {self.file_flair} -d {self.temp_dir}')

  def predict_yolo(self, img):
    '''
    Predicting with YOLO given img slice
    '''
    height, width = img.shape[:2]
    results = self.model(img, size=self.yolomodel.img_size, augment=False)  # custom inference size
    preds   = results.pandas().xyxy[0]
    bboxes  = preds[['xmin','ymin','xmax','ymax']].values
    classes = preds['class'].values
    return classes,bboxes

  def show_img_bbox(self,img,classes,bboxes):
    '''
    Draws bounding box over img
    '''
    image_bbox=np.zeros((240,240))
    if len(bboxes)==0 or classes[0]==1:
      return image_bbox
    color=(255,0,0)
    image_bbox = cv2.rectangle(image_bbox,(int(bboxes[0][0]),int(bboxes[0][1])),(int(bboxes[0][2]),int(bboxes[0][3])),color,2)
    return image_bbox

  def get_results(self):
    '''
    Plots results and save it to a csv file
    '''
    df = pd.DataFrame(columns = ['slice','class','xmin','ymin','xmax','ymax'])
    yolo_img = np.zeros((1,240,240,155))
    for idx,file in enumerate(sorted(os.listdir(self.temp_dir))):
        path = os.path.join(self.temp_dir,file)
        img = plt.imread(path)[...,::-1]
        classes,bboxes = self.predict_yolo(img)
        img_bbox = self.show_img_bbox(img,classes,bboxes)
        if len(img_bbox.shape)==2:
            yolo_img[0,:,:,idx] = img_bbox
        else:
            yolo_img[0,:,:,idx] = img_bbox[:,:,0]
        df.loc[idx,'slice']=idx
        if(len(bboxes)!=0):
            df.loc[idx,['xmin','ymin','xmax','ymax']]=bboxes[0]
            df.loc[idx,'class']=classes[0]
        else:
            df.loc[idx,['xmin','ymin','xmax','ymax']]=np.array([0,0,240,240])
            df.loc[idx,'class']=1
      

    for col in df.columns:
      if col!='file':
        df[col]=df[col].apply(lambda x:round(x))

    df.to_csv(self.output_csv,index=False)
    yolo_img = np.rot90(yolo_img,3,axes=(1,2))
    affine = [[-1,-0,-0,0],[-0,-1,-0,239],[0,0,1,0],[0,0,0,1]]
    yolo = nib.Nifti1Image(yolo_img[0], affine)
    nib.save(yolo, os.path.join(self.root_dir,self.result_dest))
    shutil.rmtree(os.path.join(self.root_dir,self.temp_dir))
    os.chdir('..')

  def return_dest(self):
    return os.path.join(self.root_dir,self.result_dest),self.output_csv
    