# imports
import os
from pyexpat import model
import shutil
import tempfile
import time
import numpy as np
import nibabel as nib
import sys
import torch

class Normalize():
    '''
    Normalize image
    '''
    def __init__(self,filename):
        self.data_root = ''
        self.filename=filename
        self.result_dest_frame = os.path.join(self.data_root,filename,'results',"normalized_by_frame.nii.gz")
        self.result_dest_volume = os.path.join(self.data_root,filename,'results',"normalized_by_volume.nii.gz")
        if torch.cuda.is_available():  
            self.device = "cuda:0" 
        else:  
            self.device = "cpu" 

    def get_img(self):
        '''
        Returns concatenation of all four modalities
        '''
        img = np.expand_dims(nib.load(os.path.join(self.data_root,self.filename,self.filename+'_flair.nii.gz')).get_fdata(),axis=0)
        # img = np.rot90(img,2,axes=(1,2))
        return img
    
    def normalize_by_frame(self,img):
        '''
        Max Normalize the concatenated modalities
        '''
        for i in range(img.shape[0]):
            arr = img[i,:,:,:]
            arr = arr[arr!=0]
            img[i,:,:,:][img[i,:,:,:]!=0] = (img[i,:,:,:][img[i,:,:,:]!=0]-np.mean(arr))/np.std(arr)
        return img

    def normalize_by_volume(self,img):
        '''
        Max Normalize the concatenated modalities
        '''
        img[img!=0] = (img[img!=0]-np.mean(img[img!=0]))/np.std(img[img!=0])
        return img

    def get_test(self):
        '''
        Returns Flair image
        '''
        img = np.expand_dims(nib.load(os.path.join(self.data_root,self.filename)).get_fdata(),axis=0)
        return img
    
    def save_nifti(self,normalized,result_dest):
        '''
        Saves the predicted file as a .nii file
        '''
        affine = [[-1,-0,-0,0],[-0,-1,-0,239],[0,0,1,0],[0,0,0,1]]
        norm = nib.Nifti1Image(normalized, affine)
        nib.save(norm, result_dest)
    
    def normalize_frame(self):
        img = self.get_img()
        normalized_img = self.normalize_by_frame(img)
        self.save_nifti(normalized_img[0],self.result_dest_frame)

    
    def normalize_volume(self):
        img = self.get_img()
        normalized_img = self.normalize_by_volume(img)
        self.save_nifti(normalized_img[0],self.result_dest_volume)

    

# if __name__=="__main__":
#     filename = sys.argv[1]
#     normalize = Normalize(filename)
#     normalize.normalize_and_savefile()




    