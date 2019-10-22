import os, torch, pdb
import numpy as np
import json
from PIL import Image
from PIL import ImageFile
import torch.utils.data as data
import random 
import collections
from numpy import random as nprandom
import pickle
import glob
import re
import numpy as np
import pandas as pd
from random import shuffle
import random
import math
import nibabel as nib
from .augmentations import *
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ADNI_3D(data.Dataset):


    def __init__(self, dir_to_scans, dir_to_tsv, mode = 'Train', n_label = 3, percentage_usage = 1.0):
        if n_label == 3:
            LABEL_MAPPING = ["CN", "MCI", "AD"]
        elif n_label == 2:
            LABEL_MAPPING = ["CN", "AD"]
        self.LABEL_MAPPING = LABEL_MAPPING     
        if mode == 'Train':
            subject_tsv = pd.io.parsers.read_csv(os.path.join(dir_to_tsv, 
                mode+'_diagnosis_ADNI.tsv'), sep='\t')
        else:
            subject_tsv = pd.io.parsers.read_csv(os.path.join(dir_to_tsv,
                mode+'_diagnosis_ADNI.tsv'), sep='\t') 

        # Clean sessions without labels
        indices_not_missing = []
        for i in range(len(subject_tsv)):
            if mode == 'Train':
                if (subject_tsv.iloc[i].diagnosis in LABEL_MAPPING):
                    indices_not_missing.append(i)
            else:
                if (subject_tsv.iloc[i].diagnosis in LABEL_MAPPING):
                    indices_not_missing.append(i)

        self.subject_tsv = subject_tsv.iloc[indices_not_missing]
        if mode == 'Train':
            self.subject_tsv = subject_tsv.iloc[np.random.permutation(int(len(subject_tsv)*percentage_usage))]
        self.subject_id = np.unique(subject_tsv.participant_id.values)
        self.index_dic = dict(zip(self.subject_id,range(len(self.subject_id))))
        self.dir_to_scans = dir_to_scans


        self.mode = mode
        self.age_range = list(np.arange(0.0,120.0,0.5))



    def __len__(self):
        return len(self.subject_tsv)

    def __getitem__(self, idx):
        try:
            path = os.path.join(self.dir_to_scans,self.subject_tsv.iloc[idx].participant_id,
                self.subject_tsv.iloc[idx].session_id,'t1/spm/segmentation/normalized_space')
            all_segs = list(os.listdir(path))
            if self.subject_tsv.iloc[idx].diagnosis == 'CN':
                label = 0
            elif self.subject_tsv.iloc[idx].diagnosis == 'MCI':
                label = 1
            elif self.subject_tsv.iloc[idx].diagnosis == 'AD':
                if self.LABEL_MAPPING == ["CN", "AD"]:
                    label = 1
                else:
                    label = 2
            else:
                print('WRONG LABEL VALUE!!!')
                label = -100
            mmse = self.subject_tsv.iloc[idx].mmse
            cdr_sub = 0#self.subject_tsv.iloc[idx].cdr #cdr_sb #cdr#
            age = list(np.arange(0.0,120.0,0.5)).index(self.subject_tsv.iloc[idx].age_rounded) #list(np.arange(0.0,25.0)).index(self.subject_tsv.iloc[idx].education_level)#

            idx_out = self.index_dic[self.subject_tsv.iloc[idx].participant_id]

            

            for seg_name in all_segs:
                if 'Space_T1w' in seg_name:
                    image = nib.load(os.path.join(path,seg_name)).get_data().squeeze()
                    image[np.isnan(image)] = 0.0
                    image = (image - image.min())/(image.max() - image.min() + 1e-6)
        
                    if self.mode == 'Train':
                        image = self.augment_image(image)

            image = np.expand_dims(image,axis =0)

            if self.mode == 'Train':
                image = self.randomCrop(image,96,96,96)
            else:
                image = self.centerCrop(image,96,96,96)

        except Exception as e:
            print(f"Failed to load #{idx}: {path}")
            print(f"Errors encountered: {e}")
            print(traceback.format_exc())
            return None,None,None,None
        return image.astype(np.float32),label,idx_out,mmse,cdr_sub,age

    def centerCrop(self, img, length, width, height):
        assert img.shape[1] >= length
        assert img.shape[2] >= width
        assert img.shape[3] >= height

        x = img.shape[1]//2 - length//2
        y = img.shape[2]//2 - width//2
        z = img.shape[3]//2 - height//2
        img = img[:,x:x+length, y:y+width, z:z+height]
        return img

    def randomCrop(self, img, length, width, height):
        assert img.shape[1] >= length
        assert img.shape[2] >= width
        assert img.shape[3] >= height
        x = random.randint(0, img.shape[1] - length)
        y = random.randint(0, img.shape[2] - width)
        z = random.randint(0, img.shape[3] - height )
        img = img[:,x:x+length, y:y+width, z:z+height]
        return img
    def augment_image(self, image):
        sigma = np.random.uniform(0.0,1.0,1)[0]
        image = scipy.ndimage.filters.gaussian_filter(image, sigma, truncate=8)
        return image

    def unpickling(self, path):
       file_return=pickle.load(open(path,'rb'))
       return file_return