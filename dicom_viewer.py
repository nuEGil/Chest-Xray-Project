import os
import cv2
import glob
import pydicom # python library for reading dicom images

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class dicom_imgview():
    def __init__(self, file_dir, annotations, detailed_annotations):
        # drawing font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # training files

        self.files = sorted(glob.glob(os.path.join(file_dir,'*.dcm')))
        self.nf = len(self.files) # count how many files we have
        print('number of files in directory:{}'.format(self.nf))

        # impor the data from the annotation files
        self.fdat = pd.read_csv(annotations) # regular annotation data
        self.det_fdat = pd.read_csv(detailed_annotations) # detailed annotation data
        print(self.fdat.info())
        print(self.det_fdat.info())
        # initialize counter
        self.f_idx = 0 # indexer
        self.skip = 1 # skip count

        # initialize image
        self.img = np.zeros((1024,1024))

    def read_img(self,fname):
        # image reader function
        ds = pydicom.dcmread(fname)
        img = ds.pixel_array

        #img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
        img = img[..., np.newaxis]
        self.img = np.concatenate([img,img,img],axis = -1)
        print('img name:{}'.format(fname))
        print('pixel array shape:{} \npixel array min:{} max:{} mean:{} std:{}'.format(self.img.shape,self.img.min(),self.img.max(),np.mean(self.img),np.std(self.img)))
        print('img type:{}'.format(type(self.img)))
    
    def draw_on_img(self):
        # File name matches the patientId so we can use that fact
        curr_id = self.files[self.f_idx].split('/')[-1].replace('.dcm','')
        print('current id:{}'.format(curr_id))
        # now we just need to search the pandas data frame to get the correct
        # bounding boxes, and the correct tags.
        fd_inds = self.fdat.index[self.fdat['patientId'] == curr_id].tolist()
        detfd_inds = self.det_fdat.index[self.det_fdat['patientId'] == curr_id].tolist()
        for fd_ind,detfd_ind in zip(fd_inds,detfd_inds):
            print(fd_ind,detfd_ind)
            bbox = self.fdat.iloc[fd_ind,1:-1]
            # once we get the indices, we need to get the  bounding boxes, and the
            # the target and the detailed label.
            cv2.putText(self.img,self.det_fdat.iloc[detfd_ind,1],(0,100),self.font,1,(0,255,0),1,cv2.LINE_AA)
            if self.fdat.iloc[fd_ind,-1] == 1:
                cv2.rectangle(self.img,(int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])),(0,255,0),3)

    def work(self):
        cv2.namedWindow('image')#,flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 512, 512)
        # record key presses
        k = cv2.waitKey(0) & 0xFF # in ascii vals
        # read initial image
        self.read_img(self.files[self.f_idx])

        # pretty much the game piece that lets us flip through the image deck
        while k!= ord('q'):
            k = cv2.waitKey(0) & 0xFF # in ascii vals
            self.f_idx = self.f_idx%self.nf

            self.draw_on_img()
            if k == ord('d'):
                self.f_idx+=self.skip
                self.read_img(self.files[self.f_idx])
                self.draw_on_img()
            elif k == ord('a'):
                self.f_idx-=self.skip
                self.read_img(self.files[self.f_idx])
                self.draw_on_img()
            elif k == ord('w'):
                self.skip+=1
            elif k == ord('s'):
                self.skip-=1
            cv2.imshow('image',self.img)

    def close_shop(self):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    dcmview  = dicom_imgview()
    dcmview.work()
    dcmview.close_shop()
