import cv2
import os
import glob
import tqdm
dir2 = './masks/'
dir3 = './mask1'
files2 = glob.glob(dir2+'*.*')
for file1 in tqdm.tqdm(files2):
        print(file1)
        img1 = cv2.imread(file1)
        _,name = os.path.split(file1)
        #if img1.shape[0] !=256 or img1.shape[1]!=256:
        img2 = cv2.resize(img1,(256,256))
        cv2.imwrite(os.path.join(dir3,name),img2)
