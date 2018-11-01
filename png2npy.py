import imageio
import glob
import numpy as np
from PIL import Image
import os
import glob
import pickle
import cv2
couter=1
images=[]
for infile in glob.glob("train/mixed/*.png"):
    img = cv2.imread(infile)
    cv2.imwrite(infile[:-3] + 'jpg', img)

filelist = glob.glob('train/mixed/*.jpg')
x = np.array([np.array(Image.open(fname)) for fname in filelist])
# pickle.dump( x, filehandle, protocol=2 )
for h in x:
    h= cv2.cvtColor(h, cv2.COLOR_BGR2GRAY)
    images.append(h)
x = np.array([np.array(i) for i in images])
x.dump('last_one.npy')
x = np.load('last_one.npy')
print(x.shape)
