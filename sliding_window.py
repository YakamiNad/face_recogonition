#the reference is the link as below :https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
from helper import pyramid
from helper import sliding_window
from knn_latest import NearestNeighbor
import numpy as np
import cv2
faces_image = np.load('olivetti_faces.npy')
#faces_target = np.load('label_target_face.npy')
#note that when you are testing age prediction,you should comment the previous line and de-comment the next line
faces_target = np.load('olivetti_faces_target.npy')
import time

#faces_image = np.load('training_image.npy')  #faces_image
#print(faces_image.shape)
#faces_target = np.load('label_target_face.npy')
# load the image and define the window width and height
image = cv2.imread('physics.png')
(winW, winH) = (64, 64)
result={}
for resized in pyramid(image, scale=1.5):
	# loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=19, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        #knn classifiers
        nn=NearestNeighbor()
        faces_image=faces_image.reshape(400,64*64)
        print(faces_target.shape)
        nn.train(faces_image,faces_target)
        window= cv2.cvtColor(window,cv2.COLOR_BGR2GRAY)
        window=window.reshape(1,64*64)
        # print(1)
        result=nn.predict(window,1)
        # print(2)
        print(result)
        if(result==5):
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.25)

		
