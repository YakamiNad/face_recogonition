from __future__ import print_function
import numpy as np
from numpy import *
import warnings
from skimage.io import imshow
import operator
from os import listdir
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
class NearestNeighbor(object):
  def __init__(self):
    pass
  def train(self, X, y):
    self.Xtr = X
    self.ytr = y
  def predict(self, X,k_value):
    num_test = X.shape[0]
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
    for i in range(num_test):
      distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
      sorted_distances=distances.argsort();
      classCount={}
      for h in range(k_value):
        response=self.ytr[sorted_distances[h]]
        classCount[response]=classCount.get(response,0)+1
      sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
      Ypred[i] = sortedClassCount[0][0]
    return Ypred
faces_image = np.load('training_image.npy')
faces_target = np.load('label_target_face.npy')
faces_image=faces_image.reshape(500,19*19)
mean_flo=(faces_image).mean(0)
mean=np.int_(mean_flo)
# imshow(mean.reshape(19,19))
dif=[]
A=[]
dif=np.subtract(faces_image,mean_flo)
# dif=dif.reshape(500,19,19)
# A=np.matmul(dif[0].reshape(19,19),dif[1].reshape(19,19))
# for x in range (2,499):
#     A=A*dif[x].reshape(19,19)
# C=np.cov(diff.transpose())
C = np.matrix(dif.transpose())*np.matrix(dif)
print(dif.shape)
evalues, evectors = np.linalg.eig(C)
# temp=np.matmul(dif,evectors
print(evalues.shape)
# print(evectors.shape)
evectors=evectors / (evectors.max()/255)
print(evectors.shape)
# for x in range(361):
#     for y in range(361):
#         if(evectors[x][y]<0):
#             np.evectors[x][y]=np.evectors[x][y]+255
temp=evectors[200].reshape(19,19)

# print(abs(evectors[0]*255))

plt.imshow(temp)
fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(evectors[i].reshape(19, 19), cmap='bone')
