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
#faces_target = np.load('label_target_face.npy')
#note that when you are testing age prediction,you should comment the previous line and de-comment the next line
faces_target = np.load('label_target_age.npy')
#print(faces_image.shape)
faces_data = faces_image.reshape(faces_image.shape[0], faces_image.shape[1]*faces_image.shape[2])

n_samples = faces_image.shape[0]
X = faces_data
n_features = faces_data.shape[1]
y = faces_target
n_classes = faces_target.shape[0]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,random_state=0)
n_components_train = 100
n_components_test = 125
pca_train = PCA(n_components=n_components_train, svd_solver='randomized',
          whiten=True).fit(Xtrain)
eigenfaces_train = pca_train.components_.reshape((100, 19*19))

pca_test = PCA(n_components=n_components_test, svd_solver='randomized',
          whiten=True).fit(Xtest)
eigenfaces_test= pca_test.components_.reshape((125, 19*19))
Xtest_pca = PCA(n_components=n_components_test, svd_solver='randomized',
          whiten=True).fit(Xtest)

# print(ytrain.shape)
# print("aaaaaaaaaaa",Xtest_pca)
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
    nn= NearestNeighbor()
    nn.train(eigenfaces_train,ytrain)
    # print('sososososos')
    # print(eigenfaces_train.shape)
    # print(ytrain.shape)
    # print(eigenfaces_test.shape)
    Yte_predict=nn.predict(eigenfaces_test,k)
    acc=np.mean(Yte_predict == ytest)
    print(Yte_predict)
    print ('accuracy: %f' % ( np.mean(Yte_predict == ytest) ))
    validation_accuracies.append((k, acc))
print(validation_accuracies)
#fig, axes = plt.subplots(3, 8, figsize=(9, 4),
#                         subplot_kw={'xticks':[], 'yticks':[]},
#                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
#for i, ax in enumerate(axes.flat):
#    temp_ma=pca_train.components_[i].reshape(19,19,3)
#    # temp_ma=temp_ma*255
#    # temp_ma=np.ceil(temp_ma)
#    temp_ma=np.int_(temp_ma)
#    ax.imshow(temp_ma)
