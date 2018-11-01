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
