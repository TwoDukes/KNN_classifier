from scipy.spatial import distance

def euc(a,b):
  return distance.euclidean(a,b)

class customKNN():
  def fit(self, features_train, labels_train):
    self.features_train = features_train
    self.labels_train = labels_train
  
  def predict(self, features_test):
    predictions = []
    for row in features_test:
      label = self.closest(row)
      predictions.append(label)
    return predictions
  
  #find the closest feature in training set and returns its label
  def closest(self, row):
    best_distance= euc(row, self.features_train[0])
    best_index = 0
    for i in range(1, len(self.features_train)):
      dist = euc(row, self.features_train[i])
      if dist < best_distance:
        best_distance = dist
        best_index = i
    return self.labels_train[best_index]


from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

#from sklearn.neighbors import KNeighborsClassifier
#^^SCIKIT-LEARN KNN^^

clf = customKNN()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
print "Accuracy: ", acc * 100, '%'