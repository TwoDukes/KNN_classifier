import random

class customKNN():
  def fit(self, features_train, labels_train):
    self.features_train = features_train
    self.labels_train = labels_train
  
  def predict(self, features_test):
    predictions = []
    for row in features_test:
      label = random.choice(self.labels_train)
      predictions.append(label)
    return predictions


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