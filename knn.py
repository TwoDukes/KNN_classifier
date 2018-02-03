from scipy.spatial import distance
from collections import Counter
from numpy import average

def euc(a,b):
  return distance.euclidean(a,b)

class customKNN():

  def __init__(self, neighbors = 1):
        self.neighbors = neighbors

  def fit(self, features_train, labels_train):
    self.features_train = features_train
    self.labels_train = labels_train
  
  def predict(self, features_test):
    predictions = []
    for row in features_test:
      label = self.closest(row)
      predictions.append(label)
    return predictions
  
  #find the closest features in training set and returns
  # the average label between them over uniform distances
  def closest(self, row):
    best_indices = []
    for n in range(0, self.neighbors):
      best_distance = euc(row, self.features_train[0])
      best_current_index = 0
      for i in range(1, len(self.features_train)):
        dist = euc(row, self.features_train[i])
        if dist < best_distance and i not in best_indices:
          best_distance = dist
          best_current_index = i
      best_indices.append(best_current_index)
    
    avg = []
    for x in best_indices:
      avg.append(self.labels_train[x])
    most_common = Counter(avg).most_common(1)
    return most_common[0][0]


from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

#from sklearn.neighbors import KNeighborsClassifier
#^^SCIKIT-LEARN KNN^^

clf = customKNN(neighbors = 5)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)


#for prediction in pred:
#  if prediction == 0:
#    print "setosa"
#  elif prediction == 1:
#    print "versicolor"
#  else:
#    print "virginica"

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
print "Accuracy: ", acc * 100, '%'