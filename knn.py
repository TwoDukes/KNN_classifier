from scipy.spatial import distance
from collections import Counter
from numpy import average

def euc(a,b):
  return distance.euclidean(a,b)

#TODO: move this to another file and import it
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
