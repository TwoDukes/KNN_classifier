

from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

#from sklearn.neighbors import KNeighborsClassifier
#^^SCIKIT-LEARN KNN^^

from knn import customKNN
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