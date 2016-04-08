import numpy as np
from sklearn import tree
features = np.array([[140,1], [130,1], [150,0], [170,0]])
labels = np.array([[0],[0],[1],[1]])
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)
print clf.predict(np.array([[150,0 ]]))
