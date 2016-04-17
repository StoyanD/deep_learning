from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
iris = load_iris()

# print iris.feature_names
# print iris.target_names
# print iris.data[0]
# print iris.target[0]

test_indx = [0,50,100]
#training datasets
training_target = np.delete(iris.target, test_indx)
training_data = np.delete(iris.data, test_indx, axis=0)

#testing datasets
test_target = iris.target[test_indx]
test_data = iris.data[test_indx]

clf = tree.DecisionTreeClassifier()
clf.fit(training_data, training_target)

print test_target
print clf.predict(test_data)

from sklearn.externals.six import StringIO
import pydotplus as pydot


dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
