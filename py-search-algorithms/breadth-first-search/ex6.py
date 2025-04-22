# EXPERIMENT - 6
# Implementation Of Decision Trees
# pip install matplotlib pandas scikit-learn pydotplus six
# pip install ipython
# pip install nltk

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv('C:/Users/java2/Downloads/Social_Network_Ads.csv')
print(data.head())

# Feature columns and target
feature_cols = ['Age', 'EstimatedSalary']
x = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

# Split into train/test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Train Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Predict and evaluate
from sklearn import metrics
y_pred = classifier.predict(x_test)
print('Accuracy Score:', metrics.accuracy_score(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualization
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min()-1, stop=x_set[:, 0].max()+1, step=0.01),
                     np.arange(start=x_set[:, 1].min()-1, stop=x_set[:, 1].max()+1, step=0.01))

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title("Decision Tree (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

# Tree Visualization (Graphviz)
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus  

# Visualize the first tree
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('decisiontree.png')
Image(filename='decisiontree.png')

# Train and visualize an optimized tree
classifier = DecisionTreeClassifier(criterion="gini", max_depth=3)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("Optimized Tree Accuracy:", metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('opt_decisiontree_gini.png')
Image(filename='opt_decisiontree_gini.png')
