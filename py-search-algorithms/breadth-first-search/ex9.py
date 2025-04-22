# EXPERIMENT - 9
#Implementation of Clustering algorithm


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target

# Split the dataset into Training and Testing sets
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_labels, test_size=0.20, random_state=42)

# Using K-Neighbors Classifier
classifier = KNeighborsClassifier(n_neighbors=6)

# Train the model
classifier.fit(x_train, y_train)

# Predict the labels for the test set
y_pred = classifier.predict(x_test)

# Print the score (accuracy)
print("Score: ", classifier.score(x_test, y_test))

# Print the classification report
print("Classification Report: \n", classification_report(y_test, y_pred))

# Print the confusion matrix
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
