# EXPERIMENT - 7
# Implementation Of SVM 
# pip install pandas matplotlib seaborn scikit-learn nltk


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score, recall_score, precision_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import nltk
nltk.download('punkt')

# Step 1: Load the data
data = pd.read_csv(
    'https://raw.githubusercontent.com/Shreyakkk/Email-Spam-Detector/master/spam.csv')

# Step 2: Word count distribution
ham_lengths = [len(word_tokenize(text))
               for text in data[data['Label'] == 'ham']['EmailText']]
spam_lengths = [len(word_tokenize(text))
                for text in data[data['Label'] == 'spam']['EmailText']]

# Note: sns.distplot is deprecated — using sns.histplot instead
sns.histplot(ham_lengths, bins=30, color='blue', stat='density', kde=True, label='Ham')
sns.histplot(spam_lengths, bins=30, color='red', stat='density', kde=True, label='Spam')
plt.title('Distribution of Number of Words')
plt.xlabel('Number of Words')
plt.legend()
plt.show()

# Step 3: Prepare the data
X = data['EmailText'].values
y = data['Label'].values

# Convert labels to integers (ham: 0, spam: 1)
y = [0 if label == 'ham' else 1 for label in y]

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Step 5: Convert text to numerical features
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# Step 6: Train the SVM model
model = SVC(kernel='linear', gamma=1, random_state=10, C=2)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
print('Score =', model.score(X_test, y_test))
y_predict_test = model.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_predict_test))
print("Accuracy Score:", accuracy_score(y_test, y_predict_test))
print("F1 Score:", f1_score(y_test, y_predict_test))
print("Recall:", recall_score(y_test, y_predict_test))
print("Precision:", precision_score(y_test, y_predict_test))

# Step 8: Confusion matrix
cm = confusion_matrix(y_test, y_predict_test)
print("\nConfusion Matrix:\n", cm)

# Step 9: Plot the heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

