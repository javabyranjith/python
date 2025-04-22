# EXPERIMENT - 3

# pip install scikit-learn
# NAIVE BAYES MODEL

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data and labels
data = [
    "This is one of the best product",
    "This is the best movie I have seen",
    "The food at this restaurant is amazing",
    "This product is terrible",
    "I wouldn't recommend this movie to anyone",
    "The service at this restaurant is awful"
]

# Corrected labels (matching 5 samples)
labels = [1, 1, 1, 0, 0, 0]  # 1 = Positive, 0 = Negative

# Define the pipeline correctly
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(data, labels)

# New data to predict
new_data = ["This is the best hotel.", 
            "My stay at this hotel was terrible."
            ]

# Make predictions
predictions = pipeline.predict(new_data)

# Display predictions
for text, pred in zip(new_data, predictions):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"'{text}' => {sentiment}")
