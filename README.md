# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the Iris dataset from Scikit-learn.
2. Split the dataset into training and testing sets, then standardize the input features.
3. Train the SGD Classifier using the training data and predict the Iris species for the test data.
4. Evaluate the model using accuracy and confusion matrix, and predict the species for a new flower sample.

## Program:

~~~py
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

# Load Iris dataset
iris = load_iris()

X = iris.data

y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Create and Train SGD Classifier
model = SGDClassifier(loss="log_loss", random_state=42, max_iter=1000)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Actual Labels:")

print(y_test)

print("\nPredicted Labels:")

print(y_pred)

print("\nAccuracy: {:.2f}%".format(accuracy * 100))

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))

# Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower = scaler.transform(new_flower)

prediction = model.predict(new_flower)

print("\nPredicted Iris Species:")

print(iris.target_names[prediction[0]])
~~~

## Output:

<img width="798" height="661" alt="image" src="https://github.com/user-attachments/assets/9f0c0e80-a602-4b2c-8e8b-ea94d224ca4e" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
