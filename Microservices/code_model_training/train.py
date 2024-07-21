# Import packages
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import gzip
import matplotlib


# Load the dataset
filepath = 'data/breast_cancer.csv'
data = pd.read_csv(filepath)

# Preprocess dataset
data = data.set_index('id')
del data['Unnamed: 32']
data['diagnosis'] = data['diagnosis'].replace(['B', 'M'], [0, 1])  # Encode y, B -> 0 , M -> 1

# Split into train and test set, 80%-20%
y = data.pop('diagnosis')
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an ensemble of 3 models
estimators = []
estimators.append(('logistic', LogisticRegression()))
estimators.append(('cart', DecisionTreeClassifier()))
estimators.append(('svm', SVC()))

# Create the Ensemble Model
ensemble = VotingClassifier(estimators)

# Make preprocess Pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer()),  # Missing value Imputer
    ('scaler', MinMaxScaler(feature_range=(0, 1))),  # Min Max Scaler
    ('model', ensemble)  # Ensemble Model
])

# Train the model
pipe.fit(X_train, y_train)

# Test Accuracy
Accuracy = pipe.score(X_test, y_test)
print("Accuracy: %s" % str(Accuracy))

# Plot confusion matrix
plt.switch_backend('TkAgg')
print(ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test))
plt.show()

# Export model
modelPath = 'model/model_binary.dat.gz'
joblib.dump(pipe, gzip.open(modelPath, "wb"))
