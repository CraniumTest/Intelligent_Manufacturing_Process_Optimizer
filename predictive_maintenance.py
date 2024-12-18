import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load sensor data from a CSV file
data = pd.read_csv('sensor_data.csv')

# Preprocessing the data
# Assuming sensor values and labels columns are available
X = data.drop('failure', axis=1)  # Features
y = data['failure']               # Labels

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save model
joblib.dump(rf, 'predictive_maintenance_model.pkl')
