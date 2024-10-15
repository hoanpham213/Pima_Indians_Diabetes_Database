# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

def load_data():
    # Load your dataset here
    data = pd.read_csv('diabetes.csv')  # Path to your data
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    return X, y

def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train):
    # SVM
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Logistic Regression
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train, y_train)
    
    # Neural Network (MLPClassifier)
    mlp_model = MLPClassifier(random_state=42, max_iter=500)
    mlp_model.fit(X_train, y_train)
    
    # Stacking Classifier
    estimators = [('svm', svm_model), ('logistic', logistic_model)]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=mlp_model)
    stacking_model.fit(X_train, y_train)
    
    return svm_model, logistic_model, mlp_model, stacking_model

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    return results
