import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def clean_and_train():
    print("Loading data...")
    df = pd.read_csv("Dataset.csv")
    
    print(f"Original Data Shape: {df.shape}")
    
    drops = ["Unit1", "Unit2", "Patient_ID", "Hour", "Unnamed: 0", "ICULOS", "HospAdmTime"]
    df.drop(columns=[col for col in drops if col in df.columns], inplace=True)
    
    dratio_init = 0.01
    dataratio = df.count() / len(df)
    lowdata = dataratio[dataratio < dratio_init].index
    print(f"Dropping high-null columns: {list(lowdata)}")
    df.drop(columns=lowdata, inplace=True)
    
    max_row_nulls = 10
    rnull = df.isnull().sum(axis=1)
    df = df[rnull <= max_row_nulls].copy()
    
    print(f"Data Shape after dropping: {df.shape}")
    
    X = df.drop(columns=['SepsisLabel'])
    y = df['SepsisLabel']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    print("Saving model artifacts...")
    artifacts = {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'features': list(X.columns)
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    
    print("Training complete and model saved to model.pkl")

if __name__ == "__main__":
    clean_and_train()
