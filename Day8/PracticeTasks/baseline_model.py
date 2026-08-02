import numpy as np
import pandas as pd  
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

def train_baseline():
    print("----Training Baseline Model----")
    raw_data=load_breast_cancer()
    X=pd.DataFrame(raw_data.data,columns=raw_data.features_names)
    y=raw_data.target
    
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    
    scalar= StandardScalar()
    X_train_scaled=scalar.fit_transform(X_train)
    X_test_scaled=scalar.transform(X_test)
    
    baseline=LogisticRegression(max_iter=10000,random_state=42)
    baseline.fit(X_train_scaled,y_train)
    
    y_pred=baseline.predict(X_test_scaled)
    
    metrices={
        "Accuracy":accuracy_score(y_test,y_pred),
        "Precision":precision_score(y_test,y_pred),
        "Recall":recall_score(y_test,y_pred),
        "f1":f1_score(y_test,y_pred)
    }
    
    print("----Baseline Metrices----")
    for k,v in metrices.items():
        print(k,":",round(v,4))
        
    print("Baseline Confusion Metrices:")
    print(confusion_matrix(y_test,y_pred))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, metrices

if __name__=="__main__":
    train_baseline() 
        
        