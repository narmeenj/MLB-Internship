import numpy as np
import pandas as pd  
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def hyperparameter_tuning():
    print("\n----Hyperparameter Tuning----")
    
    raw_data=load_breast_cancer()
    X=pd.DataFrame(raw_data.data,columns=raw_data.feature_names)
    y=raw_data.target
       
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
       
    scalar= StandardScaler()
    X_train_scaled=scalar.fit_transform(X_train) 
    
    parameter={'penalty':['l1','l2'],
               'C':[0.001,0.01,0.1,1,10,100],
               'solver':['liblinear','saga']}
    
    print("\nSearching across parameters:")
    print("Penalties:\n",parameter['penalty'])
    print("C Strengths:\n",parameter['C'])
    print("Solvers:\n",parameter['solver'])
    
    grid_search = GridSearchCV(estimator=LogisticRegression(max_iter=10000,random_state=42),
                               param_grid=parameter,
                               cv=5,
                               scoring='f1',
                               n_jobs=-1)
    
    print("\nRunning grid variations to optimize F1-Score..")
    grid_search.fit(X_train_scaled,y_train)
    
    print("\nBest Hyperparameters:")
    print(grid_search.best_params_)
    print("Highest Validation Cross-Validation F1-Score achieved during tuning:")
    print(round(grid_search.best_score_,4))
    print()
if __name__=="__main__":
    hyperparameter_tuning()     
    
    
    