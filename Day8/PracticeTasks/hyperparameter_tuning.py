import numpy as np
import pandas as pd  
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def hyperparameter_tuning():
    print("----Hyperparameter Tuning----")
    
    raw_data=load_breast_cancer()
    X=pd.DataFrame(raw_data.data,columns=raw_data.features_names)
    y=raw_data.target
       
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
       
    scalar= StandardScalar()
    X_train_scaled=scalar.fit_transform(X_train) 
    
    parameter={'penalty':['11','12'],
               'C':[0.001,0.01,0.1,1,10,100],
               'solver':['liblinear','saga']}
    
    print("Searching across parameters:")
    print("Penalties:",parameter['penalty'])
    print("C Strengths:",parameter['C'])
    print("Solvers:",parameter['solver'])
    
    grid_search = GridSearchCV(estimator=LogisticRegression(max_iter=10000,random_state=42),
                               parameter=parameter,
                               cv=5,
                               scoring='f1',
                               n_jobs=1)
    
    print("Running grid variations to optimize F1-Score..")
    grid_search.fit(X_train_scaled,y_train)
    
    print("Best Hyperparameters:")
    print(grid_search.best_params_)
    print("Highest Validation Cross-Validation F1-Score achieved during tuning:")
    print(round(grid_search.best_score_,4))
    
if __name__=="__main__":
    hyperparameter_tuning()     
    
    
    