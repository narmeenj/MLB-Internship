import numpy as np   
import pandas as pd    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('Day6/practice_tasks/student_performance.csv')

if "Average_Score" not in data.columns:
    subject_col=['English','Mathematics','Statistics','Database','Python','Machine_Learning']
    data["Average_Score"]=data[subject_col].mean(axis=1)
    data.to_csv('Day6/practice_tasks/student_preprocessed_data.csv',index=False)
 
y=data["Average_Score"]

X_raw=data.drop(columns=['English','Mathematics','Statistics','Database','Python','Machine_Learning',"Average_Score"],errors="ignore")
    
category_col=X_raw.select_dtypes(include=["object"]).columns.tolist()
X=pd.get_dummies(X_raw,columns=category_col,drop_first=True)   

X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scale=scaler.fit_transform(X_train)
X_test_scale=scaler.transform(X_test)

np.save("Day6/practice_tasks/X_train_scale.npy",X_train_scale)
np.save("Day6/practice_tasks/X_test_scale.npy",X_test_scale)
np.save("Day6/practice_tasks/y_train.npy",y_train.to_numpy())
np.save("Day6/practice_tasks/y_test.npy",y_test.to_numpy())

print("Processed Metrics saved to disk!")