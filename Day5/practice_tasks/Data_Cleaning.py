import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

data=pd.read_csv('Day5/practice_tasks/student_performance.csv')

subjects=['English','Mathematics','Statistics','Database','Python','Machine_Learning']
for i in subjects:
    if data[i].isnull().any():
        data[i]=data[i].fillna(data[i].mean())
        
data=data.drop_duplicates(subset=['Roll_No'])

data=data.rename(columns={'Machine_Learning':'Machine_Learning_Marks'})     
subjects=['English','Mathematics','Statistics','Database','Python','Machine_Learning_Marks']

data['Average_Score']=data[subjects].mean(axis=1)
def givePerformance(score):
    if score>=90:
        return 'Excellent'
    elif score>=80:
        return 'Good'
    elif score>=70:
        return 'Average'
    else:
        return "Needs Improvement."
    
data['Performance']=data['Average_Score'].apply(givePerformance)    

data.to_csv("Day5/practice_tasks/cleaned_student_performance.csv",index=False)          