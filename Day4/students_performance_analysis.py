import numpy as np 
import pandas as pd 

student_data=pd.read_csv("Day4/student_record.csv")

print("\nBasic Information:")
student_data.info()

print("\nAverage for each subject:")
subjects=["English","Mathematics","Physics","Chemistry","Biology","Computer"]

for i in subjects:
    if i in student_data.columns:
        print(i,"Average:",student_data[i].mean())
print()        

print("Top 5 studnets:")
top5=student_data.sort_values(by="Percentage",ascending=False) 
print(top5.head(5))


classAvg_percentage=student_data["Percentage"].mean()
print("\nBelow average students:") 
belowAvg=student_data[student_data["Percentage"]<classAvg_percentage]   
print(belowAvg.head(5))   

print("\nTotal Number of Students:",len(student_data))

student_data.to_csv("Day4/processed_student_record.csv",index=False)