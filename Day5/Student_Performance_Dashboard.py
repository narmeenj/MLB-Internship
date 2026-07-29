import os
import pandas as pd
import matplotlib.pyplot as plt             
import seaborn as sns   

file_path='Day5/cleaned_student_performance.csv'
if not os.path.exists(file_path):
    file_path='cleaned_student_performance.csv'
    
data=pd.read_csv(file_path)   

data=data.rename(columns={'Machine_Learning_Marks':'Machine_Learning'})     

print("________________________________")
print("Student Performance Dashboard")
print("________________________________")
print()


total_students=len(data)
print("Toatl Students: ",total_students)
print()

print("Average Score Per Subject:")
subjects=['English','Mathematics','Statistics','Database','Python','Machine_Learning']
sub_average=data[subjects].mean()
for sub,avg_score in sub_average.items():
    clean_name=sub.replace('_Marks','')
    print("  -",clean_name,":","{:.2f}%".format(avg_score))
print()
    
print("Top 5 Performing Students:")  
top5=data.sort_values(by='Average_Score',ascending=False).head(5)
for index,row in top5.iterrows():
    print(row['Student_Name'],"-ID:",row['Roll_No'],"(Avg:","{:.2f}%)".format(row['Average_Score']))
print()
    
print("Students Requiring Improvement(Below 70% Average):") 
need_imp=data[data['Average_Score']<70]   
if need_imp.empty:
    print("No Student Below Average!")
else:
    for index,row in need_imp.iterrows():
        print(row['Student_Name'],"-ID:",row['Roll_No'],"(Avg:","{:.2f}%)".format(row['Average_Score']))  
print()
        
print("Subject with Highest Class Average:")   
highest_sub=sub_average.idxmax()
highest_avg=sub_average.max()
highest_name=highest_sub.replace('_Marks','')   
print(highest_name,"{:.2f}%".format(highest_avg))    
print()
      
print("________________________________")


sns.set_theme(style="whitegrid") 

save_charts='Day5'
os.makedirs(save_charts,exist_ok=True)

#Bar Chart
plt.figure(figsize=(12,5))
sns.barplot(x='Student_Name',y='Average_Score',data=data.head(15),palette='viridis',hue='Student_Name',legend=False)
plt.title('Average Marks per Student',fontsize=13,fontweight='bold')
plt.xticks(rotation=45,ha='right')
plt.ylabel('Average Score (%)')
plt.ylim(0,100)
plt.tight_layout()
plt.savefig(os.path.join(save_charts,'Student Average Bar Chart.png'),dpi=300)
plt.show()
plt.close()

#Histogram
plt.figure(figsize=(8,4))
sns.histplot(data['Average_Score'],bins=10,kde=True,color='purple')
plt.title('Distribution of Student Average Scores',fontsize=13,fontweight='bold')
plt.ylabel('Number of Students')
plt.xlabel('Average Score Range')
plt.tight_layout()
plt.savefig(os.path.join(save_charts,'Score Distribution Histogram.png'),dpi=300)
plt.show()
plt.close()

#Scatter Plot
plt.figure(figsize=(7,5))
sns.scatterplot(x='Python',y='Machine_Learning',data=data,hue='Performance', palette='Set1',s=80)
plt.title('Python Marks vs. Machine Learning Marks',fontsize=12,fontweight='bold')
plt.xlabel('Python Marks')
plt.ylabel('Machine Learning Marks')
plt.tight_layout()
plt.savefig(os.path.join(save_charts,'Python vs ML Scatter Plot.png'),dpi=300)
plt.show()
plt.close()

#Pie Chart
plt.figure(figsize=(6,6))
performanceTotal=data['Performance'].value_counts()
plt.pie(performanceTotal, labels=performanceTotal.index,autopct='%1.1f%%', colors=sns.color_palette('pastel')[0:len(performanceTotal)],startangle=140)
plt.title('Distribution of Students by Performance',fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save_charts,'Performance Pie Chart.png'),dpi=300)
plt.show()
plt.close()

#Box Plot
plt.figure(figsize=(10,5))
sns.boxplot(data=data[subjects],palette='Set3')
plt.title('Spread of Marks Across All Subjects',fontsize=12,fontweight='bold')
plt.ylabel('Marks Distribution')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(save_charts,'Subject Spread Box Plot.png'),dpi=300)
plt.show()
plt.close()    
