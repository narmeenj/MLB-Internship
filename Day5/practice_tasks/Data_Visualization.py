import pandas as pd 
import matplotlib.pyplot as plt       
import seaborn as sns 
import os   
  
  
data=pd.read_csv('Day5/practice_tasks/cleaned_student_performance.csv') 

sns.set_theme(style="whitegrid") 
update_subjects=['English','Mathematics','Statistics','Database','Python','Machine_Learning_Marks']

save_charts='Day5/practice_tasks/'
os.makedirs(save_charts,exist_ok=True)

#Bar Chart
plt.figure(figsize=(12,5))
sns.barplot(x='Student_Name',y='Average_Score',data=data.head(15),palette='viridis',hue='Student_Name',legend=False)
plt.title('Average Marks per Student')
plt.xticks(rotation=45,ha='right')
plt.ylabel('Average Score')
plt.ylim(0,100)
plt.tight_layout()
plt.savefig(os.path.join(save_charts,'Student Average Bar Chart.png'),dpi=300)
plt.show()
plt.close()

#Histogram
plt.figure(figsize=(8,4))
sns.histplot(data['Average_Score'],bins=10,kde=True,color='purple')
plt.title('Distribution of Student Average Scores')
plt.ylabel('Average Score Range')
plt.xlabel('Number of Students')
plt.tight_layout()
plt.savefig(os.path.join(save_charts,'Score Distribution Histogram.png'),dpi=300)
plt.show()
plt.close()

#Scatter Plot
plt.figure(figsize=(7,5))
sns.scatterplot(x='Python',y='Machine_Learning_Marks',data=data,hue='Performance', palette='Set1')
plt.title('Python Marks vs. Machine Learning Marks')
plt.ylabel('Python Marks')
plt.xlabel('Machine Learning Marks')
plt.tight_layout()
plt.savefig(os.path.join(save_charts,'Python vs ML Scatter Plot.png'),dpi=300)
plt.show()
plt.close()

#Pie Chart
plt.figure(figsize=(6,6))
performanceTotal=data['Performance'].value_counts()
plt.pie(performanceTotal, labels=performanceTotal.index,autopct='%1.1f%%', colors=sns.color_palette('pastel')[0:len(performanceTotal)],startangle=140)
plt.title('Distribution of Students by Performance')
plt.tight_layout()
plt.savefig(os.path.join(save_charts,'Performance Pie Chart.png'),dpi=300)
plt.show()
plt.close()

#Box Plot
plt.figure(figsize=(10,5))
sns.boxplot(data=data[update_subjects],palette='Set3')
plt.title('Spread of Marks Acroos All Subjects')
plt.ylabel('Marks Distribution')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(save_charts,'Subject Spread Box Plot.png'),dpi=300)
plt.show()
plt.close()
