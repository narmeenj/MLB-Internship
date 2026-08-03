import pandas as pd  
from sklearn.datasets import load_breast_cancer

def explore_data():
    print("----Loading Breast Cnacer Wisconsin Dataset----")
    raw_data=load_breast_cancer()
    
    df=pd.DataFrame(raw_data.data,columns=raw_data.feature_names)
    df['target']=raw_data.target
    
    print("\nFirst 5 rows of the dataset:")
    print(df.head(5))
    
    print("\nDataset Structural Information:")
    df.info()
    
    print("\nFirst 5 numerical features:")
    print(df.iloc[:,:5].describe())
    
    print("\nTarget Class Distribution:")
    distribution=df['target'].value_counts()
    percentage=df['target'].value_counts(normalize=True)*100
    
    for label,name in [(0,'Malignant'),(1,'Benign')]:
        print("Class", label,"(",name,");",distribution[label],"samples(",round(percentage[label],2),"%)")
    print()    
if __name__=="__main__":
    explore_data()        