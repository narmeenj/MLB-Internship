import pandas as pd  
from sklearn.datasets import load_breast_cancer

def explore_data():
    print("----Loading Breast Cnacer Wisconsin Dataset----")
    raw_data=load_breast_cancer()
    
    df=pd.DataFrame(raw_data.data,columns=raw_data.features_names)
    df['target']=raw_data.target
    
    print("First 5 rows of the dataset:")
    print(df.head(5))
    
    print("Dataset Structural Information:")
    df.info()
    
    print("First 5 numerical features:")
    print(df.iloc[:,:5].describe())
    
    print("Target Class Distribution:")
    distribution=df['target'].value_counts()
    percentage=df['target'].value_counts(normalize=True)*100
    
    for label,name in [(0,'Malignant'),(1,'Benign')]:
        print("Class" label,"(",name,");",distribution[label],"samples(",round(percentage[label],2),"%)")
        
if __name__=="__main__":
    explore_data()        