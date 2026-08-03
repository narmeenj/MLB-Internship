import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns   
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

def breast_cancer_predict():
    print("\n----Breast Cancer Prediction System----")
    raw_data=load_breast_cancer()
    X=pd.DataFrame(raw_data.data,columns=raw_data.feature_names)
    y=pd.Series(raw_data.target, name='target')
    
    print("\nTarget Class Distribution:")
    distribution=y.value_counts()
    percentage=y.value_counts(normalize=True)*100
    
    for label,name in [(0,'Malignant'),(1,'Benign')]:
        print("Class", label,"(",name,");",distribution[label],"samples(",round(percentage[label],2),"%)")
        
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)  
    
    scalar= StandardScaler()
    X_train_scaled=scalar.fit_transform(X_train)
    X_test_scaled=scalar.transform(X_test)
        
    baseline=LogisticRegression(max_iter=10000,random_state=42)
    baseline.fit(X_train_scaled,y_train)
    y_pred_base=baseline.predict(X_test_scaled)  
    
    b_accuracy=accuracy_score(y_test,y_pred_base)
    b_precision=precision_score(y_test,y_pred_base)
    b_recall=recall_score(y_test,y_pred_base)
    b_f1=f1_score(y_test,y_pred_base) 
    
    tune=LogisticRegression(C=0.1,penalty='l2',solver='liblinear',max_iter=10000,random_state=42)
    tune.fit(X_train_scaled,y_train)
    y_pred_tune=tune.predict(X_test_scaled)  
    
    t_accuracy=accuracy_score(y_test,y_pred_tune)
    t_precision=precision_score(y_test,y_pred_tune)
    t_recall=recall_score(y_test,y_pred_tune)
    t_f1=f1_score(y_test,y_pred_tune) 
    
    cm_base = confusion_matrix(y_test, y_pred_base)
    cm_tuned = confusion_matrix(y_test, y_pred_tune)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
    axes[0].set_title('Baseline Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
    axes[1].set_title('Tuned Model Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('Day8/confusion_matrix.png', dpi=300)
    print("\nMatrix heatmap graphic generated successfully as 'confusion_matrix.png'.")
    
    print("\n-----System Performance Comparison Report-----")
    print("-------------------------------------------------")
    print("Metric          | Baseline Model | Tuned Model")
    print("-------------------------------------------------")
    print("Accuracy        |", round(b_accuracy, 4), "       |", round(t_accuracy, 4))
    print("Precision       |", round(b_precision, 4), "       |", round(t_precision, 4))
    print("Recall(Critical)|", round(b_recall, 4), "       |", round(t_recall, 4))
    print("F1-Score        |", round(b_f1, 4), "       |", round(t_f1, 4))

    print()
if __name__ == "__main__":
    breast_cancer_predict()