import os
import matplotlib.pyplot as plt
import numpy as np   
import pandas as pd    
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

X_train_scale=np.load("Day6/practice_tasks/X_train_scale.npy")
X_test_scale=np.load("Day6/practice_tasks/X_test_scale.npy")
y_train=np.load("Day6/practice_tasks/y_train.npy")
y_test=np.load("Day6/practice_tasks/y_test.npy")

print("Processed arrays loaded to memory!")

model=LinearRegression()
model.fit(X_train_scale,y_train)
print("Model training complete.")

y_predict=model.predict(X_test_scale)

mae=mean_absolute_error(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
r2=r2_score(y_test,y_predict)

print("Mean Absolute Error (MAE):{:.2F}".format(mae))
print("Mean Squared Error (MSE):{:.2F}".format(mse))
print("R2 Score (Accuracy):{:.4F}".format(r2))

compare_data=pd.DataFrame({"Actual":y_test,"Predicted":y_predict})
print("\n______Actual Values vs Predicted Values______")
print(compare_data.head(10).round(2))


#Scatter Plot
print("Generating prediction Visualization Plot...")
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_predict,color="blue",alpha=0.6, label="Predicted vs Actual")

perfect_line=np.linspace(min(y_test),max(y_test),100)
plt.plot(perfect_line,perfect_line,color="red",linestyle="--",linewidth=2,label="Perfect Fit (Ideal)")

plt.title("Student Score Prediction System: Actual vs Predicted Values")
plt.xlabel("Actual Average Scores")
plt.ylabel("Predicted Average Scores")
plt.legend(loc="upper left")
plt.grid(True,linestyle=":",alpha=0.6)

output_pic="Day6/Student_Performance_Plot.png"
plt.savefig(output_pic,dpi=300)
plt.show()
plt.close()

print("Visualization Plot saved as: {}".format(output_pic))