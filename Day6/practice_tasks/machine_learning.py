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
print(compare_data.head(5).round(2))