# Day 6: Student Score Prediction System

This project contains a decoupled Machine Learning workflow designed to predict a student's `Average_Score` based on performance metrics, isolating data preprocessing safely away from model training.

## 🛠️ Data Preprocessing Insights
* **Importance**: Raw datasets contain text labels and varied feature ranges that algorithmic models cannot ingest directly. Preprocessing converts variables into clean numerical matrices.
* **Categorical Data**: Handled using One-Hot Encoding via `pd.get_dummies(drop_first=True)` to convert string labels into binary vectors without creating artificial numeric hierarchies.
* **Feature Scaling**: Applied `StandardScaler` to ensure columns with large numerical values do not mathematically overpower smaller numeric scales.

## 🔀 Train-Test Splitting & Data Leakage Prevention
* **Train-Test Split**: Splitting 80% for training and 20% for testing ensures we evaluate the model on completely unseen data, mirroring real-world application.
* **Preventing Leakage**: The scaler is strictly `.fit_transform()` evaluated on the training set alone, and applied via `.transform()` to the test set. This isolates test distribution parameters completely from model knowledge.

## 📊 Evaluation Metrics Used
1. **Mean Absolute Error (MAE)**: Quantifies the average absolute deviation of predictions from actual grades.
2. **Mean Squared Error (MSE)**: Calculates the average squared error differences, punishing outlier errors significantly more.
3. **R² Score (Coefficient of Determination)**: Defines the proportion of variance in the student scores predictable from input features.

## 📈 Performance & Observations
* When the individual scores were accidentally left inside the feature matrix, the model suffered from **Data Leakage**, showing a deceptive performance of `MAE: 0.19` and `R²: 0.9980`.
* After correctly dropping individual test scores, the model yields realistic, organic performance metrics. 
