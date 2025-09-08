import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

# Sample dataset
data={
    'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Temperature': [30, 32, 34, 33, 31, 29, 28, 27, 26, 25],
    'Humidity': [70, 65, 60, 62, 68, 75, 80, 85, 90, 95],
    'WindSpeed': [10, 12, 14, 11, 13, 15, 9, 8, 7, 6],
    'Rainfall': [5, 3, 0, 1, 4, 6, 8, 10, 12, 15],
    'NextDayTemperature': [32, 34, 33, 31, 29, 28, 27, 26, 25, 24]
}
df=pd.DataFrame(data)

# Features and target
x=df[['Temperature','Humidity','WindSpeed','Rainfall']]
y=df['NextDayTemperature']

# Split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Train model
model=LinearRegression()
model.fit(x_train,y_train)

# Predictions
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

# Plot results
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='o')
plt.title('Actual vs Predicted Next Day Temperature')
plt.xlabel('Test Sample Index')
plt.ylabel('Next Day Temperature')
plt.legend()
plt.show()

# ✅ FIX: Only include the same features used for training
new_data=pd.DataFrame({
    'Temperature':[29],
    'Humidity':[80],
    'WindSpeed':[12],
    'Rainfall':[4]
})

predicted_temperature=model.predict(new_data)
print(f"Predicted Next Day Temperature: {predicted_temperature[0]:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
print(f"Accuracy: {r2*100:.2f}%")
