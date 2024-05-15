import numpy as np
import pandas as pd
from math import sqrt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



df = pd.read_csv("ConsumoCo2.csv")

print(df.head())
print(df.info())

X = df[["ENGINESIZE"]]
Y = df["CO2EMISSIONS"]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

predCO2 = model.predict(X_test)
mse = mean_squared_error(y_test, predCO2)
accuracy = model.score(X_test,y_test)
msa = mean_absolute_error(y_test,predCO2)

print(f'\nAccuracy: {accuracy}')
print(f'Mean square error: {mse}')

print("\nValue of A", model.intercept_)
print("Value of B", model.coef_)

print(f"\nSum of squares{np.sum((predCO2-y_test)**2)}")
print(f"Mean absolute error:{msa}")

print(f"\nMean square error square root:{sqrt(mean_squared_error(y_test,predCO2))} ")
print(f"R2 Score:{r2_score(predCO2,y_test)} ")

results = pd.DataFrame({"ENGINESIZE": X_test["ENGINESIZE"], "CO2EMISSIONS actual": y_test, "CO2EMISSIONS prediction": predCO2})

print(results)

plt.scatter(X_test, y_test, color='blue', label='actual data')
plt.plot(X_test, predCO2, color='red', linewidth=2, label='prediction')
plt.xlabel('Engine size')
plt.ylabel('CO2 emission')
plt.legend()
plt.show()

