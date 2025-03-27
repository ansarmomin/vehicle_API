import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


error = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {error}")


import joblib
joblib.dump(model, "vehicle_price_model.pkl")
