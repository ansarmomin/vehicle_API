import pandas as pd


df = pd.read_csv(r"C:\Users\Ridaa\OneDrive\Desktop\ansar\vehicle_data.csv")  


print(df.head())
print(df.info())
print(df.describe())
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\Ridaa\OneDrive\Desktop\ansar\vehicle_data.csv")


df = df.dropna()


features = ['year', 'mileage', 'cylinders', 'fuel', 'transmission', 'body', 'drivetrain']
target = 'price'

df = df[features + [target]]


df = pd.get_dummies(df, columns=['fuel', 'transmission', 'body', 'drivetrain'])


X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)