import pandas as pd
import pickle

dataset = pd.read_csv(r"C:\Users\nikhi\Desktop\Linear_regression_heroku\Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

pickle.dump(regressor,open(r"C:\Users\nikhi\Desktop\Linear_regression_heroku\model.pkl",'wb'))