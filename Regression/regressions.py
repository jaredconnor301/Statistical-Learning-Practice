import pandas as pd
import quandl, math, datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
style.use("ggplot")

# Gathering dataframe
df = quandl.get("WIKI/GOOGL")

# Cut down dataset to only display information that is needed
df = df[['Adj. Open','Adj. Close','Adj. Volume','Adj. High', 'Adj. Low']]

# Creating High / Low Percentage
df['High_Low_pct_change'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.00

# Creating percent change dataframe
df['pct_change'] = (df['Adj. Open'] - df['Adj. Close']) / df['Adj. Open'] * 100.00

# Create a new dataframe that combines close, high/low %chg, % change, and volume
df = df[['Adj. Close', 'High_Low_pct_change', 'pct_change', 'Adj. Volume']]

# Define the forecast column for use in the future
forecast_col = 'Adj. Close'
df.fillna(value = -99999, inplace = True)

# Define the forecast range and period in question
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])

# Divide the model into train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# define the classifier
clf = LinearRegression()

# Fit the model
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_recent)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
