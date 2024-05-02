# %%
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv('boston_housing.csv')

# %%
df

# %%
df.head()

# %%
df.tail()

# %%
df.describe()

# %%
df.dtypes

# %%
df.shape

# %%
df.isnull().sum()

# %%
df.corr()

# %%
sns.heatmap(df.corr())

# %%
df['CRIM'] = df['CRIM'].fillna(df['CRIM'].mean(skipna=True))

# %%
df['ZN'] = df['ZN'].fillna(df['ZN'].mean(skipna=True))

# %%
df['INDUS'] = df['INDUS'].fillna(df['INDUS'].mean(skipna=True))

# %%
df['CHAS'] = df['CHAS'].fillna(df['CHAS'].mean(skipna=True))


# %%
df['AGE'] = df['AGE'].fillna(df['AGE'].mean(skipna=True))
df['LSTAT'] = df['LSTAT'].fillna(df['LSTAT'].mean(skipna=True))


# %%
df.isnull().sum()

# %%
# def min_max_normalize(name):
#     global df
#     df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min() )

# columns = list(df.columns)
# columns
# for i in range(len(columns)):
#     min_max_normalize(columns[i])
# from sklearn.preprocessing import StandardScalar


def z_normalize(name):
    global df
    df[name] = abs((df[name] - df[name].mean()) / (df[name].std()))


columns = list(df.columns)
columns
for i in range(len(columns)):
    z_normalize(columns[i])


# %%
df

# %%
lin_model = LinearRegression()

X = df[['LSTAT', 'RM']]
y = df['MEDV']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lin_model.fit(X_train, y_train)

# %%
y_pred = lin_model.predict(X_test)

# %%
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mse)
print(r2)
