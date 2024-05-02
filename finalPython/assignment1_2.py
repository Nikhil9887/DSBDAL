# %%
from sklearn.preprocessing import StandardScalar
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
df = pd.read_csv("./StudentsPerformance.csv")
df.head()

# %%
df.dtypes

# %%
df.info()

# %%
df.describe()
# math score is an object so we need to convert its type to float64

# %%
df.isnull().sum()

# %%
df.shape

# %%
df.isnull()

# %%
df.head()

# %%
cols = ['gender', 'group', 'parental_level_of_education', 'lunch',
        'test_preparation_course', 'math_score', 'reading_score', 'writing_score']
df.columns = cols

# %%
df.head()

# %%
df['math_score'].unique()

# %%
df['math_score'] = df['math_score'].replace('?', float('nan'))

# %%
df['math_score'].unique()

# %%
df['math_score'] = df['math_score'].astype('float64')

# %%
df.dtypes

# %%
df.describe()

# %%
sns.boxplot(data=df, x='reading_score')

# %%
sns.boxplot(data=df, x='writing_score')

# %%
df.isnull().sum()

# %%
mean_math_score = df['math_score'].dropna().mean()
print(mean_math_score)
df['math_score'] = df['math_score'].fillna(mean_math_score)

# %%
df.isnull().sum()

# %%
mean_reading_score = df['reading_score'].dropna().mean()
print(mean_reading_score)
df['reading_score'] = df['reading_score'].fillna(mean_reading_score)

# %%
mean_writing_score = df['writing_score'].dropna().mean()
print(mean_writing_score)
df['writing_score'] = df['writing_score'].fillna(mean_writing_score)

# %%
df.isnull().sum()

# %%
sns.boxplot(data=df, x='math_score')

# %%
sns.boxplot(data=df, x='reading_score')

# %%
sns.boxplot(data=df, x='writing_score')

# %%
# data normalization


def min_max_normalize(name):
    global df
    df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min())


min_max_normalize('math_score')
min_max_normalize('reading_score')
min_max_normalize('writing_score')

# %%
df.head(10)

# %%
df

# %%
gender_map = {'male': 1, 'female': 0}
df['gender'] = df['gender'].map(gender_map)
df

# %%
df.dtypes

# %%
lunch_map = {"standard": 1, "free/reduced": 0}
df['lunch'] = df['lunch'].map(lunch_map)
df

# %%
df

# %%
df.dtypes

# %%


def encode_categorical(feature):
    label = 0
    values = df[feature].unique()
    for val in values:
        df.loc[df[feature] == val, feature] = label
        label += 1
    df[feature] = df[feature].astype('int64')


encode_categorical('group')
encode_categorical('parental_level_of_education')
df

# %%
df.dtypes

# %%
test_map = {'none': 0, 'completed': 1}
df['test_preparation_course'] = df['test_preparation_course'].map(test_map)
df

# %%
df.dtypes

# %%
sns.boxplot(data=df, x='math_score')

# %%


def remove_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

    return df


df = remove_outliers(df, 'math_score')

# %%
sns.boxplot(data=df, x='math_score')

# %%
scalar = StandardScalar()
# min_max_scalar = MinMaxScalar()

df['math_zscore'] = scalar.fit_transform(df['math_score'])
df
