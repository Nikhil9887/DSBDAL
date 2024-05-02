# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %% [markdown]
#

# %%
df = pd.read_csv("Titanic-Dataset.csv")

# %%
df

# %%
df.head()

# %%
df.tail()

# %%
df.dtypes

# %%
df.info()

# %%
df.describe()

# %%
df.isnull().sum()

# %%
df.shape

# %%
sns.heatmap(df.isnull(), yticklabels=False, cbar=False)

# %%
df = df.drop('Cabin', axis=1)

# %%
df.head()

# %%
df.columns

# %%
df.Pclass.unique()

# %%
df.PassengerId.unique()

# %%
df = df.drop('PassengerId', axis=1)

# %%
sns.heatmap(df.isnull(), yticklabels=False, cbar=False)

# %%
df

# %%
df['Survived'].value_counts()

# %%
sns.boxplot(data=df, x='Pclass', y='Age')

# %%
df.groupby('Pclass')['Age'].describe()

# %%
# we will replace null values in age column by median of the pclass they were in


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if (pd.isnull(Age)):
        if (Pclass == 1):
            return 37
        elif (Pclass == 2):
            return 29
        else:
            return 24
    else:
        return Age


df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis=1)

# %%
df.isnull().sum()

# %%
sns.heatmap(df.isnull(), yticklabels=False, cbar=False)

# %%
df.dropna(inplace=True)

# %%
df.isnull().sum()

# %%
# age vs survived
# pclass vs survived
# embarked vs survived
# parch vs survived
# sibsp vs surivived
# fare vs survived

sns.boxplot(data=df, y="Age", hue="Survived")

# %%
sns.countplot(data=df, x="Sex", hue="Survived")

# %%
sns.boxplot(data=df, y='Age', x='Sex', hue='Survived')

# %%
sns.stripplot(data=df, x='Fare', hue='Sex',  jitter=True, size=5)

# %%
df.dtypes
