# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
categorical -> nominal(eye color, country of residence, marital status) and ordinal(education level - high school, bachelors degree, masters degree, phd, customer satisfaction rating - low, medium, high)
quantitative -> discrete(number of children in a family) and continuous(height, weight, time)

# %%
%matplotlib inline

# %%
df = pd.read_csv('Iris.csv')

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
df.columns

# %%
df['Species'].unique()

# %%
df.isnull().sum()

# %%
# hist plots
sns.histplot(data=df, x='SepalLengthCm', bins=10)

# %%
sns.histplot(data=df, x='PetalLengthCm', bins=10)

# %%
sns.boxplot(data=df, y='SepalLengthCm', x='Species')

# %%
# comparing plots
sns.kdeplot(data=df, x='SepalLengthCm')
