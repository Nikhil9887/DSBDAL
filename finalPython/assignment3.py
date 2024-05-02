# %%
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv('Iris.csv')
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
df.info()

# %%
df.isnull().sum()

# %%
df['Species'].unique()

# %%
species_map = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
df['Species'] = df['Species'].map(species_map)

# %%
df

# %%
sns.pairplot(data=df)

# %%
sns.boxplot(data=df, y="SepalLengthCm", hue="Species")

# %%
sns.boxplot(data=df, y="PetalLengthCm", hue="Species")

# %%
df.groupby('Species')['PetalLengthCm'].agg(
    ['mean', 'median', 'min', 'max', 'std']).reset_index()

# %%
df.groupby('Species')['SepalLengthCm'].agg(
    ['mean', 'median', 'min', 'max', 'std']).reset_index()

# %%
df.corr()

# %%
df.groupby('Species')['PetalLengthCm'].describe().reset_index()
