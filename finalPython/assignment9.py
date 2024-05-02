# %%
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("Social_Network_Ads.csv")

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
df.corr(numeric_only=True)

# %%
sns.heatmap(df.corr(numeric_only=True))

# %%
df = df.drop('User ID', axis=1)

# %%
df.columns

# %%
sns.heatmap(df.corr(numeric_only=True))

# %%
sns.boxplot(data=df, x="Age", hue="Purchased")

# %%
sns.histplot(data=df, x="Age", hue="Purchased")

# %%
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# %%

# %%
X = df[['Age']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

lin_model = LogisticRegression()
lin_model.fit(X_train, y_train)


# %%

# %%
y_pred = lin_model.predict(X_test)

# %%
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mse)
print(r2)

# %%
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %%
y_pred

# %%
np.array(y_test)

# %%
tp = cm[1, 1]
print(tp)

# %%
tn = cm[0, 0]
print(tn)

# %%
fn = cm[1, 0]
print(fn)

# %%
fp = cm[0, 1]
print(fp)

# %%
accuracy = (tp + tn) / (tp + tn + fn + fp)
error = 1 - accuracy
precision = (tp) / (tp + fp)
recall = (tp) / (tp + fn)

print(accuracy)
print(error)
print(precision)
print(recall)
