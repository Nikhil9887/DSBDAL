# %%
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv("iris.csv")
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
df.shape

# %%
df.columns

# %%
df.isnull().sum()

# %%

# %%


def z_normalize(feature):
    global df
    df[feature] = (df[feature] - df[feature].mean()) / (df[feature].std())


cols = list(df.columns)
print(cols)
del cols[4]
print(cols)
for i in range(len(cols)):
    z_normalize(cols[i])

# %%
df

# %%
df.describe()

# %%
df.shape


# %%
gaussian = GaussianNB()

# %%
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
print(X)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# %%
gaussian.fit(X_train, y_train)

# %%
y_pred = gaussian.predict(X_test)
print(len(y_pred))

# %%
print(np.array(y_pred))

# %%
print(np.array(y_test))

# %%
cm = confusion_matrix(y_test, y_pred)
cm

# %%
accuracy = accuracy_score(y_test, y_pred)
accuracy

# %%
precision = precision_score(y_test, y_pred, average='macro')
precision

# %%
recall = recall_score(y_test, y_pred, average='macro')
recall

# %%
f1 = f1_score(y_test, y_pred, average='macro')
f1

# %%
a = df.shape
a

# %%
cm

# %%
# correct = 0
# wrong = 0
# for row in range(a[0]):
#     for col in range(a[1]):
#         if (row == col):
#             correct += cm[row][col]
#         else:
#             wrong += cm[row][col]

# accuracy = correct / (correct + wrong)
# print(accuracy)

# %%


# %%
tp, fn, fp, tn = confusion_matrix(y_test, y_pred, labels=[1, 0]).reshape(-1)
