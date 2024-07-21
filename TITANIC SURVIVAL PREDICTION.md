# CODSOFT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("/content/Titanic-Dataset.csv")
df.head()

df.shape

df.describe()

df['Survived'].value_counts()

#let's visualize the count of survivals wrt class
sns.countplot(x='Survived', hue='Pclass', data=df)

df['Sex']

#let's visualize the count wrt gender
sns.countplot(x='Survived', hue='Sex', data=df)

#look at survived rate by sex
df.groupby('Sex')[['Survived']].mean()  

df['Sex'].unique()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

df.head()

df['Sex'],df['Survived']

sns.countplot(x='Sex', hue='Survived', data=df)

sns.heatmap(df.isnull(),yticklabels=False,cmap="viridis")

df.isnull().sum()

df.head()

df=df.drop(['Age'],axis=1)

df_final=df

df_final.head()

MODEL TRAINING

X=df[['Pclass','Sex']]
Y=df['Survived']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lnb=LogisticRegression()
lnb.fit(X_train,Y_train)

MODEL PREDICTION

pred=print(lnb.predict(X_test))

print(Y_test)

import warnings
warnings.filterwarnings('ignore')

res=log.predict([[2,0]])

if(res==0):
  print("So Sorry! They did not Survive")
else:
    print("Congrats! They Survived")
