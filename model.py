from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df=pd.read_csv('iris.csv')

X=df[["Sepal_Length","Sepal_Width","Petal_Length","Petal_Width"]]
y=df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)


pickle.dump(classifier,open("model.pkl","wb"))



