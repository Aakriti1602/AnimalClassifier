import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('Downloads/zoo.csv')

factors = list(df.columns)
factors.remove('class_type')
factors.remove('animal_name')

X = df[factors].values
Y = df.class_type
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.8, random_state = 0)


knn = KNeighborsClassifier(n_neighbors = 2)

knn.fit(X_train, Y_train)
X_new=np.array([[1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0]])
pred = knn.predict(X_new)
print(pred)
