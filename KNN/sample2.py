import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# eğitim seviyesi 0=lise , 1 lisans, 2 YL
# tecrübe yılı

data = [
    [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0],
    [1, 0, 0], [1, 2, 0], [1, 2, 1], [1, 2, 0],
    [1, 4, 1], [1, 5, 1], [2, 0, 0], [2, 1, 1],
    [2, 2, 1], [2, 3, 1], [2, 4, 1], [2, 5, 1],
    [2, 6, 1], [2, 7, 1], [2, 8, 1], [2, 9, 1]
]

df = pd.DataFrame(data,columns=["school","year","hired"])

X = df[["school","year"]]
y = df["hired"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

k_values = range(1,16)
scores = []

# Accuracy score based k selection
for k in k_values:
  model = KNeighborsClassifier(n_neighbors=k)
  model.fit(X_train,y_train)
  y_prediction = model.predict(X_test)
  accuracy = accuracy_score(y_test,y_prediction)
  scores.append(accuracy)

print(scores)

applicant = np.array([[0,3]])
prediction = model.predict(applicant)
print("Evet" if prediction[0]==1 else "Hayır")

#Cross validation