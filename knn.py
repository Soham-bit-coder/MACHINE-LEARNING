import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('KNNAlgorithmDataset.csv')

# Drop unnecessary ID column
data = data.drop(columns=['id'], errors='ignore')

# Remove extra empty / unnamed column
data = data.dropna(axis=1)

# Encode target column
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

print(data.tail())
print(data.columns)
print(data.isnull().sum())
print(data.dtypes)
print(data.describe())
features=['radius_mean',
'texture_mean',
'perimeter_mean',
'area_mean',
'concavity_mean',
'radius_worst',
'perimeter_worst',
'area_worst',
'concavity_worst']
# Feature–label split (best practice)
X = data[features].values
y = data['diagnosis'].values

print(X)
print(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)
# Train-test split (best practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# KNN model creation
best_score = 0
best_k = 0

# for i in range(1, 16,2):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     print(i,accuracy_score(y_test, knn.predict(X_test)))

#     if accuracy_score(y_test, knn.predict(X_test)) > best_score:
#         best_score = accuracy_score(y_test, knn.predict(X_test))
#         best_k = i
# knn = KNeighborsClassifier(n_neighbors=best_k)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(best_k,best_score)

pred=knn.predict(X_test)
print(pred)
print(accuracy_score(y_test, pred))

from sklearn.decomposition import PCA

# Reduce 9 features to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Train KNN on PCA data
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_pca, y)

# Meshgrid
x_min, x_max = X_pca[:,0].min() - 1, X_pca[:,0].max() + 1
y_min, y_max = X_pca[:,1].min() - 1, X_pca[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap=plt.cm.coolwarm, edgecolor='k', s=50)
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.title("KNN Decision Boundary (2 Classes, k=5, PCA)")
plt.show()


import random

# Pick a random index from the dataset
random_index = random.randint(0, len(X) - 1)

random_sample = X[random_index].reshape(1, -1)
actual_label = y[random_index]

prediction = knn.predict(random_sample)

print("Random sample index:", random_index)
print("Predicted class:", "Malignant" if prediction[0] == 1 else "Benign")
print("Actual class   :", "Malignant" if actual_label == 1 else "Benign")
