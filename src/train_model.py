import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, '..', 'data', 'KNNAlgorithmDataset.csv')
reports_path = os.path.join(base_path, '..', 'reports')

if not os.path.exists(reports_path):
    os.makedirs(reports_path)

df = pd.read_csv(data_path)
df.drop(['id'], axis=1, inplace=True)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

error_rate = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 20), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.savefig(os.path.join(reports_path, 'elbow_method.png'))
print("📊 Elbow Method graph saved in 'reports/' folder.")

model = KNeighborsClassifier(n_neighbors=11)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Breast Cancer Prediction')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(reports_path, 'confusion_matrix.png')) # Save image
print("📊 Confusion Matrix graph saved in 'reports/' folder.")

print(f"\n✅ Final Accuracy with K=11: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\nClose the graph windows to finish the script.")
plt.show()