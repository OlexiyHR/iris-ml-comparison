from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['class'] = iris.target

print("Iris Dataset Sample:")
pd.set_option('display.max_columns', None)
print(iris_df.head())

# Розділення на ознаки (X) і мітки класів (y)
X = iris_df.iloc[:, :-1].values  # Всі колонки, окрім останньої
y = iris_df.iloc[:, -1].values   # Остання колонка - мітки класів

# Розділення на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ініціалізація методу KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Навчання моделі
knn.fit(X_train, y_train)

# Передбачення
y_pred = knn.predict(X_test)

# Оцінка точності
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nAccuracy of KNN model: {accuracy:.2f}%")