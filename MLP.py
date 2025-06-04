from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Завантаження набору Iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['class'] = iris.target

# Розділення на ознаки (X) і мітки класів (y)
X = iris_df.iloc[:, :-1].values  # Всі колонки, окрім останньої
y = iris_df.iloc[:, -1].values   # Остання колонка - мітки класів

# Розділення на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ініціалізація нейронної мережі
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)

# Навчання моделі
mlp.fit(X_train, y_train)

# Передбачення
y_pred = mlp.predict(X_test)

# Оцінка точності
accuracy = accuracy_score(y_test, y_pred) * 100  # Точність у відсотках
print(f"Accuracy of Neural Network model: {accuracy:.2f}%")

# Отримання ймовірностей для кожного класу
probabilities = mlp.predict_proba(X_test)

# Побудова графіку ймовірностей для кожного класу
x = np.arange(len(X_test))
classes = iris.target_names

plt.figure(figsize=(12, 6))
for i, class_name in enumerate(classes):
    plt.bar(x + i * 0.25, probabilities[:, i], width=0.25, label=f'Class {class_name}')

plt.xlabel("Test Sample Index")
plt.ylabel("Probability")
plt.title("Predicted Probabilities for Each Class (Neural Network)")
plt.legend()
plt.xticks(x)
plt.show()