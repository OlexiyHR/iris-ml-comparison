from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Завантаження набору Iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['class'] = iris.target

# Виведення перших рядків даних
pd.set_option('display.max_columns', None)
print("Iris Dataset Sample:")
print(iris_df.head())

# Розділення на ознаки (X) і мітки класів (y)
X = iris_df.iloc[:, :-1].values  # Всі колонки, окрім останньої
y = iris_df.iloc[:, -1].values   # Остання колонка - мітки класів

# Розділення на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ініціалізація Наївного Байєсівського класифікатора
nb = GaussianNB()

# Навчання моделі
nb.fit(X_train, y_train)

# Передбачення
y_pred = nb.predict(X_test)

# Оцінка точності
accuracy = accuracy_score(y_test, y_pred) * 100  # Конвертуємо точність у відсотки
print(f"\nAccuracy of Naive Bayes model: {accuracy:.2f}%")

# Передбачення ймовірностей для тестового набору
probabilities = nb.predict_proba(X_test)

# Побудова графіку ймовірностей
classes = iris.target_names  # Назви класів
x = np.arange(len(X_test))   # Номери тестових зразків

# Побудова стовпчастого графіку ймовірностей
fig, ax = plt.subplots(figsize=(12, 6))
for i, class_name in enumerate(classes):
    ax.bar(x + i*0.25, probabilities[:, i], width=0.25, label=f'Class {class_name}')

# Налаштування графіку
ax.set_xlabel("Test Sample Index")
ax.set_ylabel("Probability")
ax.set_title("Predicted Probabilities for Each Class (Naive Bayes)")
ax.legend()
plt.xticks(x)
plt.show()