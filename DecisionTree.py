from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Завантаження набору Iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['class'] = iris.target

# Розділення на ознаки (X) і мітки класів (y)
X = iris_df.iloc[:, :-1].values  # Всі колонки, окрім останньої
y = iris_df.iloc[:, -1].values   # Остання колонка - мітки класів

# Розділення на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ініціалізація розв'язуючого дерева
tree = DecisionTreeClassifier(random_state=42, max_depth=3)  # max_depth обмежує глибину дерева

# Навчання моделі
tree.fit(X_train, y_train)

# Передбачення
y_pred = tree.predict(X_test)

# Оцінка точності
accuracy = accuracy_score(y_test, y_pred) * 100  # Точність у відсотках
print(f"Accuracy of Decision Tree model: {accuracy:.2f}%")

# Виведення логіки дерева у текстовій формі
tree_rules = export_text(tree, feature_names=iris.feature_names)
print("\nDecision Tree Rules:\n")
print(tree_rules)

# Візуалізація дерева
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()