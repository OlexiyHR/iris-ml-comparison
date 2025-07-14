# ML Classification Comparison

У цьому проєкті порівнюються чотири класичні алгоритми класифікації на основі датасету Iris:

- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Multi-layer Perceptron (MLP)

## Дані
Використовується вбудований датасет `Iris` з бібліотеки `sklearn`.

## Ціль
Передбачити вид ірису (Setosa, Versicolor, Virginica) на основі довжини і ширини пелюсток і чашолистків.

## Метрики
Точність моделі (Accuracy)

## Результати
| Модель           | Accuracy |
|------------------|----------|
| KNN              | 100.00%  |
| Naive Bayes      | 97.78%   |
| Decision Tree    | 100.00%  |
| MLP              | 97.78%   |

## Використані бібліотеки
- Scikit-learn
- Pandas
- Matplotlib / Seaborn (опціонально для графіків)

## Як запустити

1. Встановіть необхідні залежності:
bash
pip install -r requirements.txt
2. Запустіть потрібний скрипт із одного з цих файлів:

- `KNeighbors.py`
- `NaiveBayes.py`
- `DecisionTree.py`
- `MLP.py`

Наприклад:

```bash
python KNeighbors.py

