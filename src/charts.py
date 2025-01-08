import matplotlib.pyplot as plt
import numpy as np

# Данные из экспериментов
experiments = [
    {"n_estimators": 50, "accuracy": 0.275, "f1_score": 0.255},
    {"n_estimators": 100, "accuracy": 0.289, "f1_score": 0.257}
]

# Извлечение данных для графиков
n_estimators = [exp["n_estimators"] for exp in experiments]
accuracies = [exp["accuracy"] for exp in experiments]
f1_scores = [exp["f1_score"] for exp in experiments]

# Создание диаграммы для accuracy
plt.figure(figsize=(10, 5))
plt.bar(n_estimators, accuracies, color=['blue', 'green'])
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Сравнение Accuracy для разных n_estimators')
plt.xticks(n_estimators)
plt.ylim(0, 1)  # Ограничение по оси Y для лучшей визуализации
for i, acc in enumerate(accuracies):
    plt.text(n_estimators[i], acc + 0.02, f"{acc:.4f}", ha='center')
plt.show()

# Создание диаграммы для F1-score
plt.figure(figsize=(10, 5))
plt.bar(n_estimators, f1_scores, color=['blue', 'green'])
plt.xlabel('n_estimators')
plt.ylabel('F1 Score')
plt.title('Сравнение F1 Score для разных n_estimators')
plt.xticks(n_estimators)
plt.ylim(0, 1)  # Ограничение по оси Y для лучшей визуализации
for i, f1 in enumerate(f1_scores):
    plt.text(n_estimators[i], f1 + 0.02, f"{f1:.4f}", ha='center')
plt.show()