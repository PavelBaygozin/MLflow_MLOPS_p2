from clearml import Task
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import torch
import os
import numpy as np

# Ограничение количества потоков CPU
torch.set_num_threads(2)

def main():
    # Инициализация задачи ClearML
    task = Task.init(project_name="MyProject", task_name="RandomForestExperiment")

    # Получение абсолютного пути к текущему файлу
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Загрузка предобработанных данных
    processed_data_dir = os.path.join(script_dir, '../data/output')
    train_data_path = os.path.join(processed_data_dir, 'train_data.pt')
    test_data_path = os.path.join(processed_data_dir, 'test_data.pt')

    train_data = torch.load(train_data_path, weights_only=False)
    test_data = torch.load(test_data_path, weights_only=False)

    print(f"Размер обучающего набора: {len(train_data)}")
    print(f"Размер тестового набора: {len(test_data)}")

    # Преобразование данных в формат, подходящий для RandomForest
    def prepare_data(dataset):
        features = dataset.tensors[0].numpy()  # Преобразуем тензор в массив NumPy
        labels = dataset.tensors[2].numpy()
        return features, labels

    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)

    # Эксперименты с разными значениями n_estimators
    n_estimators_list = [50, 100]  # Значения гиперпараметра для экспериментов

    for n_estimators in n_estimators_list:
        print(f"\nЗапуск эксперимента с n_estimators = {n_estimators}")

        # Создание и обучение модели RandomForest
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=2)  # n_jobs для использования 2 потоков
        model.fit(X_train, y_train)

        # Оценка модели
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")

        # Логирование параметров и метрик в ClearML
        task.get_logger().report_scalar(title="Metrics", series="Accuracy", value=accuracy, iteration=n_estimators)
        task.get_logger().report_scalar(title="Metrics", series="F1 Score", value=f1, iteration=n_estimators)
        task.get_logger().report_single_value(name="n_estimators", value=n_estimators)

        # Сохранение модели
        model_path = f"model_n_estimators_{n_estimators}.pkl"
        import joblib
        joblib.dump(model, model_path)
        task.upload_artifact(name=model_path, artifact_object=model_path)

if __name__ == "__main__":
    main()