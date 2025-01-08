import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import os

# Ограничение количества потоков CPU
torch.set_num_threads(2)

def main():
    try:
        # Получение абсолютного пути к текущему файлу
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Создание директории для обработанных данных, если она отсутствует
        output_dir = os.path.join(script_dir, '../data/output')
        os.makedirs(output_dir, exist_ok=True)

        # Загрузка исходных данных
        raw_data_path = os.path.join(script_dir, '../data/input/train.csv')
        dataset = pd.read_csv(raw_data_path)

        # Проверка на отсутствующие значения в колонке 'Text'
        print("Количество пропущенных значений в текстах:", dataset['Text'].isnull().sum())

        # Заполнение пропущенных значений пустыми строками
        dataset['Text'] = dataset['Text'].fillna('')

        # Извлечение текстов и меток
        text_data = dataset['Text'].values
        sentiment_labels = dataset['Sentiment'].values

        # Преобразование текстовых меток в числовые
        encoder = LabelEncoder()
        sentiment_labels = encoder.fit_transform(sentiment_labels)

        # Разделение данных на обучающую и тестовую выборки
        train_texts, test_texts, train_labels, test_labels = train_test_split(text_data, sentiment_labels, test_size=0.8, random_state=42)

        # Векторизация текстов с использованием CountVectorizer
        vectorizer = CountVectorizer(max_features=5000)  # Ограничиваем количество признаков
        X_train = vectorizer.fit_transform(train_texts).toarray()
        X_test = vectorizer.transform(test_texts).toarray()

        # Преобразование данных в тензоры PyTorch
        train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
        test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))

        # Сохранение обработанных данных
        train_data_path = os.path.join(output_dir, 'train_data.pt')
        test_data_path = os.path.join(output_dir, 'test_data.pt')
        torch.save(train_data, train_data_path)
        torch.save(test_data, test_data_path)
        print("Обработка данных завершена. Результаты сохранены в data/output.")

    except Exception as e:
        print(f"Ошибка в main_data.py: {e}")
        raise

if __name__ == '__main__':
    main()