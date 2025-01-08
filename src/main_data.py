from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Ограничение количества потоков CPU
torch.set_num_threads(2)

def main():

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

    # Загрузка токенизатора BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Подготовка данных для модели
    def prepare_dataset(texts, labels, tokenizer, max_length=64):
        encoded_inputs = tokenizer(texts.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], torch.tensor(labels))
        return dataset

    # Создание обучающего и тестового наборов данных
    train_data = prepare_dataset(train_texts, train_labels, tokenizer)
    test_data = prepare_dataset(test_texts, test_labels, tokenizer)

    # Сохранение обработанных данных
    train_data_path = os.path.join(output_dir, 'train_data.pt')
    test_data_path = os.path.join(output_dir, 'test_data.pt')
    torch.save(train_data, train_data_path)
    torch.save(test_data, test_data_path)
    print("Обработка данных завершена. Результаты сохранены в data/output.")

if __name__ == '__main__':
    main()