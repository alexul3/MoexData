import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')  # отключаем лишние предупреждения

# Загружаем модель сентимент-анализа для русского языка
print("Загрузка модели...")
sentiment_model = pipeline("sentiment-analysis",
                           model="cointegrated/rubert-tiny-sentiment-balanced")
print("Модель загружена!")

# Читаем исходный Excel-файл (замените на имя вашего файла)
df = pd.read_excel('news.xlsx', engine='openpyxl')

# Убедимся, что нужные столбцы присутствуют


# Объединяем заголовок и текст новости, заполняем пропуски
texts = (df['title'].fillna('') + ' ' + df['body'].fillna('')).tolist()

# Функция для получения числовой оценки тональности от -1 до 1
def get_sentiment_score(text):
    # Модель имеет ограничение на количество токенов (512), обрезаем текст
    if len(text) > 512:
        text = text[:512]
    result = sentiment_model(text)[0]
    label = result['label']      # 'positive', 'negative' или 'neutral'
    score = result['score']      # уверенность модели (0..1)
    if label == 'positive':
        return score
    elif label == 'negative':
        return -score
    else:  # neutral
        return 0.0

# Анализируем все новости с прогресс-баром
tqdm.pandas(desc="Анализ тональности")
df['тональность'] = texts
df['тональность'] = df['тональность'].progress_apply(get_sentiment_score)

# Сохраняем результат в новый файл
df.to_excel('news_with_sentiment.xlsx', index=False, engine='openpyxl')
print("Готово! Результат сохранён в 'news_with_sentiment.xlsx'")