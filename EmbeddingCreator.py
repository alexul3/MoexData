import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # для прогресс-бара

# ----------------------------
# 1. Загрузка данных
# ----------------------------
INPUT_FILE = "news_test.xlsx"  # укажите путь к вашему файлу
OUTPUT_FILE = r"D:\ot\news_with_embeddings.xlsx"

# Читаем Excel, предполагаем, что первый столбец содержит новости
df = pd.read_excel(INPUT_FILE, header=None)
text_column = df.columns[0]  # первая колонка
print(f"Загружено {len(df)} новостей")


# ----------------------------
# 2. Функция очистки текста
# ----------------------------
def clean_news_text(text):
    if not isinstance(text, str):
        return ""

    # Убираем внешние кавычки, если весь текст в них обёрнут
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]

    # Удаляем HTML-теги
    text = re.sub(r'<.*?>', ' ', text)

    # Удаляем URL (http, https, www)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # Удаляем эмодзи и нестандартные символы (оставляем буквы, цифры, базовую пунктуацию)
    text = re.sub(r'[^\w\s.,!?:;%\-–—«»()]', ' ', text, flags=re.UNICODE)

    # Заменяем переносы строк и лишние пробелы на один пробел
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Применяем очистку ко всем текстам
df['clean_text'] = df[text_column].apply(clean_news_text)
print("Очистка текста завершена")

# ----------------------------
# 3. Загрузка модели и генерация эмбеддингов
# ----------------------------
MODEL_NAME = 'deepvk/USER2-base'
print(f"Загружаем модель {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME,
                            trust_remote_code=True)  # trust_remote_code может потребоваться для кастомных моделей

# Получаем список очищенных текстов
texts = df['clean_text'].tolist()

# Генерация эмбеддингов батчами (для ускорения и прогресс-бара)
print("Генерация эмбеддингов...")
embeddings = model.encode(
    texts,
    batch_size=32,  # можно увеличить, если хватает памяти GPU
    show_progress_bar=True,
    convert_to_numpy=True
)

# Размерность эмбеддинга (обычно 768)
embedding_dim = embeddings.shape[1]
print(f"Получены эмбеддинги размерности {embedding_dim}")

# ----------------------------
# 4. Сохранение результата в Excel
# ----------------------------
# Создаём DataFrame с оригинальным текстом, очищенным текстом и колонками эмбеддингов
embedding_columns = [f"emb_{i}" for i in range(embedding_dim)]
embedding_df = pd.DataFrame(embeddings, columns=embedding_columns)

# Объединяем с исходными данными
result_df = pd.concat([
    df[[text_column]].rename(columns={text_column: 'original_text'}),
    df[['clean_text']],
    embedding_df
], axis=1)

# Сохраняем в Excel
result_df.to_excel(OUTPUT_FILE, index=False)
print(f"Результат сохранён в {OUTPUT_FILE}")