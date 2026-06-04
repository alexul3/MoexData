import pandas as pd

# Читаем исходный Excel-файл
df = pd.read_excel("for_embedding.xlsx")

# Проверяем наличие нужных столбцов
if 'title' not in df.columns or 'body' not in df.columns:
    raise ValueError("В файле отсутствуют столбцы 'title' и/или 'body'")

# Заполняем пропуски пустыми строками
df['title'] = df['title'].fillna('')
df['body'] = df['body'].fillna('')

# Создаём колонку 'news' – объединение заголовка и текста
df['news'] = (df['title'] + ' ' + df['body']).str.strip()

# Записываем только колонку 'news' в новый Excel-файл
df[['news']].to_excel("news_embedding_input.xlsx", index=False)

print(f"Создан Excel-файл 'news_embedding_input.xlsx' с {len(df)} записями.")