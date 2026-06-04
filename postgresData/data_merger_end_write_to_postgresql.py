import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# ================== НАСТРОЙКИ ПОДКЛЮЧЕНИЯ ==================
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',
    'user': 'postgres',
    'password': '123'
}

EXCEL_FILE = '../data_merge.xlsx'
SHEET_NAME = 'Sheet1'

# Столбцы в Excel (без заголовка, порядок важен)
COLUMNS = ['title', 'text', 'date', 'time', 'tags', 'source', 'sentiment', 'tickers_raw']

# ================== ЧТЕНИЕ EXCEL ==================
df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, header=None, names=COLUMNS)

# Объединяем дату и время в timestamp
df['published_at'] = pd.to_datetime(
    df['date'].astype(str) + ' ' + df['time'].astype(str),
    errors='coerce'
)

# Очистка: убираем строки, где дата не распозналась (опционально)
df = df.dropna(subset=['published_at'])

# Приводим sentiment к float (замена запятых на точки, если нужно)
df['sentiment'] = df['sentiment'].astype(str).str.replace(',', '.').astype(float)

# ================== ПАРСИНГ ТИКЕРОВ ==================
# Создаём словарь {тикер: id} и список связей (индекс новости -> тикеры)
ticker_set = set()
news_ticker_links = []  # (idx_новости, символ_тикера)

for idx, row in df.iterrows():
    raw = row['tickers_raw']
    if pd.isna(raw) or str(raw).strip() == '':
        continue
    # Разделяем по запятой, чистим пробелы, убираем пустые строки
    symbols = [s.strip() for s in str(raw).split(',') if s.strip()]
    for sym in symbols:
        ticker_set.add(sym)
        news_ticker_links.append((idx, sym))

# Присваиваем тикерам id (для последующей вставки)
ticker_to_id = {sym: i+1 for i, sym in enumerate(sorted(ticker_set))}  # ID с 1

# ================== ПОДКЛЮЧЕНИЕ К POSTGRES ==================
conn = psycopg2.connect(**DB_CONFIG)
conn.autocommit = False  # будем управлять транзакцией вручную
cur = conn.cursor()

try:
    # ---------- СОЗДАНИЕ ТАБЛИЦ (если не существуют) ----------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tickers (
            id SERIAL PRIMARY KEY,
            symbol TEXT UNIQUE NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id SERIAL PRIMARY KEY,
            title TEXT,
            text TEXT,
            published_at TIMESTAMP,
            sentiment DOUBLE PRECISION,
            source TEXT,
            tags TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_ticker (
            news_id INTEGER REFERENCES news(id),
            ticker_id INTEGER REFERENCES tickers(id),
            PRIMARY KEY (news_id, ticker_id)
        );
    """)

    # ---------- ВСТАВКА ТИКЕРОВ ----------
    # Игнорируем конфликты (если тикер уже существует)
    execute_values(cur,
        "INSERT INTO tickers (id, symbol) VALUES %s ON CONFLICT (symbol) DO NOTHING",
        [(tid, sym) for sym, tid in ticker_to_id.items()]
    )

    # ---------- ВСТАВКА НОВОСТЕЙ ----------
    # Готовим данные для вставки, сохраняем соответствие индекс -> news_id
    news_data = []
    for idx, row in df.iterrows():
        news_data.append((
            row['title'] if not pd.isna(row['title']) else None,
            row['text'] if not pd.isna(row['text']) else None,
            row['published_at'],
            row['sentiment'],
            row['source'] if not pd.isna(row['source']) else None,
            row['tags'] if not pd.isna(row['tags']) else None
        ))

    # Вставляем все новости и возвращаем id (через RETURNING)
    insert_news_sql = """
        INSERT INTO news (title, text, published_at, sentiment, source, tags)
        VALUES %s
        RETURNING id
    """
    # execute_values может вернуть список id? Будем вставлять и забирать id.
    # Альтернатива: использовать отдельный запрос для каждой строки, но это медленно.
    # Воспользуемся генератором и фиксируем id через временную таблицу или смещение.
    # Самый простой способ: вставлять построчно с RETURNING, но для больших данных это неэффективно.
    # Для учебных целей допустимо. Либо использовать nextval последовательности заранее.

    # Более элегантный способ: выполнить массовую вставку с RETURNING через
    # конструкцию WITH ... INSERT ... RETURNING, передавая данные как VALUES.
    # Например:
    # WITH rows AS (
    #     INSERT INTO news (...) VALUES (...), (...), ... RETURNING id
    # )
    # SELECT id FROM rows;
    # execute_values может помочь, но не возвращает id автоматически.
    # Поэтому воспользуемся временной таблицей или вычислим id заранее.

    # Вариант: вычислим id заранее через nextval и вставим с явным id.
    # Получим начальное значение последовательности:
    cur.execute("SELECT nextval('news_id_seq') FROM generate_series(1, %s)", (len(news_data),))
    new_ids = [row[0] for row in cur.fetchall()]  # список будущих id
    # Вставляем с явными id
    execute_values(cur,
        "INSERT INTO news (id, title, text, published_at, sentiment, source, tags) VALUES %s",
        [(new_ids[i], *news_data[i]) for i in range(len(news_data))]
    )
    # Сохраняем отображение индекс DataFrame -> news_id
    idx_to_news_id = dict(zip(df.index, new_ids))

    # ---------- ВСТАВКА СВЯЗЕЙ ----------
    links_for_insert = []
    for df_idx, sym in news_ticker_links:
        news_id = idx_to_news_id[df_idx]
        ticker_id = ticker_to_id[sym]
        links_for_insert.append((news_id, ticker_id))

    # Убираем возможные дубликаты (если вдруг есть)
    links_for_insert = list(set(links_for_insert))

    execute_values(cur,
        "INSERT INTO news_ticker (news_id, ticker_id) VALUES %s ON CONFLICT DO NOTHING",
        links_for_insert
    )

    conn.commit()
    print(f"Успешно перенесено: {len(news_data)} новостей, {len(ticker_set)} тикеров, {len(links_for_insert)} связей.")



except Exception as e:
    conn.rollback()
    print(f"Ошибка при переносе данных: {e}")
    raise
finally:
    cur.close()
    conn.close()

