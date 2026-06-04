import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path

# ================== НАСТРОЙКИ ПОДКЛЮЧЕНИЯ ==================
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',
    'user': 'postgres',
    'password': '123'
}

EXCEL_NEWS = 'data_merge.xlsx'
CANDLES_DIR = Path('../candles_data')

# ================== ФУНКЦИИ ==================
def extract_ticker(filename):
    return filename.stem.split('_')[0].upper()

def ensure_ticker_exists(cur, ticker_symbol):
    """Получить id тикера (добавляет, если нет)."""
    cur.execute("SELECT id FROM tickers WHERE symbol = %s", (ticker_symbol,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO tickers (symbol) VALUES (%s) RETURNING id", (ticker_symbol,))
    return cur.fetchone()[0]

# ================== ЧТЕНИЕ НОВОСТЕЙ ==================
COLUMNS = ['title', 'text', 'date', 'time', 'tags', 'source', 'sentiment', 'tickers_raw']
df = pd.read_excel(EXCEL_NEWS, sheet_name='Sheet1', header=None, names=COLUMNS)

# Объединяем дату и время
df['published_at'] = pd.to_datetime(
    df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce'
)
df = df.dropna(subset=['published_at'])
df['sentiment'] = df['sentiment'].astype(str).str.replace(',', '.').astype(float)

# Парсим тикеры из строки
ticker_set = set()
news_ticker_links = []
for idx, row in df.iterrows():
    raw = row['tickers_raw']
    if pd.isna(raw) or str(raw).strip() == '':
        continue
    symbols = [s.strip() for s in str(raw).split(',') if s.strip()]
    for sym in symbols:
        ticker_set.add(sym)
        news_ticker_links.append((idx, sym))

# Присваиваем тикерам id (1..N) для последующей вставки
ticker_sorted = sorted(ticker_set)
ticker_to_id = {sym: i+1 for i, sym in enumerate(ticker_sorted)}

# ================== ПОДКЛЮЧЕНИЕ ==================
conn = psycopg2.connect(**DB_CONFIG)
conn.autocommit = False
cur = conn.cursor()

try:
    # ---------- СОЗДАНИЕ ТАБЛИЦ ----------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tickers (
            id SERIAL PRIMARY KEY,
            symbol TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS news (
            id SERIAL PRIMARY KEY,
            title TEXT,
            text TEXT,
            published_at TIMESTAMP,
            sentiment DOUBLE PRECISION,
            source TEXT,
            tags TEXT
        );
        CREATE TABLE IF NOT EXISTS news_ticker (
            news_id INTEGER REFERENCES news(id),
            ticker_id INTEGER REFERENCES tickers(id),
            PRIMARY KEY (news_id, ticker_id)
        );
        CREATE TABLE IF NOT EXISTS price_daily (
            ticker_id INTEGER REFERENCES tickers(id),
            date DATE,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume BIGINT,
            PRIMARY KEY (ticker_id, date)
        );
        CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_daily(ticker_id, date DESC);
    """)

    # ================== 1. ТИКЕРЫ (из новостей) ==================
    # Вставляем с явными id, подавляя дубликаты
    execute_values(cur,
        "INSERT INTO tickers (id, symbol) VALUES %s ON CONFLICT (symbol) DO NOTHING",
        [(tid, sym) for sym, tid in ticker_to_id.items()]
    )
    # Синхронизируем последовательность, чтобы избежать конфликта в будущем
    cur.execute("""
        SELECT setval('tickers_id_seq', coalesce((SELECT max(id) FROM tickers), 1))
    """)

    # ================== 2. НОВОСТИ ==================
    # Резервируем id для новостей
    cur.execute("SELECT nextval('news_id_seq') FROM generate_series(1, %s)", (len(df),))
    news_ids = [row[0] for row in cur.fetchall()]
    idx_to_news_id = dict(zip(df.index, news_ids))

    news_data = []
    for _, row in df.iterrows():
        news_data.append((
            row['title'] if not pd.isna(row['title']) else None,
            row['text'] if not pd.isna(row['text']) else None,
            row['published_at'],
            row['sentiment'],
            row['source'] if not pd.isna(row['source']) else None,
            row['tags'] if not pd.isna(row['tags']) else None
        ))

    execute_values(cur,
        "INSERT INTO news (id, title, text, published_at, sentiment, source, tags) VALUES %s",
        [(news_ids[i], *news_data[i]) for i in range(len(news_ids))]
    )
    # Синхронизируем последовательность для news
    cur.execute("""
        SELECT setval('news_id_seq', coalesce((SELECT max(id) FROM news), 1))
    """)

    # ================== 3. СВЯЗЬ НОВОСТИ-ТИКЕРЫ ==================
    links = []
    for df_idx, sym in news_ticker_links:
        news_id = idx_to_news_id[df_idx]
        tid = ticker_to_id[sym]
        links.append((news_id, tid))
    links = list(set(links))

    execute_values(cur,
        "INSERT INTO news_ticker (news_id, ticker_id) VALUES %s ON CONFLICT DO NOTHING",
        links
    )

    # ================== 4. ЗАГРУЗКА ЦЕН ИЗ ПАПКИ ==================
    candle_files = list(CANDLES_DIR.glob('*_candles_*.xlsx'))
    print(f"Найдено файлов свечей: {len(candle_files)}")

    for fpath in candle_files:
        ticker = extract_ticker(fpath)
        print(f"Обрабатываю {fpath.name} (тикер {ticker})...")

        df_price = pd.read_excel(fpath, parse_dates=['TRADEDATE'])
        df_price = df_price[['TRADEDATE', 'open', 'close', 'high', 'low', 'value']].copy()
        df_price.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
        df_price['date'] = df_price['date'].dt.date
        df_price = df_price.dropna()

        ticker_id = ensure_ticker_exists(cur, ticker)   # теперь безопасно

        price_tuples = [
            (
                ticker_id,
                row['date'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                int(row['volume']) if pd.notna(row['volume']) else 0
            )
            for _, row in df_price.iterrows()
        ]

        execute_values(cur,
            """
            INSERT INTO price_daily (ticker_id, date, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (ticker_id, date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
            """,
            price_tuples
        )
        print(f"  -> {len(price_tuples)} записей")

    conn.commit()
    print("Все данные успешно загружены.")

except Exception as e:
    conn.rollback()
    print(f"Ошибка: {e}")
    raise
finally:
    cur.close()
    conn.close()