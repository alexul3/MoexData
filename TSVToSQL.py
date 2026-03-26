import psycopg2
import csv
import ast
from datetime import datetime

DB_NAME = "diplom"
DB_USER = "postgres"
DB_PASSWORD = "123"
DB_HOST = "localhost"
DB_PORT = "5432"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    title TEXT,
    score FLOAT,
    link TEXT UNIQUE,
    summary TEXT,
    published TIMESTAMP WITH TIME ZONE,
    tickers TEXT[]
);
"""

def parse_tickers(tickers_str):
    """Преобразует строку вида "['GAZP', 'NVTK']" в список Python."""
    if tickers_str and tickers_str.strip():
        try:
            return ast.literal_eval(tickers_str)
        except (SyntaxError, ValueError):
            return []
    return []

def main():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()

    cur.execute(CREATE_TABLE_SQL)
    conn.commit()

    # Чтение файла
    with open('data.tsv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            title = row['title']
            score = float(row['score']) if row['score'] else None
            link = row['link']
            summary = row['summary']
            published_str = row['published']
            tickers_str = row['tickers']

            # Парсинг даты
            try:
                published = datetime.strptime(published_str, "%a, %d %b %Y %H:%M:%S %z")
            except (ValueError, TypeError):
                published = None

            # Парсинг тикеров
            tickers_list = parse_tickers(tickers_str)

            # Вставка записи (игнорируем дубликаты по ссылке)
            insert_sql = """
            INSERT INTO news (title, score, link, summary, published, tickers)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (link) DO NOTHING;
            """
            cur.execute(insert_sql, (title, score, link, summary, published, tickers_list))

    conn.commit()
    cur.close()
    conn.close()
    print("Данные успешно загружены в PostgreSQL.")

if __name__ == "__main__":
    main()