import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import time
import requests
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------
# 1. Загрузка новостей
# ------------------------------
news_path = 'rework_data_expanded.xlsx'
df_news = pd.read_excel(news_path)

df_news['datetime'] = pd.to_datetime(
    df_news['date'].astype(str) + ' ' + df_news['time'].astype(str),
    errors='coerce'
)
before = len(df_news)
df_news = df_news.dropna(subset=['datetime'])
print(f'Удалено {before - len(df_news)} строк с некорректной датой/временем')

df_news['ticker'] = df_news['Ticket'].str.upper().str.strip()

# ------------------------------
# 2. Определяем период загрузки цен
# ------------------------------
min_dt = df_news['datetime'].min()
max_dt = df_news['datetime'].max()

start_date = (min_dt - timedelta(days=7)).strftime('%Y-%m-%d')
end_date = (max_dt + timedelta(days=2)).strftime('%Y-%m-%d')
print(f'Диапазон цен: {start_date} – {end_date}')

tickers = df_news['ticker'].unique()
print(f'Тикеров для загрузки: {len(tickers)}')

# ------------------------------
# 3. Настройка кэша и устойчивой сессии
# ------------------------------
cache_dir = Path('cache')
cache_dir.mkdir(exist_ok=True)


def get_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.mount('https://', HTTPAdapter(max_retries=retries))
    return s


# ------------------------------
# 4. Функция загрузки свечей с кэшированием
# ------------------------------
BASE_URL = 'https://iss.moex.com/iss/engines/stock/markets/shares/securities'


def fetch_candles_with_cache(ticker, interval, from_date, till_date):
    cache_file = cache_dir / f"{ticker}_{interval}_{from_date}_{till_date}.csv"

    if cache_file.exists():
        if cache_file.stat().st_size > 0:
            try:
                df = pd.read_csv(cache_file, parse_dates=['begin'])
                if not df.empty:
                    return df
            except Exception:
                pass
        cache_file.unlink(missing_ok=True)

    all_data = []
    start = 0
    limit = 500
    session = get_session()

    while True:
        url = f"{BASE_URL}/{ticker}/candles.json"
        params = {
            'from': from_date,
            'till': till_date,
            'interval': interval,
            'start': start,
            'limit': limit,
        }
        try:
            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            j = resp.json()
            cols = j['candles']['columns']
            rows = j['candles']['data']
        except Exception as e:
            print(f'Ошибка {ticker} (start={start}): {e}. Прерываем для этого тикера.')
            break

        if not rows:
            break

        df_chunk = pd.DataFrame(rows, columns=cols)
        all_data.append(df_chunk)

        if len(rows) < limit:
            break
        start += limit
        time.sleep(0.05)

    if not all_data:
        pd.DataFrame().to_csv(cache_file, index=False)
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df['begin'] = pd.to_datetime(df['begin'])
    df.to_csv(cache_file, index=False)
    return df


# ------------------------------
# 5. Параллельная загрузка всех тикеров
# ------------------------------
close_frames = []
open_frames = []


def load_one_ticker(ticker):
    df_10m = fetch_candles_with_cache(ticker, 10, start_date, end_date)
    df_day = fetch_candles_with_cache(ticker, 24, start_date, end_date)
    return ticker, df_10m, df_day


print('Начинаю загрузку рыночных данных...')
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(load_one_ticker, t): t for t in tickers}
    for i, future in enumerate(as_completed(futures), 1):
        ticker, df_10m, df_day = future.result()
        if not df_10m.empty:
            tmp = df_10m[['begin', 'close']].copy()
            tmp['ticker'] = ticker
            close_frames.append(tmp)
        if not df_day.empty:
            tmp = df_day[['begin', 'open']].copy()
            tmp['ticker'] = ticker
            open_frames.append(tmp)
        if i % 5 == 0 or i == len(tickers):
            print(f'Обработано {i}/{len(tickers)} тикеров')

# ------------------------------
# 6. Сборка DataFrames свечей (без глобальной сортировки – она будет внутри цикла)
# ------------------------------
if close_frames:
    df_close = pd.concat(close_frames, ignore_index=True)
    df_close = df_close.dropna(subset=['close'])
else:
    raise ValueError('Не удалось загрузить ни одной 10-минутной свечи')

if open_frames:
    df_open = pd.concat(open_frames, ignore_index=True)
    df_open = df_open.dropna(subset=['open'])
else:
    raise ValueError('Не удалось загрузить ни одной дневной свечи')

print(f'Загружено 10-минутных свечей: {len(df_close)}')
print(f'Загружено дневных свечей: {len(df_open)}')

# ------------------------------
# 7. Сопоставление новостей с ценами – НАДЁЖНЫЙ МЕТОД (по тикерам)
# ------------------------------
result_frames = []

for ticker in tickers:
    # Новости для этого тикера
    news_t = df_news[df_news['ticker'] == ticker].sort_values('datetime')
    # Свечи для этого тикера
    close_t = df_close[df_close['ticker'] == ticker].sort_values('begin')
    open_t = df_open[df_open['ticker'] == ticker].sort_values('begin')

    if close_t.empty or open_t.empty:
        continue  # нет данных – пропускаем тикер

    # merge_asof backward для получения close до новости
    merged = pd.merge_asof(
        news_t,
        close_t[['begin', 'close']],
        left_on='datetime',
        right_on='begin',
        direction='backward',
        suffixes=('', '_pre')
    )
    # merge_asof forward для получения open после новости
    merged = pd.merge_asof(
        merged.sort_values('datetime'),
        open_t[['begin', 'open']],
        left_on='datetime',
        right_on='begin',
        direction='forward',
        suffixes=('', '_next')
    )
    result_frames.append(merged)

if not result_frames:
    raise ValueError('Ни для одного тикера не найдены полные данные')

df_result = pd.concat(result_frames, ignore_index=True)

# ------------------------------
# 8. Расчёт целевой переменной
# ------------------------------
mask = (df_result['close'] > 0) & (df_result['open'] > 0)
df_result = df_result[mask]

df_result['Target'] = np.log(df_result['open'] / df_result['close'])

# Удаляем вспомогательные колонки
df_result = df_result.drop(columns=['begin', 'begin_next', 'close', 'open'], errors='ignore')

# Опциональная обрезка выбросов
CAP = 0.25
df_result['Target'] = df_result['Target'].clip(-CAP, CAP)

# ------------------------------
# 9. Сохранение результата
# ------------------------------
output_path = 'rework_data_with_target.xlsx'
df_result.to_excel(output_path, index=False)
print(f'Готово! Результат сохранён в {output_path}')