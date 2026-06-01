import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ------------------------------
# 1. Загрузка датасета новостей
# ------------------------------
news_df = pd.read_excel('new_data.xlsx')

# Приводим столбец с датой к типу datetime.date
# Ожидаемые названия столбцов: Дата, Время, Тон новости, тикер компании
news_df['Дата'] = pd.to_datetime(news_df['Дата']).dt.date


# ------------------------------
# 2. Подготовка тикеров для Yahoo Finance
# ------------------------------
# Считаем, что в столбце 'тикер компании' хранятся короткие имена (SBER, GAZP и т.д.)
# Для MOEX обычно добавляется суффикс '.ME'
def to_yahoo_ticker(ticker: str) -> str:
    # Если тикер уже содержит точку, оставляем как есть, иначе добавляем '.ME'
    return ticker if '.' in ticker else f'{ticker}.ME'


news_df['yahoo_ticker'] = news_df['тикер компании'].astype(str).apply(to_yahoo_ticker)
unique_tickers = news_df['yahoo_ticker'].unique().tolist()

# ------------------------------
# 3. Загрузка исторических цен
# ------------------------------
min_date = news_df['Дата'].min()
max_date = news_df['Дата'].max()

# Расширим диапазон, чтобы гарантированно получить следующий торговый день
# Берём +10 дней с запасом – yfinance отдаст только реальные торги
start_download = min_date - timedelta(days=5)
end_download = max_date + timedelta(days=10)

print(f"Загружаем котировки с {start_download} по {end_download} для {len(unique_tickers)} тикеров...")

# Скачиваем скорректированные цены закрытия ('Adj Close') для всех тикеров разом
prices_raw = yf.download(
    tickers=unique_tickers,
    start=start_download,
    end=end_download,
    auto_adjust=True,  # чтобы получить скорректированную цену
    progress=False,
    group_by='ticker'
)

# Формируем словарь: тикер -> Series с датами в индексе и ценой закрытия
price_series = {}
if len(unique_tickers) > 1:
    for ticker in unique_tickers:
        if ticker in prices_raw.columns.levels[0]:
            # Извлекаем 'Close' для тикера и удаляем пропуски
            series = prices_raw[ticker]['Close'].dropna()
            if not series.empty:
                price_series[ticker] = series
else:
    # Случай одного тикера – структура колонок без мультииндекса
    series = prices_raw['Close'].dropna()
    if not series.empty:
        price_series[unique_tickers[0]] = series

print(f"Котировки получены для {len(price_series)} тикеров.")


# ------------------------------
# 4. Функция расчёта целевой переменной
# ------------------------------
def compute_price_change(row):
    ticker = row['yahoo_ticker']
    news_date = row['Дата']

    if ticker not in price_series:
        return None

    prices = price_series[ticker]
    # Преобразуем news_date в Timestamp для сравнения
    news_ts = pd.Timestamp(news_date)

    # Цена на дату новости или последняя известная до неё (если дата выпала на выходной/праздник)
    before = prices.loc[prices.index <= news_ts]
    if before.empty:
        return None
    price_before = before.iloc[-1]

    # Цена следующего торгового дня (строго после даты новости)
    after = prices.loc[prices.index > news_ts]
    if after.empty:
        return None
    price_after = after.iloc[0]

    # Процентное изменение к следующему дню
    change_pct = (price_after - price_before) / price_before * 100
    return change_pct


# Применяем построчно
news_df['Изменение цены %'] = news_df.apply(compute_price_change, axis=1)

# ------------------------------
# 5. Сохранение результата
# ------------------------------
output_file = 'new_data_with_target.xlsx'
news_df.to_excel(output_file, index=False)
print(f"Готово! Целевая переменная добавлена. Файл сохранён: {output_file}")
print(f"Всего строк: {len(news_df)}, из них с рассчитанным изменением: {news_df['Изменение цены %'].notna().sum()}")