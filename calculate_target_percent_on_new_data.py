import pandas as pd
import requests
from apimoex import get_board_history
import datetime
import warnings
import yfinance as yf  # pip install yfinance

warnings.filterwarnings('ignore')

# ------------------------------
# 1. Загрузка датасета новостей
# ------------------------------
news_df = pd.read_excel('rework_data_expanded.xlsx')
news_df = news_df.rename(columns={
    'date': 'Дата',
    'Тикеты комбинированная модель': 'Тикер'
})
news_df['Дата'] = pd.to_datetime(news_df['Дата']).dt.date

def clean_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper()
    if '.' in t:
        t = t.split('.')[0]
    return t

news_df['Ticker'] = news_df['Тикер'].apply(clean_ticker)

# ==============================
# СЛОВАРЬ ПЕРЕИМЕНОВАНИЙ ТИКЕРОВ (MOEX)
# ==============================
TICKER_MAP = {
    'X5': 'FIVE',
    'YNDX': 'YDEX',
    'MAIL': 'VKCO',
}

unique_tickers = sorted(news_df['Ticker'].unique())

# ------------------------------
# 2. Загрузка котировок с MOEX
# ------------------------------
BOARD = 'TQBR'
min_date = news_df['Дата'].min()
max_date = news_df['Дата'].max()
start_date = min_date - datetime.timedelta(days=60)  # увеличили запас
end_date = max_date + datetime.timedelta(days=20)

print(f"MOEX: загружаем котировки с {start_date} по {end_date} для {len(unique_tickers)} тикеров...")

price_series = {}
failed_tickers = []

with requests.Session() as session:
    for original_ticker in unique_tickers:
        request_ticker = TICKER_MAP.get(original_ticker, original_ticker)
        try:
            data = get_board_history(
                session,
                security=request_ticker,
                board=BOARD,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                columns=('TRADEDATE', 'CLOSE', 'BOARDID')
            )
            if data:
                df = pd.DataFrame(data)
                df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
                df = df.set_index('TRADEDATE')['CLOSE'].dropna()
                if not df.empty:
                    price_series[original_ticker] = df
                else:
                    failed_tickers.append(f"{original_ticker} (запрос как {request_ticker}): пустые цены")
            else:
                failed_tickers.append(f"{original_ticker} (запрос как {request_ticker}): пустой ответ MOEX")
        except Exception as e:
            failed_tickers.append(f"{original_ticker} (запрос как {request_ticker}): ошибка {e}")

print(f"MOEX: успешно загружены котировки для {len(price_series)} тикеров.")
if failed_tickers:
    print(f"Не загружены ({len(failed_tickers)}):")
    for f in failed_tickers[:5]:  # покажем только первые 5
        print("  ", f)

# ------------------------------
# 3. Загрузка котировок с Yahoo Finance (FALLBACK)
# ------------------------------
# Для Yahoo Finance тикеры обычно имеют суффикс .ME
# Но можно задать точечные исключения
YAHOO_TICKER_MAP = {
    'X5': 'FIVE.ME',        # FIVE.ME – если торгуется
    'YNDX': 'YDEX.ME',
    'MAIL': 'VKCO.ME',
    # Для остальных добавляем .ME автоматически
}

def get_yahoo_symbol(orig_ticker: str) -> str:
    if orig_ticker in YAHOO_TICKER_MAP:
        return YAHOO_TICKER_MAP[orig_ticker]
    return f"{orig_ticker}.ME"

yahoo_price_series = {}
yahoo_failed = []

print("\nYahoo Finance: загружаем fallback-котировки...")
for orig_ticker in unique_tickers:
    ysym = get_yahoo_symbol(orig_ticker)
    try:
        # Загружаем данные за тот же период с запасом
        df = yf.download(
            ysym,
            start=start_date,
            end=end_date + datetime.timedelta(days=1),  # yfinance включает end невключительно
            progress=False,
            auto_adjust=False  # берём неcкорректированные цены
        )
        if df.empty:
            yahoo_failed.append(f"{orig_ticker} ({ysym}): нет данных")
            continue
        # Используем колонку 'Close' (не 'Adj Close')
        if 'Close' in df.columns:
            series = df['Close']
        else:
            series = df['Adj Close']  # запасной вариант
        series.index = pd.to_datetime(series.index).date  # приводим к date для совместимости
        series = series.dropna()
        if not series.empty:
            yahoo_price_series[orig_ticker] = series
        else:
            yahoo_failed.append(f"{orig_ticker} ({ysym}): все цены NaN")
    except Exception as e:
        yahoo_failed.append(f"{orig_ticker} ({ysym}): ошибка {e}")

print(f"Yahoo: успешно загружены котировки для {len(yahoo_price_series)} тикеров.")
if yahoo_failed:
    print(f"Не загружены из Yahoo ({len(yahoo_failed)}):")
    for f in yahoo_failed[:5]:
        print("  ", f)

# ------------------------------
# 4. Расчёт целевой переменной (с fallback)
# ------------------------------
def compute_price_change(row):
    ticker = row['Ticker']
    news_date = row['Дата']
    news_ts = pd.Timestamp(news_date)

    # 1. Пробуем MOEX
    if ticker in price_series:
        prices = price_series[ticker]
        before = prices.loc[prices.index < news_ts]
        after = prices.loc[prices.index > news_ts]
        if not before.empty and not after.empty:
            price_before = before.iloc[-1]
            price_after = after.iloc[0]
            change_pct = (price_after - price_before) / price_before * 100
            return change_pct, 'ok'

    # 2. Fallback – Yahoo Finance
    if ticker in yahoo_price_series:
        prices = yahoo_price_series[ticker]
        # Индексы в yahoo_price_series уже date, сравнение работает с pd.Timestamp
        before = prices.loc[prices.index < news_date]
        after = prices.loc[prices.index > news_date]
        if not before.empty and not after.empty:
            price_before = before.iloc[-1]
            price_after = after.iloc[0]
            change_pct = (price_after - price_before) / price_before * 100
            return change_pct, 'ok'
        else:
            # Yahoo есть, но не хватает данных – уточняем причину
            if before.empty:
                return None, 'no_price_before'
            else:
                return None, 'no_price_after'

    # 3. Нет данных ни в одном источнике
    if ticker not in price_series and ticker not in yahoo_price_series:
        return None, 'no_ticker_data'
    else:
        # Данные тикера были, но не покрывают нужный диапазон – смотрим, что пропущено
        # (MOEX уже проверен, если мы здесь, значит MOEX не дал оба значения)
        if ticker in price_series:
            prices = price_series[ticker]
            before = prices.loc[prices.index < news_ts]
            after = prices.loc[prices.index > news_ts]
            if before.empty:
                return None, 'no_price_before'
            else:
                return None, 'no_price_after'
        else:
            # Случай, когда MOEX нет, а Yahoo не помог – сообщаем no_ticker_data
            return None, 'no_ticker_data'

results = news_df.apply(compute_price_change, axis=1, result_type='expand')
news_df['Изменение цены %'] = results[0]
news_df['Причина пропуска'] = results[1]

# ------------------------------
# 5. Анализ пропусков
# ------------------------------
print("\nСтатистика заполнения целевой переменной:")
print(f"Всего строк: {len(news_df)}")
print(f"С таргетом: {news_df['Изменение цены %'].notna().sum()}")
print(f"Пропущено: {news_df['Изменение цены %'].isna().sum()}")
print("\nПричины пропусков:")
print(news_df['Причина пропуска'].value_counts())

# ------------------------------
# 6. Сохранение результата
# ------------------------------
output_file = 'new_data_with_target.xlsx'
news_df.to_excel(output_file, index=False)
print(f"\nФайл сохранён: {output_file}")