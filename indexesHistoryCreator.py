import pandas as pd
import numpy as np
from moexalgo import Ticker
import ta
from datetime import datetime
import time
import os

# Список тикеров
tickers = [
    'ROSN', 'SBER', 'VTBR', 'LKOH', 'GAZP', 'NVTK', 'T', 'PLZL', 'TATN', 'AFLT',
    'RNFT', 'YNDX', 'GMKN', 'SMLT', 'RUAL', 'PHOR', 'FIVE', 'SIBN', 'SNGS', 'OZON',
    'MTSS'
]

# Период данных
START_DATE = '2022-01-01'
END_DATE = '2024-12-31'

# Пауза между запросами, чтобы не нагружать API
REQUEST_PAUSE = 0.5

def fetch_ticker_data(ticker: str) -> pd.DataFrame:
    """
    Загружает дневные свечи для одного тикера с MOEX через moexalgo.
    Возвращает DataFrame с колонками: open, high, low, close, volume.
    """
    try:
        t = Ticker(ticker)
        candles = t.candles(start=START_DATE, end=END_DATE)
        if candles.empty:
            print(f"⚠️ Нет данных для {ticker}")
            return pd.DataFrame()

        # Приводим колонки к единому виду
        df = candles[['begin', 'open', 'high', 'low', 'close', 'volume']].copy()
        df.rename(columns={'begin': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"❌ Ошибка при загрузке {ticker}: {e}")
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет технические индикаторы для переданного DataFrame.
    Требуются колонки: open, high, low, close.
    """
    if df.empty or len(df) < 50:  # SMA 50 требует минимум 50 дней
        return pd.DataFrame()

    # Returns (дневная доходность)
    df['returns'] = df['close'].pct_change()

    # SMA 20 / SMA 50
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_ratio'] = df['sma_20'] / df['sma_50']

    # RSI 14
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    # Bollinger Bands %B
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ATR (Average True Range)
    df['atr'] = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=14
    ).average_true_range()

    # Удаляем промежуточные колонки, оставляем только нужные
    result = df[[
        'close', 'returns', 'sma_20', 'sma_50', 'sma_ratio',
        'rsi_14', 'bb_upper', 'bb_lower', 'bb_mavg', 'bb_percent_b', 'atr'
    ]].copy()

    return result

def main():
    print(f"🚀 Начало загрузки данных для {len(tickers)} тикеров")
    print(f"📅 Период: {START_DATE} → {END_DATE}")
    print("-" * 50)

    # Создаём Excel writer с движком openpyxl
    output_file = f"technical_indicators_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        successful = 0
        failed = []

        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Обработка {ticker}...")
            df = fetch_ticker_data(ticker)

            if df.empty:
                failed.append(ticker)
                continue

            df_indicators = compute_indicators(df)
            if df_indicators.empty:
                print(f"⚠️ Недостаточно данных для расчёта индикаторов по {ticker}")
                failed.append(ticker)
                continue

            # Сохраняем в отдельный лист Excel
            df_indicators.to_excel(writer, sheet_name=ticker[:31])  # ограничение длины листа
            successful += 1

            # Пауза, чтобы не перегружать сервер MOEX
            time.sleep(REQUEST_PAUSE)

        # Сохраняем сводный лист со списком успешных/провальных тикеров
        summary = pd.DataFrame({
            'Успешно': pd.Series(tickers[:successful]),
            'Ошибка': pd.Series(failed)
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)

    print("-" * 50)
    print(f"✅ Успешно обработано: {successful} тикеров")
    if failed:
        print(f"❌ Не удалось загрузить/обработать: {', '.join(failed)}")
    print(f"💾 Результат сохранён в файл: {output_file}")

if __name__ == "__main__":
    main()