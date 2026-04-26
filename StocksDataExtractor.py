import pandas as pd
import requests
import apimoex
import os
import time
import re
from datetime import timedelta

def safe_filename(ticker):
    return re.sub(r'[\\/*?:"<>|]', '_', ticker)

def get_all_candles(ticker, start_date, end_date, interval=24):
    """
    Получает свечи тикера с MOEX, автоматически разбивая длинный период
    на части по 365 дней (ограничение API для внутридневных интервалов).
    """
    print(f"  Загружаю {ticker} (интервал {interval} мин) ...")
    session = requests.Session()
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    all_frames = []

    # Разбиваем период на куски не длиннее 365 дней
    current_start = start_dt
    while current_start < end_dt:
        chunk_end = min(current_start + timedelta(days=365), end_dt)
        start_str = current_start.strftime('%Y-%m-%d')
        end_str = chunk_end.strftime('%Y-%m-%d')
        print(f"    Запрос периода: {start_str} — {end_str}")
        try:
            candles = apimoex.get_board_candles(
                session,
                security=ticker,
                board='tqbr',
                market='shares',
                engine='stock',
                start=start_str,
                end=end_str,
                interval=interval
            )
        except Exception as e:
            print(f"    Ошибка запроса: {e}")
            return None

        if candles:
            df_chunk = pd.DataFrame(candles)
            all_frames.append(df_chunk)
        else:
            print(f"    Нет данных за {start_str}–{end_str}")
        current_start = chunk_end + timedelta(days=1)
        time.sleep(0.2)  # пауза между частями

    if not all_frames:
        print(f"    Данные для {ticker} полностью отсутствуют.")
        return None

    df = pd.concat(all_frames, ignore_index=True)
    # Приводим дату и делаем индекс
    df['begin'] = pd.to_datetime(df['begin'])
    df.rename(columns={'begin': 'TRADEDATE'}, inplace=True)
    df.set_index('TRADEDATE', inplace=True)
    # Удаляем возможные дубликаты (если запросы пересекаются)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    print(f"    Загружено {len(df)} часовых записей.")
    return df

if __name__ == "__main__":
    start = "2022-01-01"
    end = "2024-12-31"

    # Чтение тикеров
    tickers_file = "unique_ru_tickets.xlsx"
    try:
        df_tickers = pd.read_excel(tickers_file, header=None)
        tickers = df_tickers.iloc[:, 0].dropna().astype(str).tolist()
        unique_tickers = list(dict.fromkeys(tickers))
        print(f"Загружено тикеров: всего {len(tickers)}, уникальных {len(unique_tickers)}")
    except Exception as e:
        print(f"Ошибка чтения файла {tickers_file}: {e}")
        exit(1)

    output_dir = "hourly_candles"   # новая папка для часовых данных
    os.makedirs(output_dir, exist_ok=True)

    # ЗАДАЁМ ЧАСОВОЙ ИНТЕРВАЛ
    INTERVAL_MINUTES = 60   # 60 минут = 1 час

    for i, ticker in enumerate(unique_tickers, 1):
        print(f"\n[{i}/{len(unique_tickers)}] Обработка {ticker}")
        df = get_all_candles(ticker, start, end, interval=INTERVAL_MINUTES)

        if df is not None and not df.empty:
            safe_ticker = safe_filename(ticker)
            filename = f"{safe_ticker}_hourly_{start}_{end}.xlsx"
            filepath = os.path.join(output_dir, filename)
            try:
                df.reset_index().to_excel(filepath, sheet_name="Hourly candles", index=False)
                print(f"    Сохранено: {filepath}")
            except Exception as e:
                print(f"    Ошибка сохранения: {e}")
        else:
            print(f"    Данные для {ticker} не получены.")

        time.sleep(0.5)

    print("\nГотово.")