import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os


def fetch_moex_history(ticker, start_date, end_date, market='shares', board='TQBR'):
    """
    Получает исторические данные по тикеру с MOEX ISS.

    Параметры:
        ticker (str): тикер акции (например, 'SBER')
        start_date (str): дата начала в формате 'YYYY-MM-DD'
        end_date (str): дата окончания в формате 'YYYY-MM-DD'
        market (str): рынок ('shares' - акции)
        board (str): режим торгов ('TQBR' - основной)

    Возвращает:
        pandas.DataFrame: таблица с историей торгов (OHLCV)
    """
    base_url = f"https://iss.moex.com/iss/history/engines/stock/markets/{market}/boards/{board}/securities/{ticker}.json"

    all_data = []
    start = 0
    limit = 100  # максимальное число записей на странице

    while True:
        params = {
            'from': start_date,
            'till': end_date,
            'start': start,
            'limit': limit
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе: {e}")
            break

        # Извлекаем блок с историческими данными
        history_block = data.get('history', {})
        columns = history_block.get('columns', [])
        rows = history_block.get('data', [])

        if not rows:
            break  # данных больше нет

        # Преобразуем в DataFrame
        df_page = pd.DataFrame(rows, columns=columns)
        all_data.append(df_page)

        # Проверяем, есть ли ещё страницы (используем информацию из курсора)
        cursor = data.get('history.cursor', {})
        cursor_data = cursor.get('data', [])
        if cursor_data:
            total = cursor_data[0][0]  # общее количество записей
            if start + len(rows) >= total:
                break
        else:
            # Если курсора нет, проверяем по количеству строк на странице
            if len(rows) < limit:
                break

        start += limit
        time.sleep(0.5)  # небольшая задержка, чтобы не нагружать сервер

    if not all_data:
        print("Данные не получены. Проверьте тикер и даты.")
        return pd.DataFrame()

    # Объединяем все страницы
    df = pd.concat(all_data, ignore_index=True)

    # Преобразуем типы данных
    # Столбцы: TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME и др.
    date_col = 'TRADEDATE'
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(date_col, inplace=True)

    # Оставляем только нужные столбцы (можно расширить при необходимости)
    cols_to_keep = ['TRADEDATE', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME']
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    df = df[existing_cols]

    # Удаляем строки с отсутствующими ценами (если торги не проводились)
    df.dropna(subset=['OPEN', 'CLOSE'], inplace=True)

    return df


def save_to_csv(df, filename):
    """Сохраняет DataFrame в CSV."""
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"Данные сохранены в {filename}")


# Пример использования
if __name__ == "__main__":
    # Задаём параметры
    ticker = 'SBER'  # Сбербанк
    start = '2024-01-01'
    end = '2024-12-31'
    output_file = f'{ticker}_daily_{start}_to_{end}.csv'

    # Получаем данные
    print(f"Загрузка данных для {ticker}...")
    df = fetch_moex_history(ticker, start, end)

    if not df.empty:
        print(f"Загружено {len(df)} записей.")
        print(df.head())
        save_to_csv(df, output_file)
    else:
        print("Не удалось получить данные.")