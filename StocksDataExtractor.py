import pandas as pd
import requests
from datetime import datetime
import time

def get_daily_candles(ticker, start_date, end_date):
    """
    Получает все дневные свечи для заданного тикера с MOEX ISS API.
    Функция автоматически обрабатывает пагинацию (постраничную выдачу данных).
    """
    all_candles = []  # Список для накопления данных со всех страниц
    current_start = 0  # Смещение, с которого начинать выборку
    page_size = 100  # Количество записей на одной странице (стандарт MOEX)
    is_more_data = True

    print(f"  Начинаю загрузку данных для {ticker}...")

    while is_more_data:
        # Формируем URL для запроса к API, добавляя параметры `start` и `limit`
        url = (
            f"https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/tqbr/securities/{ticker}/candles.json"
            f"?from={start_date}&till={end_date}&interval=24"
            f"&start={current_start}&limit={page_size}"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # Извлекаем данные для текущей страницы
            candles_data = data['history']['data']

            # Если данных на странице нет, значит, мы всё загрузили
            if not candles_data:
                is_more_data = False
                print(f"    Все данные для {ticker} загружены.")
                break

            # Добавляем данные текущей страницы в общий список
            all_candles.extend(candles_data)
            print(f"    Загружена страница {current_start // page_size + 1}, получено {len(candles_data)} записей.")

            # Готовимся к загрузке следующей страницы
            current_start += page_size

            # Небольшая пауза, чтобы не превысить лимит запросов (20 запросов в секунду)
            time.sleep(0.1)

        except requests.exceptions.RequestException as e:
            print(f"    Ошибка при запросе к API: {e}")
            return None
        except KeyError as e:
            print(f"    Не удалось найти данные в ответе. Возможно, тикер '{ticker}' не найден. Ошибка: {e}")
            return None

    # После сбора всех страниц создаем DataFrame
    if not all_candles:
        print(f"    Данные для {ticker} за указанный период не найдены.")
        return None

    # Нам нужны колонки из первого (любого) ответа. Возьмем их из последнего запроса.
    candles_columns = data['history']['columns']
    df = pd.DataFrame(all_candles, columns=candles_columns)

    # Преобразуем колонку с датой в формат datetime
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])

    # Устанавливаем дату в качестве индекса
    df.set_index('TRADEDATE', inplace=True)

    print(f"    Готово! Загружено {len(df)} записей для {ticker}.")
    return df


if __name__ == "__main__":
    # Параметры дат (можно изменить при необходимости)
    start = "2022-01-01"
    end = "2024-12-31"

    # Чтение списка тикеров из Excel-файла (первый столбец, без заголовка)
    tickers_file = "tickers_clean.xlsx"
    try:
        df_tickers = pd.read_excel(tickers_file, header=None)  # без заголовка
        # Извлекаем первый столбец, отбрасываем пустые значения, приводим к строке
        tickers = df_tickers.iloc[:, 0].dropna().astype(str).tolist()
        # Убираем дубликаты, сохраняя порядок (опционально)
        unique_tickers = list(dict.fromkeys(tickers))
        print(f"Загружено {len(tickers)} тикеров, уникальных: {len(unique_tickers)}")
    except Exception as e:
        print(f"Ошибка при чтении файла {tickers_file}: {e}")
        exit(1)

    # Папка для сохранения результатов (можно создать, если её нет)
    import os
    output_dir = "candles_data"
    os.makedirs(output_dir, exist_ok=True)

    # Проходим по каждому уникальному тикеру
    for i, ticker in enumerate(unique_tickers, 1):
        print(f"\n[{i}/{len(unique_tickers)}] Обработка тикера: {ticker}")
        df = get_daily_candles(ticker, start, end)

        if df is not None:
            # Формируем имя файла: тикер_даты.xlsx
            excel_filename = f"{ticker}_candles_{start}_{end}.xlsx"
            filepath = os.path.join(output_dir, excel_filename)
            # Сбрасываем индекс, чтобы дата стала обычной колонкой
            df_reset = df.reset_index()
            df_reset.to_excel(filepath, sheet_name="Daily candles", index=False)
            print(f"    Данные сохранены в файл: {filepath}")
        else:
            print(f"    Не удалось загрузить данные для {ticker}")

        # Небольшая пауза между разными тикерами, чтобы не перегружать API
        time.sleep(0.5)

    print("\nВсе тикеры обработаны.")