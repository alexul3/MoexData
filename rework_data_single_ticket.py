import pandas as pd
import ast

def parse_tickers(ticker_str):
    """
    Преобразует ячейку с тикерами в список отдельных тикеров.
    Обрабатывает строки-списки Python (на случай, если записано как ["SBER", "GAZP"]),
    а также строки с разделителями: запятая или точка с запятой.
    """
    if pd.isna(ticker_str) or ticker_str == '':
        return []

    s = str(ticker_str).strip()

    # Попытка интерпретировать как Python-список
    if s.startswith('[') and s.endswith(']'):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass

    # Разделители: запятая или точка с запятой
    if ',' in s:
        items = [x.strip() for x in s.split(',') if x.strip()]
        if items:
            return items
    if ';' in s:
        items = [x.strip() for x in s.split(';') if x.strip()]
        if items:
            return items

    # Одиночный тикер без разделителей
    return [s] if s else []

# 1. Загружаем данные
df = pd.read_excel('rework_data.xlsx')

# 2. Явно указываем столбец с тикерами (шестой столбец, индекс F)
TICKER_COL = 'Тикеты комбинированная модель'
if TICKER_COL not in df.columns:
    raise ValueError(f"Столбец '{TICKER_COL}' не найден в файле. Проверьте названия столбцов.")

print(f"Используем столбец с тикерами: '{TICKER_COL}'")

# 3. Преобразуем тикеры в списки и разворачиваем датафрейм
df['tickers_list'] = df[TICKER_COL].apply(parse_tickers)
df_exploded = df.explode('tickers_list', ignore_index=False)

# 4. Удаляем строки, где после разворачивания тикер остался пустым (если исходно не было тикеров)
df_exploded = df_exploded.dropna(subset=['tickers_list'])

# 5. Заменяем исходный столбец с тикерами одиночным значением
df_exploded[TICKER_COL] = df_exploded['tickers_list']
df_exploded = df_exploded.drop(columns=['tickers_list'])

# 6. Обработка даты – оставляем только дату (без времени)
#    Если в столбце date уже чистые даты, ничего не сломается.
df_exploded['date'] = pd.to_datetime(df_exploded['date'], errors='coerce').dt.strftime('%Y-%m-%d')

# 7. Замена неопределённого времени на полдень (12:00:00)
#    Учитываем возможные варианты написания: "not stated", "Not Stated" и т.п.
df_exploded['time'] = df_exploded['time'].fillna('not stated')  # на случай NaN
mask = df_exploded['time'].str.lower().str.strip() == 'not stated'
df_exploded.loc[mask, 'time'] = '12:00:00'

# 8. Сбрасываем индекс для аккуратного вида
df_exploded = df_exploded.reset_index(drop=True)

# 9. Сохраняем результат
output_file = 'rework_data_expanded.xlsx'
df_exploded.to_excel(output_file, index=False)

print(f'Готово! Исходных строк: {len(df)}, после разделения: {len(df_exploded)}.')
print(f'Результат сохранён в файл: {output_file}')