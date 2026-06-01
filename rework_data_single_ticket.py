import pandas as pd
import ast

def parse_tickers(ticker_str):
    """
    Преобразует ячейку с тикерами в список отдельных тикеров.
    Обрабатывает строки-списки Python (например, "['SBER', 'GAZP']"),
    а также строки с разделителями: запятая или точка с запятой.
    """
    if pd.isna(ticker_str) or ticker_str == '':
        return []

    s = str(ticker_str).strip()

    # Попытка интерпретировать как Python-список (начинается с [ и заканчивается ])
    if s.startswith('[') and s.endswith(']'):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                # Оставляем только непустые тикеры
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass

    # Разделители: сначала запятая, затем точка с запятой
    if ',' in s:
        items = [x.strip() for x in s.split(',') if x.strip()]
        if items:
            return items
    if ';' in s:
        items = [x.strip() for x in s.split(';') if x.strip()]
        if items:
            return items

    # Если разделителей нет, считаем, что это одиночный тикер
    return [s] if s else []

# 1. Загружаем данные
df = pd.read_excel('rework_data.xlsx')

# 2. Определяем колонку с тикерами (возможные названия)
TICKER_COL = None
for col in df.columns:
    if 'тикер' in col.lower():
        TICKER_COL = col
        break

if TICKER_COL is None:
    raise ValueError(
        "Не удалось найти колонку с тикерами. "
        "Проверьте названия столбцов или задайте имя вручную (TICKER_COL)."
    )
print(f"Найдена колонка с тикерами: '{TICKER_COL}'")

# 3. Преобразуем каждую ячейку в список тикеров
df['tickers_list'] = df[TICKER_COL].apply(parse_tickers)

# 4. Разворачиваем строки: одна строка на каждый тикер
df_exploded = df.explode('tickers_list', ignore_index=False)

# 5. Удаляем строки, где после развёртки тикер остался пустым (если исходно не было тикеров)
df_exploded = df_exploded.dropna(subset=['tickers_list'])

# 6. Заменяем исходную колонку с тикерами одиночным значением
df_exploded[TICKER_COL] = df_exploded['tickers_list']
df_exploded = df_exploded.drop(columns=['tickers_list'])

# 7. Сбрасываем индекс
df_exploded = df_exploded.reset_index(drop=True)

# 8. Сохраняем результат
output_file = 'rework_data_expanded.xlsx'
df_exploded.to_excel(output_file, index=False)

print(f'Готово! Исходных строк: {len(df)}, после разделения: {len(df_exploded)}.')
print(f'Результат сохранён в файл: {output_file}')