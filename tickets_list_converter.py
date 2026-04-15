import pandas as pd
import re

# Загружаем файл Excel (предполагается, что данные в первом столбце, без заголовка)
input_file = 'tickers_list.xlsx'
output_file = 'tickers_clean.xlsx'

# Читаем файл: header=None означает, что в первой строке нет заголовка
df = pd.read_excel(input_file, header=None)

# Берём первый столбец (индекс 0)
column_data = df[0]

# Список для всех тикетов
all_tickers = []

# Проходим по каждой ячейке
for value in column_data:
    # Пропускаем пустые значения (None, NaN) и строки, состоящие только из пробелов
    if pd.isna(value):
        continue
    # Преобразуем в строку и удаляем лишние пробелы по краям
    cell_str = str(value).strip()
    if not cell_str:
        continue

    # Если в ячейке есть запятая, разбиваем на части
    if ',' in cell_str:
        parts = cell_str.split(',')
        for part in parts:
            ticker = part.strip()
            if ticker:  # игнорируем пустые части (например, при двух запятых подряд)
                all_tickers.append(ticker)
    else:
        # Иначе добавляем всю ячейку как один тикет
        all_tickers.append(cell_str)

# Создаём DataFrame из полученного списка
result_df = pd.DataFrame(all_tickers, columns=['Ticker'])

# Сохраняем в новый Excel-файл (без индекса)
result_df.to_excel(output_file, index=False)

print(f"Обработка завершена. Найдено тикетов: {len(all_tickers)}")
print(f"Результат сохранён в {output_file}")