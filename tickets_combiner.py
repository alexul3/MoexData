import pandas as pd
import numpy as np

# Загрузка Excel файла (предполагается, что файл называется 'tickets_combo.xlsx')
# Если файл имеет другое расширение (.xls), замените engine='openpyxl' на engine='xlrd'
# Предполагаем, что в файле нет заголовков, данные начинаются с первой строки
df = pd.read_excel('tickets_combo.xlsx', header=None, usecols='A:B', engine='openpyxl')


# Функция для объединения тикетов из двух ячеек
def combine_tickets(row):
    # Получаем значения из столбцов A (индекс 0) и B (индекс 1)
    val_a = row[0]
    val_b = row[1]

    # Обработка пустых/NaN значений
    if pd.isna(val_a):
        val_a = ''
    if pd.isna(val_b):
        val_b = ''

    # Разбиваем строки на списки тикетов, удаляем лишние пробелы
    tickets_a = [t.strip() for t in str(val_a).split(',') if t.strip()]
    tickets_b = [t.strip() for t in str(val_b).split(',') if t.strip()]

    # Объединяем списки, сохраняя порядок и удаляя дубликаты
    # Используем dict.fromkeys для сохранения порядка первого вхождения
    combined = list(dict.fromkeys(tickets_a + tickets_b))

    # Соединяем обратно в строку через запятую с пробелом
    return ', '.join(combined)


# Применяем функцию к каждой строке
df['Combined'] = df.apply(combine_tickets, axis=1)

# Сохраняем результат в новый Excel файл (только столбец с объединёнными тикетами)
result_df = df[['Combined']]
result_df.to_excel('tickets_combined.xlsx', index=False, header=False, engine='openpyxl')

print("Готово! Результат сохранён в файл 'tickets_combined.xlsx'")