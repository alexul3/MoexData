import pandas as pd
import numpy as np

# Загрузка данных из исходного файла
input_file = 'new_data_with_target.xlsx'   # имя исходного файла
output_file = 'filtered.xlsx'  # имя нового файла

# Читаем Excel-файл (предполагается, что данные на листе Sheet1)
df = pd.read_excel(input_file, sheet_name='Sheet1')

# Отбираем строки, где столбец 'Изменение цены %' не является пустым (NaN)
# и не равен пустой строке (на случай, если там записана пустая строка)
filtered_df = df[df['Изменение цены %'].notna()]

# Также можно дополнительно исключить строки, где в столбце 'Причина пропуска'
# указаны причины отсутствия цены (это не обязательно, т.к. notna уже отфильтрует)
# filtered_df = filtered_df[filtered_df['Причина пропуска'].isin(['ok', np.nan])]

# Сохраняем отфильтрованные данные в новый Excel-файл
filtered_df.to_excel(output_file, index=False)

print(f"Сохранено {len(filtered_df)} строк в файл {output_file}")