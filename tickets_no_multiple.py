import pandas as pd
import sys


def remove_duplicates(input_file, output_file):
    """
    Удаляет дубликаты из первого столбца Excel-файла и сохраняет результат.

    Параметры:
    input_file (str): путь к исходному Excel-файлу
    output_file (str): путь для сохранения результата
    """
    try:
        # Читаем файл без заголовка (первая строка — данные)
        df = pd.read_excel(input_file, header=None)

        # Берём первый столбец
        first_col = df.iloc[:, 0]

        # Удаляем пустые значения и дубликаты
        unique_values = first_col.dropna().drop_duplicates()

        # Создаём новый DataFrame
        result_df = pd.DataFrame(unique_values)

        # Сохраняем в Excel без индекса и заголовка
        result_df.to_excel(output_file, index=False, header=False)

        print(f"Готово! Найдено {len(unique_values)} уникальных тикетов.")
        print(f"Результат сохранён в '{output_file}'")

    except FileNotFoundError:
        print(f"Ошибка: файл '{input_file}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    # Получаем имена файлов (из аргументов командной строки или вводом)
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = "tickers_clean.xlsx"
        output_file = "unique.xlsx"

    remove_duplicates(input_file, output_file)