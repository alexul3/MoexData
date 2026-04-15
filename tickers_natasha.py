import re
import pandas as pd
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)
from fuzzywuzzy import fuzz

# ------------------------------------------------------------
# 1. Инициализация Natasha (NER для русского языка)
# ------------------------------------------------------------
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

# ------------------------------------------------------------
# 2. Словарь сопоставления названий компаний с тикерами
# ------------------------------------------------------------
COMPANY_TO_TICKER = {
    # Банки и финансы
    "сбербанк": "SBER",
    "сбер": "SBER",
    "втб": "VTBR",
    "московская биржа": "MOEX",
    "мособлбанк": "MOBB",
    "росбанк": "ROSB",
    "тинькофф": "TCSG",
    "ткс групп": "TCSG",
    "т-банк": "TCSG",

    # Нефть и газ
    "газпром": "GAZP",
    "газпром нефть": "SIBN",
    "лукойл": "LKOH",
    "роснефть": "ROSN",
    "сургутнефтегаз": "SNGS",
    "татнефть": "TATN",
    "новатэк": "NVTK",

    # Металлы и горная добыча
    "норильский никель": "GMKN",
    "норникель": "GMKN",
    "северсталь": "CHMF",
    "нлмк": "NLMK",
    "ммк": "MAGN",
    "русал": "RUAL",
    "полюс": "PLZL",
    "полиметалл": "POLY",
    "алроса": "ALRS",

    # Телекоммуникации и IT
    "мтс": "MTSS",
    "ростелеком": "RTKM",
    "яндекс": "YNDX",
    "вк": "VKCO",
    "озон": "OZON",

    # Розничная торговля
    "магнит": "MGNT",
    "х5": "FIVE",
    "пятёрочка": "FIVE",
    "перекрёсток": "FIVE",
    "лента": "LENT",
    "мвидео": "MVID",
    "детский мир": "DSKY",

    # Электроэнергетика
    "интер рао": "IRAO",
    "русгидро": "HYDR",
    "юнипро": "UPRO",
    "огк-2": "OGKB",
    "мрск": "MRKP",

    # Транспорт
    "аэрофлот": "AFLT",
    "совкомфлот": "FLOT",
    "нмтп": "NMTP",
    "глобалтранс": "GLTR",

    # Химия и удобрения
    "фосагро": "PHOR",
    "акрон": "AKRN",
    "уралкалий": "URKA",

    # Девелопмент и строительство
    "пик": "PIKK",
    "группа лср": "LSRG",
    "эталон": "ETLN",

    # Прочее
    "мечел": "MTLR",
    "система": "AFKS",
    "распадская": "RASP",
    "сэтл групп": "SEGR",
}


# ------------------------------------------------------------
# 3. Функции нормализации и поиска тикера
# ------------------------------------------------------------
def normalize_company_name(name: str) -> str:
    """Нормализация названия компании."""
    name = name.lower().strip()
    name = re.sub(r'\b(пао|оао|зао|ооо|ао|группа|компания|корпорация)\b', '', name)
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def find_ticker(company_name: str, threshold: int = 85) -> str:
    """Поиск тикера по названию компании."""
    norm_name = normalize_company_name(company_name)

    # Точное совпадение
    if norm_name in COMPANY_TO_TICKER:
        return COMPANY_TO_TICKER[norm_name]

    # Поиск по подстроке
    for key in COMPANY_TO_TICKER:
        if key in norm_name or norm_name in key:
            return COMPANY_TO_TICKER[key]

    # Нечёткое сравнение
    best_match = None
    best_score = 0
    for key in COMPANY_TO_TICKER:
        score = fuzz.ratio(norm_name, key)
        if score > best_score:
            best_score = score
            best_match = key

    if best_score >= threshold:
        return COMPANY_TO_TICKER[best_match]

    return ""


def extract_tickers_from_text(text: str) -> str:
    """
    Извлекает тикеры из текста новости.
    Возвращает строку с тикерами через запятую (или пустую строку).
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    tickers = set()
    for span in doc.spans:
        if span.type == "ORG":
            ticker = find_ticker(span.text)
            if ticker:
                tickers.add(ticker)

    return ", ".join(sorted(tickers))


# ------------------------------------------------------------
# 4. Основная функция обработки Excel-файла
# ------------------------------------------------------------
def process_excel_file(input_file: str = "news.xlsx",
                       output_file: str = "news_with_tickers (1).xlsx",
                       text_column: int = 0) -> None:
    """
    Читает Excel-файл, обрабатывает первый столбец с новостями,
    записывает результат в соседний столбец и сохраняет файл.

    :param input_file: путь к исходному файлу
    :param output_file: путь для сохранения результата
    :param text_column: индекс столбца с текстом (0 = первый)
    """
    # Чтение Excel
    try:
        df = pd.read_excel(input_file, header=None)  # без заголовков
    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_file}' не найден.")
        return

    # Определяем столбец с текстом (первый по умолчанию)
    if df.shape[1] <= text_column:
        print(f"Ошибка: В файле только {df.shape[1]} столбцов, а указан индекс {text_column}.")
        return

    text_series = df.iloc[:, text_column]

    # Создаём список для результатов
    ticker_column = []
    total = len(text_series)

    print(f"Начинаем обработку {total} новостей...")

    for idx, text in enumerate(text_series, 1):
        if idx % 50 == 0:
            print(f"Обработано {idx}/{total}...")

        tickers_str = extract_tickers_from_text(text)
        ticker_column.append(tickers_str)

    # Добавляем столбец с тикерами справа от исходного
    df.insert(text_column + 1, f"Tickers", ticker_column)

    # Сохраняем результат
    df.to_excel(output_file, index=False, header=False)
    print(f"Готово! Результат сохранён в '{output_file}'.")


# ------------------------------------------------------------
# 5. Запуск
# ------------------------------------------------------------
if __name__ == "__main__":
    # Укажите имя входного файла
    process_excel_file("news.xlsx", "news_with_tickers.xlsx")