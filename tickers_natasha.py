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
import spacy
from rapidfuzz import fuzz

# ------------------------------------------------------------
# 1. Инициализация Natasha и SpaCy
# ------------------------------------------------------------
# Загружаем модель SpaCy один раз для эффективности
nlp_spacy = spacy.load("ru_core_news_lg")

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

# ------------------------------------------------------------
# 2. Расширенный словарь: топ-100 компаний и отраслевые группы
# ------------------------------------------------------------
COMPANY_TO_TICKER = {
    # Банки и финансы (по данным МосБиржи, доля сектора 27%)
    "сбербанк": "SBER",
    "сбер": "SBER",
    "втб": "VTBR",
    "московская биржа": "MOEX",
    "мосбиржа": "MOEX",
    "тинькофф": "TCSG",
    "т-технологии": "TCSG",
    "ткс групп": "TCSG",
    "т-банк": "TCSG",
    "росбанк": "ROSB",
    "мособлбанк": "MOBB",
    "банк санкт-петербург": "BSPB",

    # Нефть и газ (вес сектора в индексе МосБиржи ~40%)
    "газпром": "GAZP",
    "газпром нефть": "SIBN",
    "лукойл": "LKOH",
    "роснефть": "ROSN",
    "сургутнефтегаз": "SNGS",
    "татнефть": "TATN",
    "новатэк": "NVTK",
    "транснефть": "TRNFP",
    "башнефть": "BANE",
    "русснефть": "RUSI",

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
    "мечел": "MTLR",
    "распадская": "RASP",
    "всмпо-ависма": "VSMO",

    # Телекоммуникации и IT
    "мтс": "MTSS",
    "ростелеком": "RTKM",
    "яндекс": "YNDX",
    "вк": "VKCO",
    "озон": "OZON",
    "софтлайн": "SOFL",
    "группа позитив": "POSI",
    "циан": "CIAN",
    "хэдхантер": "HEAD",

    # Розничная торговля
    "магнит": "MGNT",
    "х5": "X5",
    "корпоративный центр икс 5": "X5",
    "пятёрочка": "X5",
    "перекрёсток": "X5",
    "лента": "LENT",
    "мвидео": "MVID",
    "детский мир": "DSKY",
    "fix price": "FIXP",

    # Электроэнергетика
    "интер рао": "IRAO",
    "русгидро": "HYDR",
    "юнипро": "UPRO",
    "огк-2": "OGKB",
    "тгк-1": "TGKA",
    "мрск": "MRKP",
    "фск еэс": "FEES",

    # Транспорт
    "аэрофлот": "AFLT",
    "совкомфлот": "FLOT",
    "нмтп": "NMTP",
    "глобалтранс": "GLTR",
    "дальневосточное морское пароходство": "FESH",

    # Химия и удобрения
    "фосагро": "PHOR",
    "акрон": "AKRN",
    "уралкалий": "URKA",
    "куйбышевазот": "KAZT",

    # Девелопмент и строительство
    "пик": "PIKK",
    "группа лср": "LSRG",
    "эталон": "ETLN",
    "самолет": "SMLT",

    # Прочее
    "система": "AFKS",
    "сэтл групп": "SEGR",
    "абрау-дюрсо": "ABRD",
    "икс 5 ритейл": "FIVE",  # Альтернативный тикер X5 Group
}

# ------------------------------------------------------------
# 3. Отраслевые правила для групп тикеров
# ------------------------------------------------------------
SECTOR_RULES = {
    # Если в новости встречаются ключевые слова, будут присвоены тикеры из группы
    "нефть": {
        "ключевые_слова": ["нефть", "нефтян", "баррель", "brent", "wti", "опек", "opec", "шельф"],
        "тикеры": ["GAZP", "LKOH", "ROSN", "SIBN", "SNGS", "TATN", "NVTK", "TRNFP", "BANE", "RUSI"]
    },
    "газ": {
        "ключевые_слова": ["газ", "газов", "спг", "трубопровод", "газпром"],
        "тикеры": ["GAZP", "NVTK", "SIBN"]
    },
    "банки": {
        "ключевые_слова": ["банк", "банков", "кредит", "ипотека", "депозит", "процентная ставка", "цб",
                           "центральный банк"],
        "тикеры": ["SBER", "VTBR", "TCSG", "MOEX", "ROSB", "MOBB", "BSPB"]
    },
    "металлы": {
        "ключевые_слова": ["металл", "сталь", "никель", "золото", "палладий", "медь", "руда", "горнодобыва"],
        "тикеры": ["GMKN", "CHMF", "NLMK", "MAGN", "RUAL", "PLZL", "POLY", "ALRS", "MTLR", "RASP", "VSMO"]
    },
    "ритейл": {
        "ключевые_слова": ["розничн", "торговл", "магазин", "покупател", "продаж", "ритейл"],
        "тикеры": ["MGNT", "X5", "LENT", "MVID", "DSKY", "FIXP"]
    },
    "телекоммуникации": {
        "ключевые_слова": ["связь", "интернет", "телеком", "мобильн", "оператор связи"],
        "тикеры": ["MTSS", "RTKM", "VKCO"]
    },
    "электроэнергетика": {
        "ключевые_слова": ["электроэнерг", "энергетик", "электричеств", "гэс", "аэс", "тэц"],
        "тикеры": ["IRAO", "HYDR", "UPRO", "OGKB", "TGKA", "MRKP", "FEES"]
    },
    "транспорт": {
        "ключевые_слова": ["авиаперевоз", "авиакомпан", "морск", "порт", "логистик", "транспортн", "пароходство"],
        "тикеры": ["AFLT", "FLOT", "NMTP", "GLTR", "FESH"]
    },
    "химия": {
        "ключевые_слова": ["удобрен", "химическ", "агрохими", "минеральн"],
        "тикеры": ["PHOR", "AKRN", "URKA", "KAZT"]
    },
    "строительство": {
        "ключевые_слова": ["строительн", "девелоп", "недвижимость", "жилье", "застройщик"],
        "тикеры": ["PIKK", "LSRG", "ETLN", "SMLT"]
    },
    "it": {
        "ключевые_слова": ["программн", "разработк", "по ", "цифров", "онлайн", "информационны"],
        "тикеры": ["YNDX", "VKCO", "OZON", "SOFL", "POSI", "CIAN", "HEAD"]
    },
}


# ------------------------------------------------------------
# 4. Функции нормализации и поиска
# ------------------------------------------------------------
def normalize_company_name(name: str) -> str:
    """Нормализация названия компании: приведение к нижнему регистру, удаление ОПФ и лишних символов."""
    name = name.lower().strip()
    name = re.sub(r'\b(пао|оао|зао|ооо|ао|группа|компания|корпорация|мкпао|пко|public joint stock)\b', '', name,
                  flags=re.IGNORECASE)
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def find_ticker_by_company(company_name: str, threshold: int = 80) -> str:
    """
    Поиск тикера по названию компании с использованием нечёткого сопоставления RapidFuzz.
    """
    norm_name = normalize_company_name(company_name)

    # 1. Точное совпадение
    if norm_name in COMPANY_TO_TICKER:
        return COMPANY_TO_TICKER[norm_name]

    # 2. Поиск по подстроке
    for key in COMPANY_TO_TICKER:
        if key in norm_name or norm_name in key:
            return COMPANY_TO_TICKER[key]

    # 3. Нечёткое сравнение (замена fuzzywuzzy на более быстрый RapidFuzz)
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


def find_tickers_by_sector(text: str) -> set[str]:
    """
    Поиск тикеров на основе отраслевых ключевых слов.
    """
    tickers = set()
    text_lower = text.lower()
    for sector, data in SECTOR_RULES.items():
        for keyword in data["ключевые_слова"]:
            if keyword in text_lower:
                tickers.update(data["тикеры"])
                break
    return tickers


def extract_tickers_from_text(text: str) -> str:
    """
    Извлекает тикеры из текста новости, комбинируя NER (Natasha + SpaCy) и отраслевые правила.
    Возвращает строку с уникальными тикерами через запятую.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    all_tickers = set()

    # --- 1. NER с помощью Natasha ---
    doc_natasha = Doc(text)
    doc_natasha.segment(segmenter)
    doc_natasha.tag_morph(morph_tagger)
    doc_natasha.parse_syntax(syntax_parser)
    doc_natasha.tag_ner(ner_tagger)

    for span in doc_natasha.spans:
        if span.type == "ORG":
            ticker = find_ticker_by_company(span.text)
            if ticker:
                all_tickers.add(ticker)

    # --- 2. NER с помощью SpaCy (дополнительный анализ) ---
    doc_spacy = nlp_spacy(text)
    for ent in doc_spacy.ents:
        if ent.label_ == "ORG":
            ticker = find_ticker_by_company(ent.text)
            if ticker:
                all_tickers.add(ticker)

    # --- 3. Применяем отраслевые правила ---
    all_tickers.update(find_tickers_by_sector(text))

    return ", ".join(sorted(all_tickers))


# ------------------------------------------------------------
# 5. Основная функция обработки Excel-файла (без изменений)
# ------------------------------------------------------------
def process_excel_file(input_file: str = "news.xlsx",
                       output_file: str = "news_with_tickers_natasha_spicy.xlsx",
                       text_column: int = 0) -> None:
    """
    Читает Excel-файл, обрабатывает первый столбец с новостями,
    записывает результат в соседний столбец и сохраняет файл.
    """
    try:
        df = pd.read_excel(input_file, header=None)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_file}' не найден.")
        return

    if df.shape[1] <= text_column:
        print(f"Ошибка: В файле только {df.shape[1]} столбцов, а указан индекс {text_column}.")
        return

    text_series = df.iloc[:, text_column]
    ticker_column = []
    total = len(text_series)

    print(f"Начинаем обработку {total} новостей...")

    for idx, text in enumerate(text_series, 1):
        if idx % 50 == 0:
            print(f"Обработано {idx}/{total}...")
        ticker_column.append(extract_tickers_from_text(text))

    df.insert(text_column + 1, f"Tickers", ticker_column)
    df.to_excel(output_file, index=False, header=False)
    print(f"Готово! Результат сохранён в '{output_file}'.")


# ------------------------------------------------------------
# 6. Запуск
# ------------------------------------------------------------
if __name__ == "__main__":
    process_excel_file("news.xlsx", "news_with_tickers_natasha_spicy.xlsx")