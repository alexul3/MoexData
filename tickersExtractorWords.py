import re
import pandas as pd
from fuzzywuzzy import fuzz
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsNERTagger,
    Doc
)

# ---------------------------
# 1. Загрузка справочников
# ---------------------------

COMPANY_TICKER = {
    # --- Energy & Oil & Gas ---
    "Газпром": "GAZP",
    "Лукойл": "LKOH",
    "Роснефть": "ROSN",
    "Сургутнефтегаз": "SNGS",
    "Новатэк": "NVTK",  # Added [citation:1][citation:2][citation:9]
    "Газпром нефть": "SIBN",  # Added [citation:1][citation:2][citation:5]
    "Татнефть": "TATN",  # Added [citation:1][citation:2][citation:9]
    "Русснефть": "RNFT",  # Added [citation:2][citation:5]
    "Башнефть": "BANE",  # Added [citation:1][citation:2]
    "НКНХ (Нижнекамскнефтехим)": "NKNC",  # Added [citation:1][citation:2]
    "Транснефть": "TRNF",  # Added [citation:5][citation:8]
    "Славнефть-ЯНОС": "JNOS",  # Added [citation:2]
    "Варьеганнефтегаз": "VJGZ",  # Added [citation:2]

    # --- Financials & Banking ---
    "Сбербанк": "SBER",
    "ВТБ": "VTBR",
    "Т-Технологии (Тинькофф)": "T",  # Added [citation:1][citation:9][citation:10]
    "Московская биржа": "MOEX",  # Added [citation:1][citation:2][citation:9]
    "Совкомбанк": "SVCB",  # Added [citation:2][citation:10]
    "МКБ (Московский кредитный банк)": "CBOM",  # Added [citation:1][citation:2][citation:10]
    "Банк Санкт-Петербург": "BSPB",  # Added [citation:2][citation:5][citation:10]
    "АФК Система": "AFKS",  # Added [citation:1][citation:2][citation:5]
    "Росгосстрах": "RGSS",  # Added [citation:1][citation:2]
    "Ренессанс Страхование": "RENI",  # Added [citation:2][citation:4][citation:10]
    "SFI": "SFIN",  # Added [citation:2][citation:5][citation:10]
    "Европлан": "LEAS",  # Added [citation:2][citation:10]
    "МТС Банк": "MBNK",  # Added [citation:2][citation:10]
    "СПБ Биржа": "SPBE",  # Added [citation:2][citation:10]
    "ДОМ.РФ": "DOMRF",  # Added [citation:10]
    "Уралсиб": "USBN",  # Added [citation:2][citation:5]
    "Приморье Банк": "PRMB",  # Added [citation:2][citation:6]
    "Авангард (банк)": "AVAN",  # Added [citation:1][citation:2]

    # --- Metals & Mining ---
    "Норникель": "GMKN",
    "Северсталь": "CHMF",  # Added [citation:1][citation:2][citation:5]
    "НЛМК": "NLMK",  # Added [citation:1][citation:2][citation:5]
    "ММК": "MAGN",  # Added [citation:1][citation:2][citation:5]
    "АЛРОСА": "ALRS",  # Added [citation:1][citation:2][citation:5]
    "Полюс": "PLZL",  # Added [citation:1][citation:2][citation:9]
    "РУСАЛ": "RUAL",  # Added [citation:1][citation:2][citation:5]
    "Мечел": "MTLR",  # Added [citation:1][citation:2][citation:5]
    "ВСМПО-АВИСМА": "VSMO",  # Added [citation:1][citation:2][citation:5]
    "Распадская": "RASP",  # Added [citation:1][citation:2][citation:5]
    "Селигдар": "SELG",  # Added [citation:1][citation:2][citation:5]
    "Южуралзолото (ЮГК)": "UGLD",  # Added [citation:2][citation:4][citation:7]
    "Полиметалл": "POLY",  # Added [citation:5][citation:8]
    "Лензолото": "LNZL",  # Added [citation:2][citation:5]
    "Бурятзолото": "BRZL",  # Added [citation:2][citation:5]
    "Русолово": "ROLO",  # Added [citation:2][citation:5]
    "Коршуновский ГОК": "KOGK",  # Added [citation:5][citation:6]
    "ЧМК": "CHMK",  # Added [citation:2][citation:5]
    "Южный Кузбасс": "UKUZ",  # Added [citation:1][citation:5]

    # --- Retail & Consumer ---
    "Магнит": "MGNT",
    "X5 Group (Пятёрочка)": "X5",  # Added [citation:2][citation:5][citation:9]
    "Лента": "LENT",  # Added [citation:1][citation:2][citation:5]
    "М.Видео": "MVID",  # Added [citation:2][citation:5]
    "Окей": "OKEY",  # Added [citation:2][citation:5]
    "Fix Price": "FIXP",  # Added [citation:2]
    "Аптеки 36.6": "APTK",  # Added [citation:1][citation:2]
    "Черкизово": "GCHE",  # Added [citation:1][citation:2][citation:5]
    "Абрау-Дюрсо": "ABRD",  # Added [citation:2][citation:5][citation:6]
    "Белуга Групп (НоваБев)": "BELU",  # Added [citation:2][citation:5]
    "Русагро": "RAGR",  # Added [citation:2][citation:5]
    "Henderson": "HNFG",  # Added [citation:2]
    "Группа ЛСР": "LSRG",  # Added [citation:1][citation:2][citation:5]
    "Самолет": "SMLT",  # Added [citation:1][citation:2][citation:5]
    "ПИК (Группа ПИК)": "PIKK",  # Added [citation:1][citation:2]
    "Эталон": "ETLN",  # Added [citation:2][citation:5]
    "Вуш (Whoosh)": "WUSH",  # Added [citation:2]
    "Делимобиль": "DELI",  # Added [citation:2][citation:5]

    # --- Technology & Telecom ---
    "Яндекс": "YDEX",  # Note: Current ticker is YDEX after restructuring [citation:2][citation:4][citation:5]
    "VK": "VKCO",  # Added [citation:2][citation:5]
    "МТС": "MTSS",  # Added [citation:1][citation:2][citation:5]
    "Ростелеком": "RTKM",  # Added [citation:1][citation:2][citation:5]
    "OZON": "OZON",  # Added [citation:2][citation:5][citation:9]
    "ЦИАН": "CNRU",  # Added [citation:2][citation:5]
    "HeadHunter": "HEAD",  # Added [citation:2][citation:5]
    "Группа Позитив": "POSI",  # Added [citation:1][citation:2][citation:5]
    "Softline": "SOFL",  # Added [citation:2][citation:5]
    "Астра (Группа Астра)": "ASTR",  # Added [citation:2][citation:4]
    "Диасофт": "DIAS",  # Added [citation:2][citation:5]
    "ВсеИнструменты.ру": "VSEH",  # Added [citation:2][citation:4]
    "МГТС": "MGTS",  # Added [citation:1][citation:2][citation:5]
    "Центральный телеграф": "CNTL",  # Added [citation:2][citation:5]
    "Башинформсвязь": "BISV",  # Added [citation:5][citation:6]
    "Таттелеком": "TTLK",  # Added [citation:2][citation:4][citation:5]
    "Аренадата": "DATA",  # Added [citation:2]
    "IVA Technologies": "IVAT",  # Added [citation:2][citation:4]

    # --- Industrial & Transport ---
    "Аэрофлот": "AFLT",
    "Совкомфлот": "FLOT",  # Added [citation:1][citation:2][citation:5]
    "ЮТэйр": "UTAR",  # Added [citation:2][citation:5]
    "КАМАЗ": "KMAZ",  # Added [citation:1][citation:2][citation:5]
    "Соллерс": "SVAV",  # Added [citation:2][citation:5]
    "ОАК (Объединенная авиастроительная корпорация)": "UNAC",  # Added [citation:1][citation:2][citation:5]
    "Яковлев (Иркут)": "IRKT",  # Added [citation:1][citation:2][citation:5]
    "НМТП (Новороссийский морской торговый порт)": "NMTP",  # Added [citation:1][citation:2]
    "FESCO (ДВМП)": "FESH",  # Added [citation:1][citation:2][citation:5]
    "КамАЗ": "KMAZ",  # Added [citation:1][citation:2]
    "ТМК": "TRMK",  # Added [citation:1][citation:2][citation:5]
    "ОМЗ": "OMZZ",  # Added [citation:5][citation:8]
    "РКК Энергия": "RKKE",  # Added [citation:2][citation:4][citation:5]
    "Мостотрест": "MSTT",  # Added [citation:2][citation:5]

    # --- Utilities & Energy Grid ---
    "РусГидро": "HYDR",  # Added [citation:1][citation:2][citation:4]
    "Интер РАО": "IRAO",  # Added [citation:1][citation:2][citation:5]
    "Юнипро": "UPRO",  # Added [citation:1][citation:2][citation:5]
    "Россети (ФСК ЕЭС)": "FEES",  # Added [citation:1][citation:2][citation:5]
    "Мосэнерго": "MSNG",  # Added [citation:1][citation:2][citation:5]
    "ТГК-1": "TGKA",  # Added [citation:2][citation:5]
    "ТГК-2": "TGKB",  # Added [citation:2][citation:5]
    "ОГК-2": "OGKB",  # Added [citation:1][citation:2][citation:5]
    "ЭЛ5-Энерго": "ELFV",  # Added [citation:2][citation:5]
    "Квадра": "TGKD",  # Added [citation:5][citation:8]
    "Россети Ленэнерго": "LSNG",  # Added [citation:1][citation:2][citation:5]
    "Россети МР (МОЭСК)": "MSRS",  # Added [citation:1][citation:2][citation:5]
    "Россети Волга": "MRKV",  # Added [citation:2][citation:3][citation:5]
    "Россети Центр": "MRKC",  # Added [citation:2][citation:3][citation:5]
    "Россети Северо-Запад": "MRKZ",  # Added [citation:2][citation:3][citation:5]
    "Россети Центр и Приволжье": "MRKP",  # Added [citation:2][citation:3][citation:5]
    "Россети Урал": "MRKU",  # Added [citation:2][citation:5]
    "Россети Сибирь": "MRKS",  # Added [citation:2][citation:5]
    "Россети Юг": "MRKY",  # Added [citation:2][citation:5]
    "ТГК-14": "TGKN",  # Added [citation:2][citation:3][citation:4]

    # --- Chemicals & Fertilizers ---
    "Фосагро": "PHOR",  # Added [citation:1][citation:2][citation:5]
    "Акрон": "AKRN",  # Added [citation:1][citation:2][citation:3]
    "Казаньоргсинтез": "KZOS",  # Added [citation:1][citation:2][citation:5]
    "КуйбышевАзот": "KAZT",  # Added [citation:1][citation:2][citation:4]
    "Нижнекамскнефтехим": "NKNC",  # Added [citation:1][citation:2]
    "Уралкалий": "URKA",  # Often mentioned but not in current top lists, common knowledge
}

# Словарь секторов: ключевое слово (или фраза) -> список тикеров компаний этого сектора
SECTOR_TICKERS = {
    # Нефтегазовый сектор
    "нефтедобывающие компании": ["GAZP", "LKOH", "ROSN", "SNGS", "TATN", "NVTK"],
    "нефтесервис и переработка": ["SIBN", "BANE", "TATNP"],  # добавлены "дочки" и переработка

    # Банковский сектор
    "банковский сектор": ["SBER", "VTBR", "TCSG"],  # TCSG – TCS Group (Тинькофф)

    # IT и интернет-компании
    "IT сектор": ["YNDX", "VKCO", "OZON", "ASTR", "SOFT", "CIAN"],

    # Ритейл (розничная торговля)
    "ритейл": ["MGNT", "LENT", "FIXP", "MVID", "OZON"],  # OZON также относится к e-commerce

    # Металлургия и горнодобыча
    "металлургия и горнодобыча": [
        "NLMK", "CHMF", "MAGN", "MTLR", "RUAL", "ALRS", "GMKN", "POLY"
    ],

    # Телекоммуникации
    "телекоммуникации": ["MTSS", "RTKM", "VEON-RX"],

    # Электроэнергетика
    "электроэнергетика": ["HYDR", "IRAO", "UPRO", "TGKA", "FESH"],

    # Химическая промышленность
    "химическая промышленность": ["AKRN", "PHOR", "NKNK", "URAL", "KZOS"],

    # Машиностроение и авиастроение
    "машиностроение": ["KMAZ", "VSMO", "IRKT", "UAZ"],

    # Транспорт и логистика
    "транспорт и логистика": ["AFLT", "NMTP", "FESH", "GTLK"],

    # Девелопмент и недвижимость
    "девелопмент и недвижимость": ["PIKK", "LSRG", "SMLT", "ETAL"],

    # Потребительские товары (продукты, алкоголь)
    "потребительские товары": ["BELU", "APTK"],

    # Финансовые услуги (биржа, инвестиционные холдинги)
    "финансовые услуги": ["MOEX", "SFIN"],

    # Фармацевтика
    "фармацевтика": ["PHST", "KRKNP"],  # Pharmstandard, Pharmstandard-Preferred? предпочтительные не включены

    # Строительные материалы и промышленность
    "стройматериалы": ["USBN", "CARM"]  # примеры: Usadba, CarMoney – менее капиталоёмкие
}

# ---------------------------
# 2. Инициализация NER-модели (Natasha)
# ---------------------------
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

# ---------------------------
# 3. Вспомогательные функции
# ---------------------------

def extract_organizations(text: str):
    """
    Извлекает названия организаций из текста с помощью Natasha.
    Возвращает список строк.
    """
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    orgs = []
    for span in doc.spans:
        if span.type == "ORG":
            # Приводим к нижнему регистру и удаляем лишние пробелы
            name = span.text.strip().lower()
            orgs.append(name)
    return orgs

def find_tickers_by_org_names(org_names, threshold=85):
    """
    Для каждого извлечённого названия организации ищет наиболее похожее
    название в справочнике COMPANY_TICKER с помощью нечёткого сравнения.
    Возвращает список тикеров.
    """
    tickers = set()
    for org in org_names:
        best_match = None
        best_score = 0
        for company_name in COMPANY_TICKER.keys():
            score = fuzz.ratio(org, company_name.lower())
            if score > best_score:
                best_score = score
                best_match = company_name
        if best_match and best_score >= threshold:
            tickers.add(COMPANY_TICKER[best_match])
    return list(tickers)

def detect_sector(text: str):
    """
    Проверяет текст на наличие ключевых слов секторов.
    Возвращает список тикеров для первого подходящего сектора.
    """
    text_lower = text.lower()
    for keyword, tickers in SECTOR_TICKERS.items():
        if keyword in text_lower:
            return tickers
    return []

def extract_explicit_tickers(text: str):
    """
    Ищет в тексте явно написанные тикеры (например, GAZP, LKOH).
    Возвращает список найденных тикеров.
    """
    # Тикеры обычно состоят из 4-5 заглавных букв, могут быть и другие варианты
    pattern = r'\b[A-Z]{4,5}\b'
    return re.findall(pattern, text)

def get_tickers(news_text: str):
    """
    Основная функция: для текста новости возвращает список тикеров.
    Стратегия:
      1. Ищем явные тикеры (регулярное выражение).
      2. Если не нашли, извлекаем названия организаций (NER) и сопоставляем с COMPANY_TICKER.
      3. Если всё ещё пусто, проверяем принадлежность к сектору.
    """
    # 1. Явные тикеры
    explicit = extract_explicit_tickers(news_text)
    if explicit:
        return list(set(explicit))  # уникальные

    # 2. NER + нечёткое сопоставление
    orgs = extract_organizations(news_text)
    if orgs:
        tickers = find_tickers_by_org_names(orgs)
        if tickers:
            return tickers

    # 3. Секторные ключевые слова
    sector_tickers = detect_sector(news_text)
    if sector_tickers:
        return sector_tickers

    # Ничего не найдено
    return []

# ---------------------------
# 4. Основная обработка Excel-файла
# ---------------------------

def process_excel(file_path, sheet_name=0, text_column='A', result_column='B'):
    """
    Читает Excel-файл, для каждой строки берёт текст из столбца text_column,
    вычисляет тикеры и записывает в столбец result_column.
    Предполагается, что заголовков в файле нет (данные начинаются с A1).
    """
    # Читаем без заголовков, чтобы удобно обращаться по индексам
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Столбцы: A -> индекс 0, B -> индекс 1
    col_a = 0
    col_b = 1

    # Если result_column указан как 'B', то индекс 1
    if result_column.upper() == 'B':
        col_b = 1
    elif result_column.upper() == 'C':
        col_b = 2
    # ... можно добавить другие столбцы

    # Проходим по всем строкам, где есть текст в столбце A
    for idx, row in df.iterrows():
        news_text = str(row[col_a]) if pd.notna(row[col_a]) else ""
        if news_text.strip():
            tickers = get_tickers(news_text)
            # Записываем тикеры в виде строки, разделённой запятыми
            df.at[idx, col_b] = ", ".join(tickers)
        else:
            df.at[idx, col_b] = ""

    # Сохраняем результат в новый файл или перезаписываем
    output_path = file_path.replace(".xlsx", "_with_tickers.xlsx")
    df.to_excel(output_path, index=False, header=False)
    print(f"Обработка завершена. Результат сохранён в {output_path}")

# ---------------------------
# 5. Запуск (если файл выполняется как скрипт)
# ---------------------------
if __name__ == "__main__":
    # Укажите путь к вашему Excel-файлу
    input_file = "news.xlsx"  # измените на свой файл
    process_excel(input_file)