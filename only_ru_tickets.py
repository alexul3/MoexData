import pandas as pd

FILE_NAME = "unique.xlsx"
SHEET_NAME = 0
COL_NAME = 0  # индекс или название колонки со словами
OUTPUT_FILE = "ru_tickets.xlsx"

# Набор тикеров российских акций (сформирован на основе открытых списков MOEX)
RUSSIAN_TICKERS = {
    "GAZP", "LKOH", "ROSN", "CHMF", "GMKN", "MGNT", "MTSS", "UPRO",
    "VTBR", "NVTK", "MOEX", "NLMK", "MAGN", "ALRS", "AFLT", "RTKM",
    "RTKMP", "HYDR", "TATN", "TATNP", "SNGS", "SNGSP", "SBER",
    "SIBN", "PHOR", "PLZL", "RUAL", "TRNFP", "AFKS", "AGRO",
    "AKRN", "ASTR", "BANE", "BANEP", "BSPB", "CBOM", "ENPG",
    "FEES", "FESH", "FIXP", "FLOT", "HEAD", "IRAO", "IRKT",
    "KMAZ", "LENT", "MSNG", "NMTP", "PIKK", "POSI", "RASP",
    "SVCB", "TRMK", "UGLD", "UNAC", "VSMO", "UWGN", "ABRD",
    "APTK", "AQUA", "BELU", "CIAN", "DATA", "DELI", "DIAS",
    "ELMT", "ETLN", "GCHE", "GEMC", "HNFG", "IVAT", "KAZT",
    "KZOS", "LEAS", "LSNG", "LSRG", "MBNK", "MDMG", "MGTSP",
    "MRKC", "MRKP", "MRKS", "MRKU", "MSRS", "MSTT", "MTLR",
    "MTLRP", "NKHP", "NKNC", "NKNCP", "OGKB", "OZPH", "PRMD",
    "RENI", "RKKE", "RNFT", "SELG", "SFIN", "SGZH", "SMLT",
    "SOFL", "SVAV", "TGKA", "VKCO", "VRSB", "VSEH", "WUSH",
    "YAKG", "CHKZ", "CHMK", "EELT", "ELFV", "EUTR", "HIMCP",
    "IGST", "IGSTP", "INGR", "KBSB", "KOGK", "KRKNP", "KROT",
    "KROTP", "KRSB", "KRSBP", "LMBZ", "LPSB", "MISB", "MISBP",
    "MRKK", "MRKV", "MRKY", "MRKZ", "MVID", "NNSB", "NNSBP",
    "OKEY", "PMSB", "PMSBP", "RGSS", "ROLO", "RTGZ", "RTSB",
    "RTSBP", "RZSB", "SAGO", "SAGOP", "SPBE", "TGKB", "TGKBP",
    "TGKN", "TNSE", "TTLK", "UKUZ", "USBN", "VEON-RX", "VJGZ",
    "VJGZP", "VRSBP", "VSYD", "WTCM", "YRSB", "YRSBP", "ZAYM",
    "OZON", "X5", "YDEX", "TCSG", "HEAD", "FIXP", "CIAN",
    "ASTR", "ELMT", "VSEH", "PRMD", "SGZH", "SOFL", "SVAV",
    "VKCO", "WUSH", "YAKG", "DATA", "DELI", "DIAS", "GEMC",
    "HNFG", "IVAT", "LEAS", "MBNK", "MDMG", "OZPH", "RENI",
    "SFIN", "VSMO", "AGRO", "AKRN", "ALRS", "BANE", "BANEP",
    "BSPB", "CBOM", "ENPG", "FEES", "FESH", "FLOT", "HYDR",
    "IRAO", "IRKT", "KMAZ", "LENT", "MAGN", "MSNG", "NMTP",
    "PIKK", "POSI", "RASP", "RTKM", "RTKMP", "SVCB", "TRMK",
    "UGLD", "UNAC", "UPRO", "UWGN", "VTBR", "ABRD", "APTK",
    "AQUA", "BELU", "CHKZ", "CHMK", "EELT", "ELFV", "EUTR",
    "GCHE", "HIMCP", "IGST", "IGSTP", "INGR", "KAZT", "KBSB",
    "KOGK", "KRKNP", "KROT", "KROTP", "KRSB", "KRSBP", "KZOS",
    "LMBZ", "LPSB", "LSNG", "LSRG", "MGTSP", "MISB", "MISBP",
    "MRKC", "MRKK", "MRKP", "MRKS", "MRKU", "MRKV", "MRKY",
    "MRKZ", "MSRS", "MSTT", "MTLR", "MTLRP", "MVID", "NKHP",
    "NKNC", "NKNCP", "NNSB", "NNSBP", "OGKB", "OKEY", "PMSB",
    "PMSBP", "RGSS", "RKKE", "RNFT", "ROLO", "RTGZ", "RTSB",
    "RTSBP", "RZSB", "SAGO", "SAGOP", "SELG", "SPBE", "TGKA",
    "TGKB", "TGKBP", "TGKN", "TNSE", "TTLK", "UKUZ", "USBN",
    "VEON-RX", "VJGZ", "VJGZP", "VRSBP", "VSYD", "WTCM",
    "YRSB", "YRSBP", "ZAYM", "CHMF", "GAZP", "GMKN", "LKOH",
    "MGNT", "MOEX", "NLMK", "NVTK", "PHOR", "PLZL", "ROSN",
    "RUAL", "SBER", "SIBN", "SNGS", "SNGSP", "TATN", "TATNP",
    "TRNFP", "YDEX", "X5", "TCSG", "HEAD",
}

df = pd.read_excel(FILE_NAME, sheet_name=SHEET_NAME)

# Отбираем строки, где слово из первого столбца есть в списке тикеров
filtered_df = df[df.iloc[:, COL_NAME].isin(RUSSIAN_TICKERS)]

filtered_df.to_excel(OUTPUT_FILE, index=False)
print(f"Готово! Оставлено {len(filtered_df)} строк с тикерами. Результат сохранён в {OUTPUT_FILE}")