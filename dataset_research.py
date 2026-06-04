import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_excel('rework_data_with_target.xlsx')

counts = df['ticker'].value_counts()
print(counts.describe())

# Топ-10 тикеров
counts.head(50).plot(kind='bar')
plt.title('Количество новостей по тикерам')
plt.show()