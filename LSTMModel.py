import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sqlalchemy
import ast
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1. Параметры и конфигурация
# ------------------------------
DB_CONNECTION_STRING = 'postgresql://user:password@localhost:5432/mydb'  # замените на свою БД
TABLE_NAME = 'news_sentiment'

EMBEDDING_DIM = 768          # размерность эмбеддинга новостей (нужно знать заранее или определить автоматически)
TICKER_EMBED_DIM = 16        # размерность эмбеддинга тикера
WINDOW_SIZE = 10             # длина последовательности для LSTM
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

# Пороги для разделения по времени (пример)
TRAIN_END = '2023-01-01'
VAL_END   = '2023-06-01'

# ------------------------------
# 2. Загрузка данных из SQL
# ------------------------------
engine = sqlalchemy.create_engine(DB_CONNECTION_STRING)
query = f"SELECT date, ticker, avg_sentiment, embedding, next_day_price_change FROM {TABLE_NAME}"
df = pd.read_sql(query, engine, parse_dates=['date'])
print(f"Загружено записей: {len(df)}")

# ------------------------------
# 3. Преобразование эмбеддинга из текста в массив
# ------------------------------
def parse_embedding(emb_str):
    """Преобразует строку вида '[0.1, 0.2, ...]' или '0.1 0.2 ...' в numpy массив"""
    if isinstance(emb_str, str):
        # Попытка распарсить как список
        try:
            # Если строка в формате списка Python
            emb = np.array(ast.literal_eval(emb_str), dtype=np.float32)
        except:
            # Иначе разделяем по пробелам/запятым
            emb_str = emb_str.replace('[', '').replace(']', '').replace(',', ' ')
            emb = np.fromstring(emb_str, sep=' ', dtype=np.float32)
        return emb
    elif isinstance(emb_str, (list, np.ndarray)):
        return np.array(emb_str, dtype=np.float32)
    else:
        raise ValueError(f"Неизвестный тип эмбеддинга: {type(emb_str)}")

# Применяем преобразование
df['embedding'] = df['embedding'].apply(parse_embedding)

# Проверим размерность
sample_emb = df['embedding'].iloc[0]
if EMBEDDING_DIM is None:
    EMBEDDING_DIM = len(sample_emb)
else:
    assert len(sample_emb) == EMBEDDING_DIM, "Размерность эмбеддинга не соответствует заданной"

print(f"Размерность эмбеддинга: {EMBEDDING_DIM}")

# ------------------------------
# 4. Агрегация по (date, ticker)
#    (если на один день и тикер несколько новостей)
# ------------------------------
# Усредняем тональность и эмбеддинги
def aggregate_group(group):
    avg_sent = group['avg_sentiment'].mean()
    avg_emb = np.mean(np.stack(group['embedding'].values), axis=0)
    # Для изменения цены берём первое (предполагаем, что для всех записей одного дня оно одинаково)
    price_change = group['next_day_price_change'].iloc[0]
    return pd.Series({
        'avg_sentiment': avg_sent,
        'embedding': avg_emb,
        'next_day_price_change': price_change
    })

df_agg = df.groupby(['date', 'ticker']).apply(aggregate_group).reset_index()
print(f"После агрегации записей: {len(df_agg)}")

# ------------------------------
# 5. Создание полного календаря для каждого тикера (заполнение пропусков)
#    Чтобы последовательности были равномерными по времени
# ------------------------------
# Определим общий временной диапазон
all_dates = pd.date_range(start=df_agg['date'].min(), end=df_agg['date'].max(), freq='D')
all_tickers = df_agg['ticker'].unique()

# Создаем мультииндекс (дата, тикер)
full_index = pd.MultiIndex.from_product([all_dates, all_tickers], names=['date', 'ticker'])
df_full = pd.DataFrame(index=full_index).reset_index()

# Объединяем с агрегированными данными
df_full = df_full.merge(df_agg, on=['date', 'ticker'], how='left')

# Заполняем пропуски в признаках нулями (можно использовать ffill, но осторожно)
sent_col = 'avg_sentiment'
emb_col = 'embedding'
target_col = 'next_day_price_change'

# Для тональности и эмбеддинга заполняем нулями (нет новостей -> нейтрально/нулевой вектор)
df_full[sent_col] = df_full[sent_col].fillna(0.0)
# Эмбеддинги: создаём массив нулей для пропусков
df_full[emb_col] = df_full[emb_col].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(EMBEDDING_DIM, dtype=np.float32))

# Целевая переменная: оставляем NaN для дней, где нет данных – при обучении такие последовательности исключаются или заполняются
# Но мы будем использовать только те дни, где есть целевая переменная. Для этого удалим строки с NaN в target.
df_full = df_full.dropna(subset=[target_col]).reset_index(drop=True)

print(f"После заполнения календаря записей: {len(df_full)}")

# ------------------------------
# 6. Сортировка по дате и тикеру
# ------------------------------
df_full = df_full.sort_values(['ticker', 'date']).reset_index(drop=True)

# ------------------------------
# 7. Создание признаков и кодировка тикеров
# ------------------------------
# Маппинг тикеров в индексы
ticker_to_idx = {ticker: i for i, ticker in enumerate(all_tickers)}
num_tickers = len(all_tickers)
df_full['ticker_idx'] = df_full['ticker'].map(ticker_to_idx)

# Объединяем признаки: тональность + эмбеддинг
features = []
for _, row in df_full.iterrows():
    feat = np.concatenate([[row['avg_sentiment']], row['embedding']])
    features.append(feat)
features = np.array(features, dtype=np.float32)

targets = df_full[target_col].values.astype(np.float32)

# Нормализация признаков (StandardScaler)
scaler_features = StandardScaler()
features_scaled = scaler_features.fit_transform(features)

scaler_target = StandardScaler()
targets_scaled = scaler_target.fit_transform(targets.reshape(-1, 1)).flatten()

# Добавим в df_full масштабированные признаки
df_full['features_scaled'] = list(features_scaled)
df_full['target_scaled'] = targets_scaled

# ------------------------------
# 8. Разделение на train/val/test по временным порогам
# ------------------------------
train_mask = df_full['date'] < TRAIN_END
val_mask = (df_full['date'] >= TRAIN_END) & (df_full['date'] < VAL_END)
test_mask = df_full['date'] >= VAL_END

df_train = df_full[train_mask].copy()
df_val   = df_full[val_mask].copy()
df_test  = df_full[test_mask].copy()

print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# ------------------------------
# 9. Класс Dataset для создания последовательностей
# ------------------------------
class TickerSequenceDataset(Dataset):
    def __init__(self, df, window_size, ticker_col='ticker_idx', feature_col='features_scaled', target_col='target_scaled'):
        """
        df должен быть отсортирован по дате внутри каждого тикера.
        Для каждого тикера создаем последовательности длины window_size.
        """
        self.window_size = window_size
        self.ticker_col = ticker_col
        self.feature_col = feature_col
        self.target_col = target_col

        self.samples = []  # список кортежей (последовательность признаков, target, индекс тикера для эмбеддинга)

        # Группируем по тикеру
        for ticker, group in df.groupby('ticker'):
            group = group.sort_values('date')  # убедимся
            features = np.stack(group[feature_col].values)
            targets = group[target_col].values
            ticker_idx = group[ticker_col].iloc[0]  # одинаков для всех строк группы

            # Скользящее окно
            for i in range(len(group) - window_size):
                seq_features = features[i:i+window_size]   # (window_size, feat_dim)
                seq_target = targets[i+window_size]        # предсказываем target для следующего дня после окна
                self.samples.append((seq_features, seq_target, ticker_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_feat, target, ticker_idx = self.samples[idx]
        return (
            torch.tensor(seq_feat, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(ticker_idx, dtype=torch.long)
        )

# Создаем датасеты
train_dataset = TickerSequenceDataset(df_train, WINDOW_SIZE)
val_dataset   = TickerSequenceDataset(df_val, WINDOW_SIZE)
test_dataset  = TickerSequenceDataset(df_test, WINDOW_SIZE)

print(f"Число последовательностей: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

# DataLoader'ы
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------
# 10. Определение модели LSTM с эмбеддингом тикера
# ------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_tickers, ticker_embed_dim, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Эмбеддинг для тикера
        self.ticker_embedding = nn.Embedding(num_tickers, ticker_embed_dim)

        # LSTM слой: входная размерность = input_size (признаки новостей) + ticker_embed_dim
        self.lstm = nn.LSTM(input_size + ticker_embed_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        # Выходной слой
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, ticker_idx):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        # Получаем эмбеддинги тикеров и расширяем на каждый шаг последовательности
        ticker_emb = self.ticker_embedding(ticker_idx)  # (batch, ticker_embed_dim)
        ticker_emb = ticker_emb.unsqueeze(1).expand(batch_size, seq_len, -1)  # (batch, seq_len, ticker_embed_dim)

        # Конкатенируем признаки
        lstm_input = torch.cat([x, ticker_emb], dim=-1)  # (batch, seq_len, input_size + ticker_embed_dim)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        # Используем последний скрытый состояние последнего слоя
        last_hidden = hidden[-1]  # (batch, hidden_size)
        output = self.fc(last_hidden)  # (batch, 1)
        return output.squeeze(-1)

# Параметры модели
input_size = features_scaled.shape[1]  # тональность + эмбеддинг
hidden_size = 64
num_layers = 2
dropout = 0.3

model = LSTMModel(input_size, hidden_size, num_layers, num_tickers, TICKER_EMBED_DIM, dropout).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ------------------------------
# 11. Функция обучения
# ------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        x, y, ticker_idx = batch
        x, y, ticker_idx = x.to(device), y.to(device), ticker_idx.to(device)

        optimizer.zero_grad()
        outputs = model(x, ticker_idx)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            x, y, ticker_idx = batch
            x, y, ticker_idx = x.to(device), y.to(device), ticker_idx.to(device)
            outputs = model(x, ticker_idx)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

# ------------------------------
# 12. Цикл обучения
# ------------------------------
best_val_loss = float('inf')
for epoch in range(1, EPOCHS+1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_lstm_model.pth')
        print(f"Epoch {epoch}: сохранена лучшая модель (val_loss={val_loss:.6f})")

    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")

print("Обучение завершено.")

# ------------------------------
# 13. Оценка на тесте
# ------------------------------
model.load_state_dict(torch.load('best_lstm_model.pth'))
test_loss = evaluate(model, test_loader, criterion, DEVICE)
print(f"Test loss (MSE): {test_loss:.6f}")

# Пример предсказания в исходном масштабе
# Для этого нужно получить предсказания для тестового набора и применить обратное преобразование scaler_target
model.eval()
predictions_scaled = []
targets_scaled = []
with torch.no_grad():
    for batch in test_loader:
        x, y, ticker_idx = batch
        x, ticker_idx = x.to(DEVICE), ticker_idx.to(DEVICE)
        outputs = model(x, ticker_idx)
        predictions_scaled.extend(outputs.cpu().numpy())
        targets_scaled.extend(y.numpy())

predictions_orig = scaler_target.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
targets_orig = scaler_target.inverse_transform(np.array(targets_scaled).reshape(-1, 1)).flatten()

# Можно вычислить дополнительные метрики, например MAE
mae = np.mean(np.abs(predictions_orig - targets_orig))
print(f"Test MAE (original scale): {mae:.4f}")