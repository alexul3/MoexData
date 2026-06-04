import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Конфигурация
# ------------------------------
MODEL_NAME = 'DeepPavlov/rubert-base-cased'   # ваша основная модель
MAX_LEN = 256                                 # макс. длина текста в токенах
BATCH_SIZE_GROUPS = 8                         # число новостных групп в одном батче (подберите под GPU)
EPOCHS = 5
LEARNING_RATE = 2e-5                          # стандартный lr для fine-tuning BERT
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используем устройство: {DEVICE}")

# ------------------------------
# 1. Загрузка и подготовка данных
# ------------------------------
df = pd.read_excel('news_embedding_input.xlsx')
df = df.dropna(subset=['news', 'ticker', 'Target'])

# Приводим datetime к единому формату и сортируем
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

# Создаём ID новости (хеш) для группировки
df['news_id'] = df['news'].apply(lambda x: hash(x))

# Проверяем, что Tone – числовой, если нет – превращаем в числа
if df['Tone'].dtype == 'object':
    tone_encoder = LabelEncoder()
    df['Tone'] = tone_encoder.fit_transform(df['Tone'])
else:
    tone_encoder = None

# ------------------------------
# 2. Сплит по новостям (без утечек)
# ------------------------------
unique_ids = df['news_id'].unique()
train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

train_df = df[df['news_id'].isin(train_ids)]
val_df   = df[df['news_id'].isin(val_ids)]
test_df  = df[df['news_id'].isin(test_ids)]

print(f"Train: {len(train_df)} rows, {train_df['news_id'].nunique()} unique news")
print(f"Val:   {len(val_df)} rows, {val_df['news_id'].nunique()} unique news")
print(f"Test:  {len(test_df)} rows, {test_df['news_id'].nunique()} unique news")

# ------------------------------
# 3. Кодирование тикеров (LabelEncoder)
# ------------------------------
ticker_encoder = LabelEncoder()
ticker_encoder.fit(df['ticker'])
num_tickers = len(ticker_encoder.classes_)

# ------------------------------
# 4. Группировка данных для DataLoader
# ------------------------------
def create_groups(dataframe):
    """Возвращает список групп, где каждая группа — DataFrame с одним news_id."""
    return [group for _, group in dataframe.groupby('news_id')]

train_groups = create_groups(train_df)
val_groups   = create_groups(val_df)
test_groups  = create_groups(test_df)

def collate_groups(groups):
    """
    Объединяет несколько групп в один батч.
    groups: list of DataFrame
    Возвращает:
        texts: list of str – уникальные тексты новостей (по одному на группу)
        tickers: LongTensor (total_rows,)
        tones: FloatTensor (total_rows,)
        targets: FloatTensor (total_rows,)
        group_sizes: LongTensor (количество строк в каждой группе)
    """
    texts = []
    ticker_list = []
    tone_list = []
    target_list = []
    group_sizes = []

    for grp in groups:
        # Текст у всей группы одинаковый
        texts.append(grp['news'].iloc[0])
        group_sizes.append(len(grp))
        # Преобразуем тикеры в индексы
        ticker_list.extend(ticker_encoder.transform(grp['ticker'].tolist()))
        tone_list.extend(grp['Tone'].tolist())
        target_list.extend(grp['Target'].tolist())

    return {
        'texts': texts,
        'tickers': torch.tensor(ticker_list, dtype=torch.long),
        'tones': torch.tensor(tone_list, dtype=torch.float),
        'targets': torch.tensor(target_list, dtype=torch.float),
        'group_sizes': torch.tensor(group_sizes, dtype=torch.long)
    }

# DataLoader'ы
train_loader = DataLoader(train_groups, batch_size=BATCH_SIZE_GROUPS, shuffle=True,
                          collate_fn=collate_groups)
val_loader   = DataLoader(val_groups, batch_size=BATCH_SIZE_GROUPS, shuffle=False,
                          collate_fn=collate_groups)
test_loader  = DataLoader(test_groups, batch_size=BATCH_SIZE_GROUPS, shuffle=False,
                          collate_fn=collate_groups)

# ------------------------------
# 5. Модель: BERT + ticker embedding + MLP
# ------------------------------
class EfficientBERTRegressor(nn.Module):
    def __init__(self, model_name, num_tickers, tone_dim=1, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.ticker_embedding = nn.Embedding(num_tickers, 32)
        combined_dim = self.hidden_size + 32 + tone_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, texts, tickers, tones, group_sizes, tokenizer, max_len, device):
        """
        texts: list of str – уникальные тексты в батче (B групп)
        tickers: LongTensor (total_rows,) – индексы тикеров всех строк
        tones: FloatTensor (total_rows,)
        group_sizes: LongTensor (B,) – количество строк для каждого текста
        """
        # 1. Токенизация всех уникальных текстов
        encoded = tokenizer(texts, padding=True, truncation=True,
                            max_length=max_len, return_tensors='pt').to(device)
        # 2. Прогон через BERT (один проход)
        bert_output = self.bert(**encoded)
        cls_emb = bert_output.last_hidden_state[:, 0, :]   # (B, hidden_size)

        # 3. Размножаем эмбеддинги согласно group_sizes
        # Например, group_sizes = [3,2] => [cls0, cls0, cls0, cls1, cls1]
        expanded_emb = torch.repeat_interleave(cls_emb, group_sizes, dim=0)  # (total_rows, hidden)

        # 4. Добавляем эмбеддинги тикеров
        tick_emb = self.ticker_embedding(tickers)  # (total_rows, 32)

        # 5. Тональность
        if len(tones.shape) == 1:
            tones = tones.unsqueeze(1)   # (total_rows, 1)

        # 6. Конкатенация и регрессия
        x = torch.cat([expanded_emb, tick_emb, tones], dim=1)
        return self.mlp(x).squeeze(-1)   # (total_rows,)

# Инициализация модели
model = EfficientBERTRegressor(MODEL_NAME, num_tickers, tone_dim=1).to(DEVICE)

# ------------------------------
# 6. Оптимизатор и планировщик
# ------------------------------
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)
criterion = nn.MSELoss()

# ------------------------------
# 7. Функции обучения и оценки
# ------------------------------
def train_epoch(model, loader, optimizer, scheduler, tokenizer, max_len, device):
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc='Training')
    for batch in progress:
        texts = batch['texts']
        tickers = batch['tickers'].to(device)
        tones = batch['tones'].to(device)
        targets = batch['targets'].to(device)
        group_sizes = batch['group_sizes'].to(device)

        optimizer.zero_grad()
        preds = model(texts, tickers, tones, group_sizes, tokenizer, max_len, device)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix({'loss': loss.item()})
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, tokenizer, max_len, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    for batch in tqdm(loader, desc='Evaluation'):
        texts = batch['texts']
        tickers = batch['tickers'].to(device)
        tones = batch['tones'].to(device)
        targets = batch['targets'].to(device)
        group_sizes = batch['group_sizes'].to(device)

        preds = model(texts, tickers, tones, group_sizes, tokenizer, max_len, device)
        loss = criterion(preds, targets)
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(loader)
    corr = np.corrcoef(all_preds, all_targets)[0, 1]
    return avg_loss, corr

# ------------------------------
# 8. Цикл обучения
# ------------------------------
best_val_loss = float('inf')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                             tokenizer, MAX_LEN, DEVICE)
    val_loss, val_corr = evaluate(model, val_loader, tokenizer, MAX_LEN, DEVICE)
    print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} | Val corr: {val_corr:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ticker_encoder': ticker_encoder,
            'tone_encoder': tone_encoder
        }, 'best_model_efficient.pt')
        print("  -> saved best model")

# ------------------------------
# 9. Оценка на тестовом наборе
# ------------------------------
checkpoint = torch.load('best_model_efficient.pt', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_corr = evaluate(model, test_loader, tokenizer, MAX_LEN, DEVICE)
print(f"\nFinal Test Loss: {test_loss:.6f}, Test Correlation: {test_corr:.4f}")