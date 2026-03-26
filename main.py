"""
Прогнозирование цен акций с использованием LSTM нейронных сетей
Исправленная версия с обработкой ошибок
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')


# ==================== 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ====================

def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загрузка исторических данных акций
    """
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    # Приводим к обычному DataFrame, если есть MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data = data[ticker] if ticker in data.columns.levels[0] else data.iloc[:, 0:5]
    return data


# Пример загрузки
ticker = "MSFT"
data = load_stock_data(ticker, "2020-01-01", "2024-01-01")
print(f"Загружено данных: {data.shape}")
print(f"Колонки: {data.columns.tolist()}")


# ==================== 2. РАСЧЕТ ТЕХНИЧЕСКИХ ИНДИКАТОРОВ ====================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Расчет технических индикаторов
    """

    df = df.copy()

    # Убедимся, что у нас есть необходимые колонки
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Отсутствует колонка {col} в данных")

    # 1. Скользящие средние
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # 2. Экспоненциальные скользящие средние
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # 3. MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # 4. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 5. Bollinger Bands - ИСПРАВЛЕННАЯ ЧАСТЬ
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()

    # Явно преобразуем в Series, если нужно
    bb_std = df['Close'].rolling(window=20).std()

    # Обрабатываем случай, когда bb_std может быть DataFrame
    if isinstance(bb_std, pd.DataFrame):
        # Если это DataFrame, берем первую колонку
        bb_std = bb_std.iloc[:, 0] if len(bb_std.columns) > 0 else bb_std.squeeze()
    elif isinstance(bb_std, pd.Series):
        # Если это Series, оставляем как есть
        pass
    else:
        # В других случаях пытаемся преобразовать
        bb_std = pd.Series(bb_std, index=df.index)

    # Теперь вычисляем полосы
    df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

    # 6. ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    # Используем concat правильно
    ranges_df = pd.DataFrame({
        'high_low': high_low,
        'high_close': high_close,
        'low_close': low_close
    })

    true_range = ranges_df.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # 7. Объемные индикаторы
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA'].replace(0, np.nan)

    # 8. Моментум
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # 9. Логарифмическая доходность и волатильность
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)

    # Удаляем NaN значения
    df = df.dropna()

    print(f"Рассчитано индикаторов: {len([col for col in df.columns if col not in required_columns])}")

    return df


# Применяем расчет индикаторов
try:
    data_with_indicators = calculate_technical_indicators(data)
    print(f"Данные после расчета индикаторов: {data_with_indicators.shape}")
except Exception as e:
    print(f"Ошибка при расчете индикаторов: {e}")
    # Используем только базовые признаки
    data_with_indicators = data.copy()
    data_with_indicators['Returns'] = np.log(data_with_indicators['Close'] / data_with_indicators['Close'].shift(1))
    data_with_indicators = data_with_indicators.dropna()

# ==================== 3. ПОДГОТОВКА ДАННЫХ ====================

from sklearn.preprocessing import StandardScaler


def prepare_sequences(data: pd.DataFrame, seq_length: int = 60, forecast_horizon: int = 5):
    """
    Подготовка данных для LSTM
    """

    # Выбираем признаки - используем только доступные
    available_features = []
    for feature in ['Close', 'Volume', 'Returns', 'SMA_10', 'SMA_20', 'SMA_50',
                    'MACD', 'MACD_Histogram', 'RSI', 'BB_Width', 'ATR',
                    'Volume_Ratio', 'Momentum', 'Volatility']:
        if feature in data.columns:
            available_features.append(feature)

    print(f"Используем признаки: {available_features}")

    features = data[available_features].values
    target = data['Close'].values.reshape(-1, 1)

    # Масштабирование
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    features_scaled = scaler_X.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target)

    # Создание последовательностей
    X, y = [], []

    for i in range(len(features_scaled) - seq_length - forecast_horizon):
        X.append(features_scaled[i:(i + seq_length)])
        y.append(target_scaled[i + seq_length + forecast_horizon - 1, 0])

    X = np.array(X)
    y = np.array(y)

    # Разделение на train/test
    split_idx = int(len(X) * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, available_features


# Подготавливаем данные
X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_cols = prepare_sequences(
    data_with_indicators, seq_length=60, forecast_horizon=5
)

# ==================== 4. СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ ====================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def create_simple_lstm(input_shape, n_features):
    """
    Создание упрощенной LSTM модели
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


# Создаем модель
model = create_simple_lstm(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    n_features=len(feature_cols)
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# Обучение
print("\nНачинаем обучение...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)


# ==================== 5. ОЦЕНКА МОДЕЛИ ====================

def evaluate_model(model, X_test, y_test, scaler_y):
    """
    Оценка модели
    """
    # Предсказания
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Обратное масштабирование
    y_test_reshaped = y_test.reshape(-1, 1)
    y_pred_reshaped = y_pred_scaled.reshape(-1, 1)

    # Создаем dummy массивы для обратного преобразования
    dummy_dim = len(feature_cols)
    dummy_test = np.zeros((len(y_test_reshaped), dummy_dim))
    dummy_pred = np.zeros((len(y_pred_reshaped), dummy_dim))

    dummy_test[:, 0] = y_test_reshaped[:, 0]
    dummy_pred[:, 0] = y_pred_reshaped[:, 0]

    y_test_original = scaler_y.inverse_transform(dummy_test)[:, 0]
    y_pred_original = scaler_y.inverse_transform(dummy_pred)[:, 0]

    # Метрики
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
    r2 = r2_score(y_test_original, y_pred_original)

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 50)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    print("=" * 50)

    return y_test_original, y_pred_original


y_test_orig, y_pred_orig = evaluate_model(model, X_test, y_test, scaler_y)

# ==================== 6. ВИЗУАЛИЗАЦИЯ ====================

import matplotlib.pyplot as plt


def plot_results(history, y_test, y_pred, y_test_orig, y_pred_orig):
    """
    Визуализация результатов
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Фактические vs Предсказанные (масштабированные)
    axes[0, 1].scatter(y_test[:200], y_pred[:200], alpha=0.5)
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0, 1].set_title('Scaled: Actual vs Predicted')
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].grid(True)

    # 3. Временной ряд
    axes[1, 0].plot(y_test_orig[:100], label='Actual', linewidth=2)
    axes[1, 0].plot(y_pred_orig[:100], label='Predicted', linewidth=2)
    axes[1, 0].set_title('Time Series: Actual vs Predicted')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Price')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4. Ошибки
    errors = y_test_orig - y_pred_orig
    axes[1, 1].hist(errors, bins=50, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].set_title('Prediction Errors Distribution')
    axes[1, 1].set_xlabel('Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


plot_results(history, y_test, model.predict(X_test, verbose=0), y_test_orig, y_pred_orig)


# ==================== 7. ПРОГНОЗ НА БУДУЩЕЕ ====================

def make_prediction(model, last_data, scaler_X, scaler_y, feature_cols, n_days=5):
    """
    Прогноз на будущее
    """
    predictions = []
    current_seq = last_data.copy()

    for _ in range(n_days):
        # Подготовка данных
        current_scaled = scaler_X.transform(
            current_seq.reshape(-1, len(feature_cols))
        ).reshape(1, current_seq.shape[0], len(feature_cols))

        # Прогноз
        pred_scaled = model.predict(current_scaled, verbose=0)[0, 0]

        # Обратное масштабирование
        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, 0] = pred_scaled
        pred = scaler_y.inverse_transform(dummy)[0, 0]

        predictions.append(pred)

        # Обновление последовательности
        new_point = current_seq[-1].copy()
        new_point[0] = pred  # Обновляем цену

        current_seq = np.vstack([current_seq[1:], new_point.reshape(1, -1)])

    return predictions


# Прогноз на 5 дней
last_sequence = X_test[-1]
future_predictions = make_prediction(
    model, last_sequence, scaler_X, scaler_y, feature_cols, n_days=5
)

print("\nПРОГНОЗ НА 5 ДНЕЙ:")
print("-" * 30)
for i, pred in enumerate(future_predictions, 1):
    print(f"День {i}: ${pred:.2f}")


# ==================== 8. АВТОМАТИЧЕСКАЯ ПРОВЕРКА И УСТАНОВКА ====================

def check_and_install_packages():
    """
    Проверка и установка необходимых пакетов
    """
    import subprocess
    import sys

    required_packages = [
        'numpy',
        'pandas',
        'yfinance',
        'scikit-learn',
        'tensorflow',
        'matplotlib'
    ]

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} уже установлен")
        except ImportError:
            print(f"✗ Устанавливаем {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} успешно установлен")


# Проверяем и устанавливаем пакеты
print("\nПРОВЕРКА ЗАВИСИМОСТЕЙ:")
check_and_install_packages()

print("\n" + "=" * 70)
print("ПРОГРАММА УСПЕШНО ЗАВЕРШЕНА!")
print(f"Модель обучена на данных {ticker}")
print(f"Использовано признаков: {len(feature_cols)}")
print("=" * 70)