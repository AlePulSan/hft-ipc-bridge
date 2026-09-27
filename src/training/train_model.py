import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import os

# Forzamos la ruta absoluta relativa a la ubicación del script
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "../../models")

# Crea la carpeta 'models' si no existe
os.makedirs(models_dir, exist_ok=True)

MODEL_PATH = os.path.join(models_dir, "quant_model.onnx")

# Carga de datos
df = yf.download('SPY', period='5y', interval='1d')

# Creación de variables
print(f"Procesando {len(df)} transacciones...")
# Homogeneizamos los nombres con el motor C++ (yfinance devuelve 'Close' y 'Volume')
df = df.rename(columns={'Close': 'price', 'Volume': 'size'})

# Ordenar por tiempo por si acaso (evitar look-ahead bias)
df = df.sort_index()

# Calcular retornos y medias móviles
df['returns'] = df['price'].pct_change()
df['ma_5'] = df['price'].rolling(window=5).mean()
df['ma_15'] = df['price'].rolling(window=15).mean()

# Valor del Target: 1 si el precio sube en el siguiente tick, 0 si baja o se mantiene
df['target'] = (df['price'].shift(-1) > df['price']).astype(int)

# Quitamos nulos
df = df.dropna()

# Seleccionar features para el modelo
features = ['price', 'size', 'returns', 'ma_5', 'ma_15']

# Split Temporal (split normal mezcla pasado y futuro. El test debe ser estricto hacia adelante)(sesgo preclasificacion)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[features].values.astype(np.float32)
y_train = train_df['target'].values
X_test = test_df[features].values.astype(np.float32)
y_test = test_df['target'].values

# Entrenamiento del modelo
print("Entrenando clasificador XGBoost...")
model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

#Medidas del rendimiento
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Precisión del modelo en test: {accuracy:.2f}")
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, preds))
print("\nReporte de Clasificación:\n", classification_report(y_test, preds))

print("\nConvirtiendo modelo a formato ONNX...")
# Definimos el tensor de entrada (5 variables tipo flotantes)
initial_type = [('float_input', FloatTensorType([None, len(features)]))]
onnx_model = convert_xgboost(model, initial_types=initial_type)

with open(MODEL_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Modelo exportado a {MODEL_PATH}")