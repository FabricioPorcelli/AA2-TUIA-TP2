import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("TensorFlow version:", tf.__version__)

# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'flappy_birds_q_table_final.pkl'
print(f"\nCargando Q-table desde {QTABLE_PATH}...")

with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

print(f"Q-table cargada con {len(q_table)} estados")

# --- Preparar datos para entrenamiento ---
X = []  # Estados discretos
y = []  # Q-values para cada acción

for state, q_values in q_table.items():
    X.append(state)
    y.append(q_values)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"\nDatos preparados:")
print(f"  - X shape: {X.shape}")
print(f"  - y shape: {y.shape}")
print(f"  - Número de features: {X.shape[1]}")
print(f"  - Número de acciones: {y.shape[1]}")
print(f"  - Q-values min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

# Dividir en train y validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDivisión train/validation:")
print(f"  - Train: {len(X_train)} muestras")
print(f"  - Validation: {len(X_val)} muestras")

# --- Definir la red neuronal ---
num_features = X.shape[1]  # 4 features
num_actions = y.shape[1]   # 2 acciones

model = keras.Sequential([
    layers.Input(shape=(num_features,)),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(64, activation='relu'),

    # Salida: Q-value para cada acción
    layers.Dense(num_actions, activation='linear')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nArquitectura del modelo:")
model.summary()

# --- Entrenar la red neuronal ---
print(" --- Entrenamiento --- ")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=1,
            min_lr=1e-6
        )
    ]
)

# --- Mostrar resultados ---
print(" --- Resultados --- ")

final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"\nMétricas finales:")
print(f"  - Train Loss (MSE): {final_train_loss:.4f}")
print(f"  - Val Loss (MSE): {final_val_loss:.4f}")

# --- Evaluar algunas predicciones ---
print(" --- Comparación Q-table vs NN --- ")

num_examples = min(10, len(X_val))
for i in range(num_examples):
    state = X_val[i:i+1]
    true_q = y_val[i]
    pred_q = model.predict(state, verbose=0)[0]
    
    print(f"\nEstado {i+1}: {state[0]}")
    print(f"  Q-table:      [{true_q[0]:6.2f}, {true_q[1]:6.2f}]")
    print(f"  Red Neuronal: [{pred_q[0]:6.2f}, {pred_q[1]:6.2f}]")
    print(f"  Mejor acción Q-table: {np.argmax(true_q)}")
    print(f"  Mejor acción Red:     {np.argmax(pred_q)}")

# --- Guardar el modelo ---
model.save('flappy_q_nn_model.keras')

print(f'\nModelo guardado en flappy_q_nn_model.keras')