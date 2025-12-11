# Conclusiones - Flappy Bird con Q-Learning y Deep Q-Learning

## 1. Ingeniería de Características y Discretización del Estado

### 1.1 Variables del Entorno

El entorno de Flappy Bird (PLE) proporciona 8 variables continuas que describen el estado del juego:

- `player_y`: posición vertical del pájaro
- `player_vel`: velocidad vertical del pájaro
- `next_pipe_dist_to_player`: distancia horizontal al próximo tubo
- `next_pipe_top_y`, `next_pipe_bottom_y`: límites del hueco del próximo tubo
- `next_next_pipe_dist_to_player`: distancia al siguiente tubo
- `next_next_pipe_top_y`, `next_next_pipe_bottom_y`: límites del hueco del siguiente tubo

### 1.2 Feature Engineering

Para reducir la dimensionalidad del espacio de estados y capturar la información más relevante, se aplicaron las siguientes transformaciones:

#### Variables derivadas:

1. **`gap1` y `gap2`**: Centro vertical de los huecos de los tubos
   ```
   gap1 = (next_pipe_top_y + next_pipe_bottom_y) / 2
   gap2 = (next_next_pipe_top_y + next_next_pipe_bottom_y) / 2
   ```

2. **`delta_y`**: Diferencia vertical entre el pájaro y el centro del próximo hueco
   ```
   delta_y = player_y - gap1
   ```
   Esta es la variable **más crítica** ya que indica si el pájaro debe subir o bajar.

3. **`pipe_trend`**: Cambio de altura entre tubos consecutivos
   ```
   pipe_trend = gap2 - gap1
   ```
   Permite anticipar si el siguiente tubo requiere ascenso o descenso.

### 1.3 Discretización con Bins Uniformes

Se redujo el espacio de estados continuo a **4 variables discretas**:

| Variable | Descripción | Rango | Bins | Interpretación |
|----------|-------------|-------|------|----------------|
| `dist_bin` | Distancia horizontal al tubo | [0, 300] | 10 | Tiempo hasta colisión |
| `dy_bin` | Desalineación vertical | [-200, 200] | 10 | Necesidad de corrección |
| `vel_bin` | Velocidad vertical | [-10, 10] | 5 | Tendencia de movimiento |
| `trend_bin` | Cambio de altura entre tubos | [-150, 150] | 7 | Anticipación |

**Espacio de estados total**: 10 × 10 × 5 × 7 = **3,500 estados**

#### Justificación de la elección:

- **Reducción drástica**: De 8 variables continuas (espacio infinito) a 3,500 estados discretos.
- **Variables informativas**: Las 4 variables capturan toda la información necesaria para tomar decisiones óptimas.
- **Bins uniformes**: Facilita la implementación y garantiza cobertura uniforme del espacio.
- **Tamaño manejable**: 3,500 estados es suficientemente pequeño para Q-Learning tabular pero suficientemente grande para capturar matices importantes.

La función de discretización utiliza bins de ancho fijo calculados como:

```python
bin_width = (max_value - min_value) / n_bins
bin_index = int((value - min_value) / bin_width)
```

Este enfoque garantiza que cada bin cubra un rango uniforme del espacio de valores.

---

## 2. Entrenamiento y Resultados

### 2.1 Agente Q-Learning (Tabular)

#### Hiperparámetros:

- **Episodios de entrenamiento**: 120,000
- **Learning rate (α)**: 0.12
- **Discount factor (γ)**: 0.95
- **Epsilon inicial**: 1.0 (exploración completa)
- **Epsilon final**: 0.01 (1% exploración)
- **Epsilon decay**: 0.9999

#### Estrategia de exploración:

Se implementó una política **epsilon-greedy** con decay exponencial, lo que permitió al agente:
1. Explorar exhaustivamente en las primeras etapas (epsilon alto)
2. Converger gradualmente hacia explotación de la política aprendida
3. Mantener un 1% de exploración al final para evitar óptimos locales

#### Resultados:

- **Reward mínimo observado**: ~17,000
- **Reward máximo observado**: ~134,000
- **Varianza**: Alta (el desempeño fluctúa significativamente entre episodios)

#### Observaciones:

- El agente **aprende a jugar exitosamente**, logrando pasar múltiples tubos consistentemente.
- La **alta varianza** sugiere que algunos estados raros o configuraciones específicas de tubos no están perfectamente aprendidos.
- La Q-table logra capturar las decisiones fundamentales (cuándo saltar vs. no saltar).

---

### 2.2 Agente con Red Neuronal (Deep Q-Learning)

#### Arquitectura de la Red:

```
Input (4 features)
    ↓
Dense(128, relu) + Dropout(0.2)
    ↓
Dense(64, relu)
    ↓
Dense(2, linear) → Q-values para cada acción
```

#### Entrenamiento:

- **Datos**: Q-table del agente tabular (3,500 estados)
- **Épocas**: 200 (con early stopping)
- **Batch size**: 32
- **Optimizador**: Adam (learning rate = 0.001)
- **Loss**: MSE (Mean Squared Error)
- **Callbacks**: Early stopping (patience=20), ReduceLROnPlateau

La red fue entrenada mediante **aprendizaje supervisado** para aproximar los Q-values de la tabla, actuando como una **aproximación funcional** continua del espacio discreto.

#### Resultados:

- **Reward observado**: >400,000 (sin observar fin de episodio)
- **Varianza**: Muy baja (desempeño extremadamente consistente)
- **Robustez**: No se observaron fallas durante pruebas extensas

#### Observaciones clave:

1. **Generalización superior**: La red neuronal **generaliza** a estados no vistos durante el entrenamiento tabular, interpolando Q-values de manera inteligente.

2. **Eliminación de la varianza**: Al suavizar la función Q, la red elimina "ruido" en las decisiones que causaba fallos ocasionales en el agente tabular.

3. **Desempeño sobresaliente**: El agente neuronal alcanza rewards **3-30× superiores** al agente tabular, sugiriendo que ha aprendido una política casi óptima.

4. **Convergencia a política determinista**: La red aprende una función continua que mapea estados → Q-values de forma consistente, eliminando ambigüedades.

---

## 3. Análisis Comparativo

### 3.1 Comparación de Desempeño

| Métrica | Q-Learning Tabular | Deep Q-Learning (NN) |
|---------|-------------------|---------------------|
| **Reward promedio** | ~50,000 | >400,000 |
| **Reward máximo** | ~134,000 | No observado (∞?) |
| **Varianza** | Alta | Muy baja |
| **Consistencia** | Moderada | Excelente |
| **Estados aprendidos** | 3,500 (discretos) | Continuo (generaliza) |
| **Velocidad de inferencia** | Instantánea (lookup) | ~1-2ms (con optimización) |

### 3.2 Ventajas y Desventajas

#### Q-Learning Tabular:

**Ventajas:**
- Simple de implementar y debuggear
- Garantías teóricas de convergencia
- Interpretabilidad directa (se puede inspeccionar la Q-table)
- No requiere frameworks de deep learning

**Desventajas:**
- Limitado a espacios de estados pequeños
- No generaliza a estados no visitados
- Alta varianza en estados poco frecuentes
- Requiere discretización manual (pérdida de información)

#### Deep Q-Learning (Red Neuronal):

**Ventajas:**
- **Generalización excepcional**: Interpola Q-values en todo el espacio
- Maneja espacios de estados continuos o muy grandes
- Suaviza decisiones, eliminando comportamientos erráticos
- Desempeño superior en prácticamente todos los casos

**Desventajas:**
- Más complejo de implementar
- Requiere Q-table pre-entrenada (en este caso)
- "Caja negra": menos interpretable
- Requiere optimización para inferencia rápida

---

## 4. Conclusiones Finales

### 4.1 Aprendizaje Logrado

1. **Q-Learning tabular es viable** para problemas con espacios de estados pequeños (~3,500 estados), logrando desempeños aceptables con suficiente entrenamiento.

2. **La discretización es crítica**: Reducir de 8 variables continuas a 4 variables discretas informativas fue clave para el éxito. Una mala elección de features habría imposibilitado el aprendizaje.

3. **Deep Q-Learning supera consistentemente** al enfoque tabular, logrando mejoras de **3-30× en reward** gracias a la capacidad de generalización de las redes neuronales.

### 4.2 Diferencia Fundamental

La **diferencia clave** entre ambos enfoques es la **generalización**:

- **Q-Learning tabular**: Aprende decisiones independientes para cada uno de los 3,500 estados discretos. Estados similares no comparten información.

- **Deep Q-Learning**: Aprende una **función continua** Q(s, a) que generaliza entre estados similares. Si aprende que en el estado (5, 3, 2, 4) debe saltar, automáticamente sabe que en (5, 3, 2, 5) probablemente también deba saltar.

### 4.3 Lecciones Prácticas

1. **Feature engineering > Cantidad de datos**: Una buena elección de variables (delta_y, pipe_trend) fue más importante que aumentar bins o episodios de entrenamiento.

2. **La exploración debe ser gradual**: El decay exponencial de epsilon (0.9999) permitió visitar suficientes estados antes de converger.

3. **Las redes neuronales "arreglan" Q-tables imperfectas**: Incluso con una Q-table con alta varianza, la red logró extraer una política casi óptima.

4. **Optimización de inferencia es crucial**: Sin `@tf.function`, el agente neuronal era inviable para juego en tiempo real.

---

## 5. Referencias

- Material teórico de la cátedra.
- PyGame Learning Environment (PLE): https://pygame-learning-environment.readthedocs.io/en/latest/user/games/flappybird.html