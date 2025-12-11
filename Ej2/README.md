# Ejercicio 2: Q-Learning en Flappy Bird

## Descripción

El objetivo de este ejercicio es entrenar agentes para resolver videojuegos sencillos usando Q-Learning y la librería PLE (PyGame Learning Environment).

Se implementan dos enfoques diferentes:

1. **Q-Learning tradicional:** Utilizando una Q-table con discretización del espacio de estados
2. **Aproximación con Red Neuronal:** Utilizando una red neuronal para aproximar la Q-table obtenida

---

## Instalación y Configuración

### 1. Navegar a la carpeta del ejercicio

```bash
cd Ej2
```

### 2. Crear el entorno virtual

Es necesario contar con Python 3.12:

**En Windows:**
```bash
py -3.12 -m venv venv
```

**En Linux/Mac:**
```bash
python3.12 -m venv venv
```

### 3. Activar el entorno virtual

**En Windows:**
```bash
.\venv\Scripts\activate
```

**En Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Instalar las dependencias

```bash
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```
Ej2/
├── test_agent.py                   # Script principal para probar agentes
├── train_q_agent.py                # Script de entrenamiento Q-Learning
├── train_q_nn.py                   # Script de entrenamiento Red Neuronal
├── agentes/                        # Carpeta con implementaciones de agentes
│   ├── base.py                     # Clase base para todos los agentes
│   ├── random_agent.py             # Agente con acciones aleatorias
│   ├── manual_agent.py             # Agente para juego manual
│   ├── dq_agent.py                 # Agente Q-Learning
│   └── nn_agent.py                 # Agente con Red Neuronal
├── ple/                            # Carpeta con codigo del PyGame Learning Environment
├── flappy_birds_q_table.pkl        # Q-table parcial
├── flappy_birds_q_table_final.pkl  # Q-table final (generada tras entrenar)
├── flappy_q_nn_model.keras         # Modelo de red neuronal (generado tras entrenar)
├── conclusiones.md                 # Análisis y comparación de resultados
├── requirements.txt
└── README.md
```

---

## Uso

### Probar Agentes Básicos

**Agente Aleatorio:**
```bash
python test_agent.py --agent agentes.random_agent.RandomAgent
```

**Agente Manual** (jugar con la barra espaciadora):
```bash
python test_agent.py --agent agentes.manual_agent.ManualAgent
```

---

## Ejercicio A: Q-Learning en Flappy Bird

### A.1 Completar el agente Q-Learning

- Edita el archivo `agentes/dq_agent.py` y completa las funciones `discretize_state` y `act` siguiendo las indicaciones y comentarios en el código.
- El objetivo es que el agente aprenda a jugar Flappy Bird usando Q-Learning tabular.

### A.2 Entrenar el agente

Ejecuta el script de entrenamiento:

```bash
python train_q_agent.py
```

Esto entrenará el agente y guardará la Q-table en un archivo.

### A.3 Probar el agente entrenado

Una vez entrenado, puedes testear el desempeño de tu agente ejecutando:

```bash
python test_agent.py --agent agentes.dq_agent.QAgent
```

**Nota:** Asegúrate de que el archivo de la Q-table guardado esté disponible como `flappy_birds_q_table_final.pkl` para que el agente lo cargue al iniciar.

---

## Ejercicio B: Aproximación de la Q-table con una Red Neuronal

### B.1 Entrenar una red neuronal para aproximar la Q-table

Utiliza el script `train_q_nn.py` para entrenar una red neuronal que aproxime la Q-table obtenida en el ejercicio anterior:

```bash
python train_q_nn.py
```

- El script carga la Q-table y entrena una red usando TensorFlow/Keras.
- Completa los placeholders de arquitectura y entrenamiento según sea necesario.
- El modelo se guardará como un TensorFlow SavedModel.

### B.2 Crear un agente basado en la red neuronal

- Completa la función `act` en `agentes/nn_agent.py` para que el agente use la red neuronal entrenada para tomar decisiones.
- El agente debe transformar el estado al formato de entrada de la red y elegir la acción con mayor valor Q predicho.

### B.3 Probar el agente neuronal

Ejecuta el script de testeo usando el agente neuronal:

```bash
python test_agent.py --agent agentes.nn_agent.NNAgent
```

**Nota:** Asegúrate de que el modelo guardado esté disponible en la ruta esperada (`flappy_q_nn_model.keras/` por defecto).

---

## Notas Adicionales

- El entorno está configurado para FlappyBird por defecto.
- Los agentes reciben la instancia del juego y la lista de acciones posibles al inicializarse.
- Para agregar un nuevo agente, crea un archivo en `agentes/` y define una clase que herede de `Agent`.
- Puedes crear tus propios agentes en la carpeta `agentes/` siguiendo la interfaz de la clase base.

---

## Análisis y Conclusiones

Los resultados detallados, análisis de la ingeniería de características (discretización) y comparación entre ambos enfoques se encuentran en el archivo **[conclusiones.md](./conclusiones.md)**.