# Ejercicio 1: Audio MNIST

## Descripción

Este ejercicio consiste en construir modelos de clasificación para reconocer dígitos hablados (0-9) a partir de clips de audio utilizando redes neuronales.

### Dataset

- **Nombre:** Spoken Digit Dataset
- **Fuente:** TensorFlow Datasets
- **Contenido:** 2500 clips de audio
  - 5 locutores diferentes
  - 50 clips por dígito por locutor

### Modelos Implementados

1. **Modelo Convolucional (CNN):** Red neuronal convolucional aplicada sobre espectrogramas de los clips de audio
2. **Modelo Recurrente (RNN):** Red neuronal recurrente aplicada sobre espectrogramas de los clips de audio
3. **Modelo Híbrido (CNN+RNN):** Arquitectura combinada que aprovecha las ventajas de ambos enfoques

### Evaluación Adicional

Se incluye un dataset de test personalizado con 10 audios (uno para cada dígito) grabados por cada integrante del grupo para evaluar la generalización de los modelos.

---

## Estructura del Proyecto

```
Ej1/
├── audio_grupo/               # Audios personalizados del grupo
│   ├── fabricio/              # 10 audios (0-9) de Fabricio
│   └── maria/                 # 10 audios (0-9) de María
├── audio_mnist.ipynb          # Notebook principal con todo el código
├── Comparacion modelos.png    # Gráfico comparativo de resultados
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Este archivo
```

---

## Instalación y Configuración

### 1. Navegar a la carpeta del ejercicio

```bash
cd Ej1
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

## Ejecución

Para ejecutar el proyecto, abre el notebook con Jupyter:

```bash
jupyter notebook audio_mnist.ipynb
```

O si prefieres usar JupyterLab:

```bash
jupyter lab audio_mnist.ipynb
```

---

## Contenido del Notebook

El notebook `audio_mnist.ipynb` está organizado en los siguientes bloques:

1. **Librerías:** Importación de todas las dependencias necesarias
2. **Dataset:** Carga del dataset Spoken Digit de TensorFlow Datasets
3. **Análisis exploratorio y visualizaciones:** Exploración inicial de los datos
4. **Preparación del dataset:** Preprocesamiento y generación de espectrogramas
5. **Modelo Convolucional (CNN):** Implementación y entrenamiento de red convolucional
6. **Modelo Recurrente (RNN):** Implementación y entrenamiento de red recurrente
7. **Modelo Híbrido (CNN+RNN):** Implementación y entrenamiento de arquitectura combinada
8. **Comparación de modelos:** Análisis comparativo de las tres arquitecturas
9. **Carga del dataset personalizado:** Carga de audios del grupo
10. **Evaluación de audios personalizados:** Testing con audios de Fabricio y María

---

## Resultados

Los resultados y métricas de evaluación se documentan en el notebook, incluyendo:

- **Accuracy** de cada modelo
- **Matrices de confusión**
- **Curvas de aprendizaje** (loss y accuracy)
- **Comparación entre arquitecturas** (visualizada en `Comparacion modelos.png`)
- **Evaluación con audios personalizados** del grupo