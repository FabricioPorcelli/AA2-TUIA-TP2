from agentes.base import Agent
import numpy as np
import tensorflow as tf

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model.keras'):
        super().__init__(actions, game)
        self.model = tf.keras.models.load_model(model_path)
        print(f"Modelo cargado desde {model_path}")
        
        # Compilar la función de inferencia (se ejecuta una sola vez)
        @tf.function
        def predict_fast(input_tensor):
            return self.model(input_tensor, training=False)
        
        self.predict_fn = predict_fast
        
        # Warm-up: ejecutar una vez para compilar
        dummy_input = tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
        self.predict_fn(dummy_input)
        print("Modelo compilado para inferencia rápida")

    def discretize_state(self, state):
        # Convertir a float
        player_y = float(state["player_y"])
        player_vel = float(state["player_vel"])
        d1 = float(state["next_pipe_dist_to_player"])
        t1 = float(state["next_pipe_top_y"])
        b1 = float(state["next_pipe_bottom_y"])
        d2 = float(state["next_next_pipe_dist_to_player"])
        t2 = float(state["next_next_pipe_top_y"])
        b2 = float(state["next_next_pipe_bottom_y"])

        # --------------------------------------------------------
        #   Feature engineering
        # --------------------------------------------------------
        
        # cálculo del "gap" (centro del tubo)
        gap1 = (t1 + b1) / 2 # next pipe
        gap2 = (t2 + b2) / 2 # next next pipe

        # diferencia vertical entre el pájaro y el agujero
        delta_y = player_y - gap1

        # tendencia del próximo tubo
        pipe_trend = gap2 - gap1

        # rangos del entorno Flappy Bird (PLE)
        ranges = {
            "dist": (0, 300),         # horizontal distance
            "delta_y": (-200, 200),   # vertical misalignment
            "vel": (-10, 10),         # velocity
            "trend": (-150, 150)      # movement of the next pipe
        }

        # cantidad de bins por variable (modificable)
        bins = {
            "dist": 10,
            "delta_y": 10,
            "vel": 5,
            "trend": 7
        }

        # Cantidad de estados: 10x10x5x7 = 3500

        # --------------------------------------------------------
        #   Función para calcular bins uniformes
        # --------------------------------------------------------
        def make_bin(x, low, high, n_bins):
            x_clipped = max(low, min(x, high))
            bin_width = (high - low) / n_bins
            b = int((x_clipped - low) // bin_width)
            return min(b, n_bins - 1)

        # --------------------------------------------------------
        # Discretización con parámetros
        # --------------------------------------------------------
        dist_bin = make_bin(d1, *ranges["dist"], bins["dist"])              # tiempo hasta colisión
        dy_bin = make_bin(delta_y, *ranges["delta_y"], bins["delta_y"])     # alineación vertical
        vel_bin = make_bin(player_vel, *ranges["vel"], bins["vel"])         # tendencia de movimiento
        trend_bin = make_bin(pipe_trend, *ranges["trend"], bins["trend"])   # anticipación del siguiente tubo

        # Estado final reducido:
        return (dist_bin, dy_bin, vel_bin, trend_bin)

    def act(self, state):
        """
        Elige una acción usando la red neuronal
        """
        # Discretizar
        discrete_state = self.discretize_state(state)
        
        # Convertir a tensor
        state_tensor = tf.constant([discrete_state], dtype=tf.float32)
        
        # Inferencia compilada
        q_values = self.predict_fn(state_tensor)[0].numpy()
        
        # Elegir mejor acción
        best_action_idx = np.argmax(q_values)
        
        return self.actions[best_action_idx]