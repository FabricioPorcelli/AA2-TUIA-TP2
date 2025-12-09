from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle

class QAgent(Agent):
    """
    Agente de Q-Learning con discretización orientada a decisiones.
    Enfocado en: ¿Necesito subir YA o debo caer?
    """
    def __init__(self, actions, game=None, learning_rate=0.1, discount_factor=0.99,
                 epsilon=0.0, epsilon_decay=0.995, min_epsilon=0.01, load_q_table_path="flappy_birds_q_table.pkl"):
        super().__init__(actions, game)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        if load_q_table_path:
            try:
                with open(load_q_table_path, 'rb') as f:
                    q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
                print(f"Q-table cargada desde {load_q_table_path}")
            except FileNotFoundError:
                print(f"Archivo Q-table no encontrado en {load_q_table_path}. Se inicia una nueva Q-table vacía.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
   
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
        Política epsilon-greedy para selección de acciones.
        """
        discrete_state = self.discretize_state(state)
        
        if np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return np.random.choice(self.actions)
        else:
            # Explotación: mejor acción según Q-table
            q_values = self.q_table[discrete_state]
            best_action_idx = np.argmax(q_values)
            return self.actions[best_action_idx]

    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la regla de Q-learning.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        # Inicializar si el estado no está en la Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        """
        Disminuye epsilon para reducir la exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """
        Guarda la Q-table en un archivo usando pickle.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardada en {path}")

    def load_q_table(self, path):
        """
        Carga la Q-table desde un archivo usando pickle.
        """
        import pickle
        try:
            with open(path, 'rb') as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde {path}")
        except FileNotFoundError:
            print(f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))