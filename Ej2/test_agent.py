from ple.games.flappybird import FlappyBird
from ple import PLE
import time
import argparse
import importlib
import sys

# --- Configuración del Entorno y Agente ---
# Inicializar el juego
game = FlappyBird()  # Usar FlappyBird en vez de Pong
env = PLE(game, display_screen=True, fps=30) # fps=30 es más normal, display_screen=True para ver

# ---- SEMILLA ALEATORIA ----
import random
import numpy as np
seed = int(time.time())
random.seed(seed)
np.random.seed(seed)
env.rng.seed(seed)     # seed del entorno PLE
# ---------------------------

# Inicializar el entorno
env.init()

# Obtener acciones posibles
actions = env.getActionSet()

# --- Argumentos ---
parser = argparse.ArgumentParser(description="Test de agentes para FlappyBird (PLE)")
parser.add_argument('--agent', type=str, required=True, help='Ruta completa del agente, ej: agentes.random_agent.RandomAgent')
args = parser.parse_args()

# --- Carga dinámica del agente usando path completo ---
try:
    module_path, class_name = args.agent.rsplit('.', 1)
    agent_module = importlib.import_module(module_path)
    AgentClass = getattr(agent_module, class_name)
except (ValueError, ModuleNotFoundError, AttributeError):
    print(f"No se pudo encontrar la clase {args.agent}")
    sys.exit(1)

# Inicializar el agente
agent = AgentClass(actions, game)

# Agente con acciones aleatorias
while True:
    env.reset_game()
    agent.reset()
    state_dict = env.getGameState()
    done = False
    total_reward_episode = 0

    last_print = time.time()

    print("\n--- Ejecutando agente ---")
    while not done:
        action = agent.act(state_dict)
        reward = env.act(action)
        state_dict = env.getGameState()
        done = env.game_over()
        total_reward_episode += reward

        # ---- PRINT CADA 10 SEGUNDOS ----
        if time.time() - last_print >= 30:
            print(f"[{int(time.time() - last_print)}s] Reward parcial: {total_reward_episode}")
            last_print = time.time()
        time.sleep(0.03)
    print(f"Recompensa episodio: {total_reward_episode}")
