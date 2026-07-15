import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Налаштування пристрою обчислень
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 1. СИМУЛЯТОР ГРИ 2048
# =====================================================================
class Game2048Sim:

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.spawn_tile()
        self.spawn_tile()
        return self.get_state()

    def get_state(self):
        # Перетворюємо матрицю у плаский масив для нейромережі
        return self.board.flatten()

    def spawn_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            r, c = empty_cells[np.random.choice(len(empty_cells))]
            self.board[r, c] = 4 if np.random.rand() < 0.1 else 2

    def _toggle_row(self, row):
        non_zero = row[row != 0]
        new_row = np.zeros(4, dtype=int)
        new_row[: len(non_zero)] = non_zero
        return new_row

    def _merge_line(self, line):
        line = self._toggle_row(line)
        points = 0
        for i in range(3):
            if line[i] != 0 and line[i] == line[i + 1]:
                line[i] *= 2
                points += line[i]
                line[i + 1] = 0
        line = self._toggle_row(line)
        return line, points

    def step(self, action):
        """0=UP, 1=DOWN, 2=LEFT, 3=RIGHT"""
        old_board = self.board.copy()
        points = 0

        if action == 2:  # LEFT
            for i in range(4):
                self.board[i], p = self._merge_line(self.board[i])
                points += p
        elif action == 3:  # RIGHT
            for i in range(4):
                flipped, p = self._merge_line(self.board[i][::-1])
                self.board[i] = flipped[::-1]
                points += p
        elif action == 0:  # UP
            for j in range(4):
                col, p = self._merge_line(self.board[:, j])
                self.board[:, j] = col
                points += p
        elif action == 1:  # DOWN
            for j in range(4):
                flipped, p = self._merge_line(self.board[:, j][::-1])
                self.board[:, j] = flipped[::-1]
                points += p

        moved = not np.array_equal(old_board, self.board)
        if moved:
            self.spawn_tile()
            self.score += points

        done = False

        # --- ОНОВЛЕНА СИСТЕМА НАГОРОД ДЛЯ ПРОБИТТЯ ПЛАТО ---
        reward = points * 2

        if np.any(self.board == 2048):
            reward = -2000
            done = True
            return self.get_state(), reward, done

        if not self.any_move_possible():
            done = True
            reward -= 500

        empty_cells_count = np.sum(self.board == 0)
        reward += empty_cells_count * 5

        return self.get_state(), reward, done

    def any_move_possible(self):
        if np.any(self.board == 0):
            return True
        for i in range(4):
            for j in range(3):
                if (
                    self.board[i, j] == self.board[i, j + 1]
                    or self.board[j, i] == self.board[j + 1, i]
                ):
                    return True
        return False


# =====================================================================
# 2. АРХІТЕКТУРА НЕЙРОМЕРЕЖІ (DQN)
# =====================================================================
class DQNNet(nn.Module):

    def __init__(self, input_dim=16, output_dim=4):
        super(DQNNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


# =====================================================================
# 3. DQN АГЕНТ (ШТУЧНИЙ ІНТЕЛЕКТ)
# =====================================================================
class DQNAgent:

    def __init__(self, state_dim=16, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.memory = deque(maxlen=20000)
        self.gamma = 0.99  # Коефіцієнт дисконтування

        # Параметри дослідження (Exploration vs Exploitation)
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9997

        self.batch_size = 64

        # Основна та цільова нейромережі
        self.model = DQNNet(state_dim, action_dim).to(device)
        self.target_model = DQNNet(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.MSELoss()

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)

        state_t = (
            torch.FloatTensor(state).unsqueeze(0).to(device)
        )  # Нормалізація не потрібна для сирого стану
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(
            np.array([t[0] for t in minibatch])
        ).to(device)
        actions = (
            torch.LongTensor(np.array([t[1] for t in minibatch]))
            .unsqueeze(1)
            .to(device)
        )
        rewards = torch.FloatTensor(
            np.array([t[2] for t in minibatch])
        ).to(device)
        next_states = torch.FloatTensor(
            np.array([t[3] for t in minibatch])
        ).to(device)
        dones = torch.FloatTensor(
            np.array([t[4] for t in minibatch])
        ).to(device)

        # Оцінка поточних Q-значень за допомогою основної мережі
        current_q = self.model(states).gather(1, actions)

        # Обчислення максимальних майбутніх Q-значень за допомогою стабільної цільової мережі
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = self.criterion(current_q, target_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# =====================================================================
# 4. ГОЛОВНИЙ ЦИКЛ НАВЧАННЯ ШІ
# =====================================================================
if __name__ == "__main__":
    sim = Game2048Sim()
    agent = DQNAgent()

    episodes = 30000
    best_score = 0
    start_time = time.time()

    print(f"[🤖] Старт тренування на пристрої: {device}")
    print("[!] Мета ШІ: Набивати бали, але уникати створення плит 2048!\n")

    for e in range(1, episodes + 1):
        state = sim.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = sim.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()

        # Зменшуємо випадковість ОДИН раз за гру (епізод)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # Оновлення стабільної цільової мережі
        if e % 100 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        if sim.score > best_score:
            best_score = sim.score

        # Виведення логів кожні 50 ігор
        if e % 50 == 0:
            max_tile = np.max(sim.board)
            elapsed = time.time() - start_time
            print(
                f"Ігра: {e:5d}/{episodes} | Рекорд: {best_score:5d} | Поточний рахунок: {sim.score:5d} | Max плитка: {max_tile:4d} | Epsilon: {agent.epsilon:.3f} | Час: {elapsed:.1f}с"
            )

        # Збереження мізків бота кожні 2000 ігор
        if e % 2000 == 0:
            torch.save(agent.model.state_dict(), f"ai_2048_brain_ep{e}.pth")
            print(
                f"--- [💾 ЗБЕРЕЖЕНО МУТАЦІЮ] Файл ai_2048_brain_ep{e}.pth записано на диск. ---"
            )