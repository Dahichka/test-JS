import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ДОДАЙ ЦІ ДВА РЯДКИ ДЛЯ КАРДИНАЛЬНОГО ПРИСКОРЕННЯ НА CPU ---
if device.type == "cpu":
    torch.set_num_threads(6)  # Використовуємо 6 ядер паралельно
    torch.set_num_interop_threads(6)

NUM_CATEGORIES = 13  # 0 (порожньо) + степені двійки від 2^1 до 2^12 (4096)


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
        onehot = np.zeros((16, NUM_CATEGORIES), dtype=np.float32)
        flat = self.board.flatten()
        for i, v in enumerate(flat):
            exponent = 0 if v == 0 else int(np.log2(v))
            exponent = min(exponent, NUM_CATEGORIES - 1)
            onehot[i, exponent] = 1.0
        return onehot.flatten()

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

    def _simulate_action(self, board, action):
        b = board.copy()
        points = 0
        if action == 2:  # LEFT
            for i in range(4):
                b[i], p = self._merge_line(b[i])
                points += p
        elif action == 3:  # RIGHT
            for i in range(4):
                flipped, p = self._merge_line(b[i][::-1])
                b[i] = flipped[::-1]
                points += p
        elif action == 0:  # UP
            for j in range(4):
                col, p = self._merge_line(b[:, j])
                b[:, j] = col
                points += p
        elif action == 1:  # DOWN
            for j in range(4):
                flipped, p = self._merge_line(b[:, j][::-1])
                b[:, j] = flipped[::-1]
                points += p
        moved = not np.array_equal(board, b)
        return b, points, moved

    def valid_actions(self):
        valid = []
        for a in range(4):
            _, _, moved = self._simulate_action(self.board, a)
            if moved:
                valid.append(a)
        return valid

    def step(self, action):
        """Оптимізована версія step без важких циклів Python"""
        new_board, points, moved = self._simulate_action(self.board, action)

        if not moved:
            return self.get_state(), -10.0, False

        self.board = new_board
        self.spawn_tile()
        self.score += points

        # --- ОПТИМІЗОВАНА СИСТЕМА НАГОРОД ---
        reward = float(points) * 1.5

        # 1. Бонус за вільне місце
        empty_cells_count = np.sum(self.board == 0)
        reward += float(empty_cells_count) * 5.0

        # 2. Матриця змійки (Векторизований NumPy розрахунок)
        snake_weights = np.array([
            [0,  1,  2,  3],
            [7,  6,  5,  4],
            [8,  9,  10, 11],
            [15, 14, 13, 12]
        ], dtype=np.float32)
        
        # Рахуємо логарифми для всього поля миттєво
        with np.errstate(divide='ignore'):
            log_board = np.where(self.board > 0, np.log2(self.board), 0.0)
        reward += float(np.sum(log_board * snake_weights)) * 2.0

        # 3. Умови завершення
        done = False
        if np.any(self.board == 2048):
            reward += 10000.0
            done = True
            return self.get_state(), reward, done

        if not self.any_move_possible():
            done = True
            reward -= 500.0

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

    def __init__(self, input_dim=16 * NUM_CATEGORIES, output_dim=4):
        super(DQNNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),  # Збільшили кількість нейронів з 512 до 1024
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),         # Додали ще один потужний шар на 512 нейронів
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


# =====================================================================
# 3. DQN АГЕНТ (ШТУЧНИЙ ІНТЕЛЕКТ)
# =====================================================================
class DQNAgent:

    def __init__(self, state_dim=16 * NUM_CATEGORIES, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ПРОКАЧКА: Пам'ять збільшено до 100 000 ходів для кращого досвіду
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.02
        
        # ПРОКАЧКА: Уповільнюємо згасання. Тепер він вчитиметься вглиб набагато довше!
        self.epsilon_decay = 0.9999  

        self.batch_size = 64

        self.model = DQNNet(state_dim, action_dim).to(device)
        self.target_model = DQNNet(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Трохи зменшимо швидкість навчання (lr), щоб більша мережа вчилася стабільніше і без різких зривів
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.criterion = nn.MSELoss()

    def choose_action(self, state, valid_actions):
        candidates = valid_actions if valid_actions else list(range(self.action_dim))

        if np.random.rand() <= self.epsilon:
            return random.choice(candidates)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_t).squeeze(0).cpu().numpy()
        best = max(candidates, key=lambda a: q_values[a])
        return best

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(device)

        current_q = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            max_next_q = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = self.criterion(current_q, target_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def evaluate_greedy(agent, episodes=5):
    sim = Game2048Sim()
    scores = []
    max_tiles = []
    reached_2048 = 0
    for _ in range(episodes):
        state = sim.reset()
        done = False
        steps = 0
        while not done and steps < 5000:
            valid = sim.valid_actions()
            if not valid:
                break
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent.model(state_t).squeeze(0).cpu().numpy()
            action = max(valid, key=lambda a: q_values[a])
            state, _, done = sim.step(action)
            steps += 1
        scores.append(sim.score)
        max_tiles.append(int(sim.board.max()))
        if np.any(sim.board == 2048):
            reached_2048 += 1
    return {
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "avg_max_tile": sum(max_tiles) / len(max_tiles),
        "reached_2048_rate": reached_2048 / episodes,
    }


# =====================================================================
# 5. ГОЛОВНИЙ ЦИКЛ НАВЧАННЯ ШІ
# =====================================================================
if __name__ == "__main__":
    import os
    import re

    sim = Game2048Sim()
    agent = DQNAgent()

    start_episode = 1
    checkpoint_dir = "."
    
    pth_files = [f for f in os.listdir(checkpoint_dir) if re.match(r"ai_2048_brain_v2_ep\d+\.pth", f)]

    if pth_files:
        episodes_nums = [int(re.search(r"ep(\d+)\.pth", f).group(1)) for f in pth_files]
        latest_idx = np.argmax(episodes_nums)
        last_saved_episode = episodes_nums[latest_idx]
        latest_file = pth_files[latest_idx]
        
        print(f"[🔄] Знайдено збережений прогрес: {latest_file}")
        agent.model.load_state_dict(torch.load(latest_file, map_location=device))
        agent.target_model.load_state_dict(agent.model.state_dict())
        
        start_episode = last_saved_episode + 1
        calculated_epsilon = 1.0 * (agent.epsilon_decay ** last_saved_episode)
        agent.epsilon = max(agent.epsilon_min, calculated_epsilon)
        
        print(f"[🚀] Успішно відновлено! Починаємо з епізоду: {start_episode}")
        print(f"[📊] Поточний адаптований Epsilon: {agent.epsilon:.4f}")
    else:
        print("[🆕] Збережень не знайдено. Починаємо навчання абсолютно з нуля.")

    episodes = 100000  
    best_score = 0
    start_time = time.time()

    print(f"[🤖] Старт тренування на пристрої: {device}")
    print("[!] Мета ШІ: максимум очок, 2048 наприкінці — це ціль, а не заборона!\n")

    for e in range(start_episode, episodes + 1):
        state = sim.reset()
        done = False
        
        # --- ТУТ МИ РОБИМО РОЗГІН НАВЧАННЯ ---
        step_counter = 0  # Створюємо лічильник для поточної гри

        while not done:
            valid = sim.valid_actions()
            if not valid:
                break
            action = agent.choose_action(state, valid)
            next_state, reward, done = sim.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            step_counter += 1
            # Замість кожного кроку викликаємо важкий replay() раз на 4 кроки
            if step_counter % 4 == 0:
                agent.replay()

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if e % 100 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        if sim.score > best_score:
            best_score = sim.score

        if e % 50 == 0:
            max_tile = np.max(sim.board)
            elapsed = time.time() - start_time
            print(
                f"Гра: {e:6d}/{episodes} | Рекорд: {best_score:6d} | Рахунок: {sim.score:6d} | "
                f"Max плитка: {max_tile:4d} | Epsilon: {agent.epsilon:.3f} | Час: {elapsed:.1f}с"
            )

        if e % 1000 == 0:
            stats = evaluate_greedy(agent, episodes=5)
            print(
                f"  [📊 ЖАДІБНА ОЦІНКА] середній рахунок={stats['avg_score']:.0f} "
                f"макс={stats['max_score']} серед. макс.плитка={stats['avg_max_tile']:.0f} "
                f"частка ігор із 2048={stats['reached_2048_rate']*100:.0f}%"
            )

        if e % 2000 == 0:
            torch.save(agent.model.state_dict(), f"ai_2048_brain_v2_ep{e}.pth")
            print(f"--- [💾 ЗБЕРЕЖЕНО] ai_2048_brain_v2_ep{e}.pth записано на диск. ---")