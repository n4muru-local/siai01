import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pygame
import matplotlib.pyplot as plt
from tqdm import tqdm

print(torch.__file__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使うのはこれなのだ：", device)

# 定数設定
GRID_SIZE = 100
NUM_AGENTS = 5
NUM_TARGETS = 30
OBSERVATION_RANGE = 5
CELL_SIZE = 8  # Pygame描画用セルサイズ
MOVE_DELTAS = [(-1,0), (-1,1), (0,1), (1,1),
               (1,0), (1,-1), (0,-1), (-1,-1)]
SCAN_DEPTH = 2   # 奥2マス
SCAN_WIDTH = 5   # 横5マス（左右2ずつ＋中心）

# DQNネットワーク
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Liner(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

# 経験リプレイバッファ
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    def __len__(self):
        return len(self.buffer)

# 学習用エージェントクラス
class SwarmAgent:
    def __init__(self, input_dim, action_dim, lr=0.001):
        self.device = device
        self.policy_net = DQN(input_dim, action_dim).to(self.device)
        self.target_net = DQN(input_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.action_dim = action_dim
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()
    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            expected_state_action_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 環境クラス（学習・テスト共通）
class SwarmEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.num_agents = NUM_AGENTS
        self.num_targets = NUM_TARGETS
        self.agents = []
        self.targets = []
        self.steps = 0
    def reset(self):
        self.steps = 0
        self.agents = []
        center = (self.grid_size // 2, self.grid_size // 2)
        for _ in range(self.num_agents):
            ori = random.randint(0, 3)
            self.agents.append({'pos': center, 'ori': ori})
        self.targets = []
        for _ in range(self.num_targets):
            t_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            self.targets.append(t_pos)
        return self.get_observations()
    def get_observations(self):
        obs = []
        for idx in range(self.num_agents):
            obs.append(self.get_agent_observation(idx))
        return obs
    def get_agent_observation(self, idx):
        agent = self.agents[idx]
        ax, ay = agent['pos']
        grid = np.zeros((OBSERVATION_RANGE*2+1, OBSERVATION_RANGE*2+1))
        for j, other in enumerate(self.agents):
            if j == idx: continue
            ox, oy = other['pos']
            if abs(ox - ax) <= OBSERVATION_RANGE and abs(oy - ay) <= OBSERVATION_RANGE:
                grid[ox - ax + OBSERVATION_RANGE, oy - ay + OBSERVATION_RANGE] = 1
        return grid.flatten()
    def get_scan_area(self, agent):
        # エージェントの視界（横SCAN_WIDTH, 奥SCAN_DEPTH）を計算するのよ
        x, y = agent['pos']
        ori = agent['ori']
        scan_coords = []
        half_width = (SCAN_WIDTH - 1) // 2
        if ori == 0:  # 上向き
            for d in range(1, SCAN_DEPTH+1):
                for offset in range(-half_width, half_width+1):
                    scan_coords.append((x - d, y + offset))
        elif ori == 1:  # 右向き
            for d in range(1, SCAN_DEPTH+1):
                for offset in range(-half_width, half_width+1):
                    scan_coords.append((x + offset, y + d))
        elif ori == 2:  # 下向き
            for d in range(1, SCAN_DEPTH+1):
                for offset in range(-half_width, half_width+1):
                    scan_coords.append((x + d, y + offset))
        elif ori == 3:  # 左向き
            for d in range(1, SCAN_DEPTH+1):
                for offset in range(-half_width, half_width+1):
                    scan_coords.append((x + offset, y - d))
        scan_coords = [(i, j) for (i, j) in scan_coords if 0 <= i < self.grid_size and 0 <= j < self.grid_size]
        return scan_coords
    def step(self, actions):
        self.steps += 1
        for i, action in enumerate(actions):
            agent = self.agents[i]
            if action < 8:
                dx, dy = MOVE_DELTAS[action]
                new_x = max(0, min(self.grid_size-1, agent['pos'][0] + dx))
                new_y = max(0, min(self.grid_size-1, agent['pos'][1] + dy))
                agent['pos'] = (new_x, new_y)
            elif action == 8:
                agent['ori'] = (agent['ori'] - 1) % 4
            elif action == 9:
                agent['ori'] = (agent['ori'] + 1) % 4
        rewards = [ -0.1 - 0.001 * self.steps for _ in range(self.num_agents) ]
        bonus = 10.0
        removed_count = 0
        # ターゲットがエージェントの視界内にあれば消去するのよ
        for target in list(self.targets):
            for agent in self.agents:
                if target in self.get_scan_area(agent):
                    self.targets.remove(target)
                    removed_count += 1
                    break
        for i in range(self.num_agents):
            rewards[i] += bonus * removed_count
        done = (len(self.targets) == 0)
        return self.get_observations(), rewards, done

# 学習・モデル保存の関数
def train():
    num_episodes = 200    # 総エピソード数
    max_steps = 200       # 1エピソードあたりの最大ステップ数
    batch_size = 32       # ミニバッチのサイズ
    target_update_interval = 10  # ターゲットネットワーク更新間隔
    action_dim = 10       # エージェントの行動数
    input_dim = (OBSERVATION_RANGE*2+1)**2
    agents = [SwarmAgent(input_dim, action_dim) for _ in range(NUM_AGENTS)]
    env = SwarmEnv()
    episode_rewards = []
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        obs = env.reset()
        total_reward = 0
        for step in range(max_steps):
            actions = []
            for i in range(NUM_AGENTS):
                action = agents[i].select_action(obs[i])
                actions.append(action)
            next_obs, rewards, done = env.step(actions)
            total_reward += sum(rewards)
            for i in range(NUM_AGENTS):
                agents[i].memory.push(obs[i], actions[i], rewards[i], next_obs[i], done)
            obs = next_obs
            for i in range(NUM_AGENTS):
                agents[i].update(batch_size)
            if done:
                break
        episode_rewards.append(total_reward)
        if episode % target_update_interval == 0:
            for agent in agents:
                agent.update_target()
    plt.figure()
    plt.plot(episode_rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.savefig("training_progress.png")
    plt.show()
    for i, agent in enumerate(agents):
        torch.save(agent.policy_net.state_dict(), f"agent_{i}.pth")
    print("Training completed and models saved.")

# テスト用エージェントクラス
class TestAgent:
    def __init__(self, input_dim, action_dim, model_path):
        self.device = device
        self.policy_net = DQN(input_dim, action_dim).to(self.device)
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()
        self.action_dim = action_dim
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

# Pygameテスト環境
def test():
    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
    pygame.display.set_caption("Test Simulation with Learned Models")
    clock = pygame.time.Clock()
    env = SwarmEnv()
    observations = env.reset()
    input_dim = (OBSERVATION_RANGE*2+1)**2
    action_dim = 10
    test_agents = [TestAgent(input_dim, action_dim, f"agent_{i}.pth") for i in range(NUM_AGENTS)]
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        actions = []
        for i, agent in enumerate(test_agents):
            action = agent.select_action(observations[i])
            actions.append(action)
        observations, rewards, done = env.step(actions)
        screen.fill((255, 255, 255))
        for x in range(0, GRID_SIZE * CELL_SIZE, CELL_SIZE):
            pygame.draw.line(screen, (200,200,200), (x, 0), (x, GRID_SIZE * CELL_SIZE))
        for y in range(0, GRID_SIZE * CELL_SIZE, CELL_SIZE):
            pygame.draw.line(screen, (200,200,200), (0, y), (GRID_SIZE * CELL_SIZE, y))
        for t in env.targets:
            rect = pygame.Rect(t[1]*CELL_SIZE, t[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (255, 0, 0), rect)
        for agent in env.agents:
            pos = agent['pos']
            center = (pos[1]*CELL_SIZE + CELL_SIZE//2, pos[0]*CELL_SIZE + CELL_SIZE//2)
            pygame.draw.circle(screen, (0,0,255), center, CELL_SIZE//2)
            ori = agent['ori']
            if ori == 0:
                offset = (0, -CELL_SIZE//2)
            elif ori == 1:
                offset = (CELL_SIZE//2, 0)
            elif ori == 2:
                offset = (0, CELL_SIZE//2)
            elif ori == 3:
                offset = (-CELL_SIZE//2, 0)
            end_pos = (center[0]+offset[0], center[1]+offset[1])
            pygame.draw.line(screen, (0,0,0), center, end_pos, 2)
        pygame.display.flip()
        clock.tick(5)
        if done:
            print("All targets discovered. Ending test simulation.")
            break
    pygame.quit()

if __name__ == "__main__":
    print("Training開始なのだ")
    train()
    print("Test実行するのだ")
    test()
