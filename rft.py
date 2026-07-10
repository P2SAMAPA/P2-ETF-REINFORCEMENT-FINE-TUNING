import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class PolicyNetwork(nn.Module):
    """
    Policy network for RL fine-tuning.
    Outputs: action (predicted return) and value (state value).
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.actor = nn.Linear(hidden_size, 1)  # action: predicted return
        self.critic = nn.Linear(hidden_size, 1)  # value: state value

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]
        action = self.actor(h).squeeze(-1)
        value = self.critic(h).squeeze(-1)
        return action, value

class RFTAgent:
    """
    Reinforcement Fine-Tuning Agent using PPO.
    Reward = portfolio return (task-performance based).
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, lr=0.0001):
        self.policy = PolicyNetwork(input_size, hidden_size, num_layers)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = deque(maxlen=1000)

    def get_action(self, state):
        self.policy.eval()
        with torch.no_grad():
            action, value = self.policy(state)
        return action.item(), value.item()

    def store_transition(self, state, action, reward, value):
        self.memory.append((state, action, reward, value))

    def train_step(self, batch_size=8, clip=0.2, epochs=5):
        if len(self.memory) < batch_size:
            return 0.0
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states = torch.cat([s for s, _, _, _ in batch], dim=0)
        actions = torch.tensor([a for _, a, _, _ in batch], dtype=torch.float32)
        rewards = torch.tensor([r for _, _, r, _ in batch], dtype=torch.float32)
        values = torch.tensor([v for _, _, _, v in batch], dtype=torch.float32)
        # Compute advantages
        advantages = rewards - values
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # PPO update
        for _ in range(epochs):
            action_pred, value_pred = self.policy(states)
            # Policy loss
            ratio = torch.exp(action_pred - actions)  # log prob ratio approximation
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-clip, 1+clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            # Value loss
            value_loss = F.mse_loss(value_pred, rewards)
            # Entropy bonus (for exploration)
            entropy = -torch.exp(action_pred) * action_pred.mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

def prepare_data(returns, macro_df, seq_len=10):
    """
    Prepare sequences for training.
    returns: pandas Series (single ETF)
    macro_df: pandas DataFrame (macro variables)
    """
    if len(returns) < seq_len + 1:
        return None, None
    common_idx = returns.index.intersection(macro_df.index)
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    X, y = [], []
    for i in range(seq_len, len(ret_aligned)):
        ret_seq = ret_aligned.iloc[i-seq_len:i].values.reshape(-1, 1)
        macro_seq = macro_aligned.iloc[i-seq_len:i].values
        seq_features = np.concatenate([ret_seq, macro_seq], axis=1)
        X.append(seq_features)
        y.append(ret_aligned.iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def rft_score(returns, macro_df, hidden_size=64, num_layers=2, seq_len=10, pretrain_epochs=20, rl_epochs=10, lr=0.001, batch_size=16, rl_lr=0.0001, rl_batch=8, ppo_clip=0.2, ppo_epochs=5):
    """
    Train RFT agent and return predicted next-day return with momentum enhancement.
    """
    X, y = prepare_data(returns, macro_df, seq_len)
    if X is None or len(X) < batch_size:
        return 0.0
    input_size = X.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Pretrain the policy network with supervised learning
    pretrain_model = PolicyNetwork(input_size, hidden_size, num_layers).to(device)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    pretrain_model.train()
    for epoch in range(pretrain_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            action, _ = pretrain_model(X_batch)
            loss = criterion(action, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    # Load pretrained weights into RL agent
    agent = RFTAgent(input_size, hidden_size, num_layers, lr=rl_lr)
    agent.policy.load_state_dict(pretrain_model.state_dict())
    agent.policy.to(device)
    # RL fine-tuning: use portfolio return as reward
    # For each episode, we simulate a portfolio using the model's predictions
    # Reward = cumulative return of the portfolio over the episode
    # We'll use the training data for RL episodes
    n_episodes = len(X) // seq_len
    for episode in range(min(rl_epochs, max(1, n_episodes))):
        episode_start = episode * seq_len
        episode_end = min((episode + 1) * seq_len, len(X))
        if episode_end - episode_start < seq_len:
            break
        episode_X = X[episode_start:episode_end]
        episode_y = y[episode_start:episode_end]
        # Rollout: get actions for the episode
        actions = []
        values = []
        states = []
        with torch.no_grad():
            for t in range(len(episode_X)):
                state = torch.tensor(episode_X[t], dtype=torch.float32).unsqueeze(0).to(device)
                action, value = agent.policy(state)
                actions.append(action.item())
                values.append(value.item())
                states.append(state)
        # Compute reward: portfolio return using the predictions
        # Simple: take long position if action > 0, short if action < 0
        # Reward = sum of returns weighted by position
        returns_ep = episode_y
        positions = np.sign(actions)
        portfolio_return = np.mean(positions * returns_ep)
        reward = portfolio_return  # task-performance reward
        # Store transitions
        for t in range(len(states)):
            agent.store_transition(states[t], actions[t], reward / len(states), values[t])
        # Train the agent on the episode
        loss = agent.train_step(batch_size=rl_batch, clip=ppo_clip, epochs=ppo_epochs)
    # Predict next day
    agent.policy.eval()
    with torch.no_grad():
        ret_seq = returns.iloc[-seq_len:].values.reshape(-1, 1)
        macro_seq = macro_df.iloc[-seq_len:].values
        last_seq = np.concatenate([ret_seq, macro_seq], axis=1)
        last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred, _ = agent.policy(last_seq)
        pred = pred.item()
    # Momentum factor
    last_return = returns.iloc[-1]
    momentum = 1.0 + last_return
    momentum = max(0.5, min(2.0, momentum))
    final_score = pred * momentum
    return float(final_score)
