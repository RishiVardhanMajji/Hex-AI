import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import numpy as np
from tqdm import tqdm
import copy
import random

from engine import HexGame
from model import HexZeroBrain
from mcts import MCTS

# --- Configuration ---
CONFIG = {
    'board_size': 5,
    'num_simulations': 50,
    'num_games': 20,
    'num_epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
    'replay_buffer_size': 10000,
    'num_iterations': 10,
    'c_puct': 1.0,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def self_play(model, board_size, num_simulations, c_puct, device):
    game = HexGame(board_size)
    mcts = MCTS(model, num_simulations, c_puct, device)
    game_history = []
    while game.get_game_status() == 0:
        root = mcts.search_for_training(game)
        policy_target = np.zeros(board_size * board_size)
        for move, child_node in root.children.items():
            move_idx = move[0] * board_size + move[1]
            policy_target[move_idx] = child_node.visit_count
        policy_target /= np.sum(policy_target)
        game_history.append((copy.deepcopy(game.board), game.current_player, policy_target))
        move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        game.make_move(move)
    final_winner = game.get_game_status()
    training_data = []
    for board_state, player_at_turn, policy in game_history:
        value = 1.0 if player_at_turn == final_winner else -1.0
        training_data.append((board_state, policy, value))
    return training_data

def train():
    board_size = CONFIG['board_size']
    device = CONFIG['device']
    model = HexZeroBrain(board_size=board_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    replay_buffer = deque(maxlen=CONFIG['replay_buffer_size'])
    print(f"Starting training on device: {device}")
    for i in range(CONFIG['num_iterations']):
        print(f"--- Iteration {i+1}/{CONFIG['num_iterations']} ---")
        model.eval()
        new_game_data = []
        for _ in tqdm(range(CONFIG['num_games']), desc="Self-Playing Games"):
            new_game_data.extend(self_play(model, board_size, CONFIG['num_simulations'], CONFIG['c_puct'], device))
        replay_buffer.extend(new_game_data)
        model.train()
        if len(replay_buffer) < CONFIG['batch_size']:
            continue

        boards, policies, values = zip(*replay_buffer)
        dataset = TensorDataset(
            torch.tensor(np.array(boards), dtype=torch.int).to(device),
            torch.tensor(np.array(policies), dtype=torch.float32).to(device),
            torch.tensor(np.array(values), dtype=torch.float32).view(-1, 1).to(device)
        )
        dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        total_policy_loss, total_value_loss = 0, 0
        
        for _ in tqdm(range(CONFIG['num_epochs']), desc="Training Epochs"):
            for boards_batch, policies_batch, values_batch in dataloader:
                optimizer.zero_grad()
                policy_logits, value_preds = model(boards_batch)
                policy_loss = -torch.sum(policies_batch * F.log_softmax(policy_logits, dim=1), dim=1).mean()
                value_loss = F.mse_loss(value_preds, values_batch)
                total_loss = policy_loss + value_loss
                
                total_loss.backward()
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        avg_p_loss = total_policy_loss / (len(dataloader) * CONFIG['num_epochs'])
        avg_v_loss = total_value_loss / (len(dataloader) * CONFIG['num_epochs'])
        print(f"Iteration {i+1} complete. Avg Policy Loss: {avg_p_loss:.4f}, Avg Value Loss: {avg_v_loss:.4f}")
        torch.save(model.state_dict(), f"model_checkpoint_iter_{i+1}.pth")

if __name__ == '__main__':
    train()