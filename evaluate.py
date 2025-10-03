import torch
from tqdm import tqdm
from engine import HexGame
from model import HexZeroBrain
from mcts import MCTS # Import the MCTS class

# --- Configuration ---
BOARD_SIZE = 5
MODEL_1_PATH = "model_checkpoint_iter_1.pth"
MODEL_2_PATH = "model_checkpoint_iter_10.pth"
NUM_GAMES = 20
NUM_SIMULATIONS = 50 # MCTS simulations for evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, board_size, device):
    """Loads a model from a checkpoint file."""
    # Add the 'weights_only=True' argument to address the FutureWarning
    model_state = torch.load(path, map_location=device, weights_only=True)
    model = HexZeroBrain(board_size=board_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model

def play_game(p1_mcts, p2_mcts, board_size):
    """Plays one game between two MCTS agents and returns the winner."""
    game = HexGame(board_size=board_size)
    while game.get_game_status() == 0:
        if game.current_player == 1:
            move = p1_mcts.search(game)
        else:
            move = p2_mcts.search(game)
        game.make_move(move)
    return game.winner

def evaluate():
    """Pits two models against each other using MCTS."""
    print("Loading models...")
    model1 = load_model(MODEL_1_PATH, BOARD_SIZE, DEVICE)
    model2 = load_model(MODEL_2_PATH, BOARD_SIZE, DEVICE)
    print("Models loaded.")

    # Create MCTS instances for each model
    mcts1 = MCTS(model1, num_simulations=NUM_SIMULATIONS, device=DEVICE)
    mcts2 = MCTS(model2, num_simulations=NUM_SIMULATIONS, device=DEVICE)

    # --- Match 1: Model 10 (P1) vs Model 1 (P2) ---
    p1_wins = 0
    print(f"\n--- Starting {NUM_GAMES // 2} games: Model 10 (P1) vs Model 1 (P2) ---")
    for _ in tqdm(range(NUM_GAMES // 2), desc="Games (M10 vs M1)"):
        winner = play_game(mcts2, mcts1, BOARD_SIZE)
        if winner == 1:
            p1_wins += 1
    
    # --- Match 2: Model 1 (P1) vs Model 10 (P2) ---
    p2_wins = 0
    print(f"\n--- Starting {NUM_GAMES // 2} games: Model 1 (P1) vs Model 10 (P2) ---")
    for _ in tqdm(range(NUM_GAMES // 2), desc="Games (M1 vs M10)"):
        winner = play_game(mcts1, mcts2, BOARD_SIZE)
        if winner == 2:
            p2_wins += 1

    total_m10_wins = p1_wins + p2_wins
    total_games = NUM_GAMES // 2 + NUM_GAMES // 2

    print("\n--- Overall Results ---")
    print(f"Total Wins for Model 10: {total_m10_wins}/{total_games}")
    print(f"Total Wins for Model 1: {total_games - total_m10_wins}/{total_games}")
    if total_games > 0:
        win_rate = total_m10_wins / total_games * 100
        print(f"Win Rate for Model 10: {win_rate:.2f}%")

if __name__ == '__main__':
    evaluate()