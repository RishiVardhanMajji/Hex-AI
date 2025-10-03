import torch
from engine import HexGame
from model import HexZeroBrain
from mcts import MCTS

# --- Configuration ---
BOARD_SIZE = 5 # Use a smaller board for faster testing
NUM_SIMULATIONS = 100 # Number of MCTS simulations per move
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_agent():
    """
    Main function to run the HexZero++ agent.
    """
    # 1. Create the instances of the game, model, and MCTS
    game = HexGame(board_size=BOARD_SIZE)
    model = HexZeroBrain(board_size=BOARD_SIZE).to(DEVICE)
    mcts = MCTS(model, num_simulations=NUM_SIMULATIONS,device=DEVICE)
    
    # Set the model to evaluation mode
    model.eval()

    print("--- Initial Board ---")
    game.display()
    print(f"Device: {DEVICE}")

    # 2. Get the intelligent move from MCTS
    # The MCTS search will now use the real HexZeroBrain for its predictions.
    # No gradients are needed as we are not training yet.
    with torch.no_grad():
        best_move = mcts.search(game)

    print(f"\nAgent chose move: {best_move}")

    # 3. Make the move on the board
    game.make_move(best_move)

    print("\n--- Board After Agent's Move ---")
    game.display()

if __name__ == '__main__':
    run_agent()