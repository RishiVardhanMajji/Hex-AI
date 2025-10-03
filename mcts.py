import numpy as np
import math
import copy
import torch

class Node:
    """A node in the Monte Carlo Search Tree."""
    def __init__(self, game, parent=None, move=None, prior_p=1.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = {}
        self.visit_count = 0
        self.total_action_value = 0.0  # W value
        self.prior_probability = prior_p # P value

    def select_child(self, c_puct):
        """Selects the child with the highest PUCT score."""
        best_score, best_child = -float('inf'), None
        for move, child in self.children.items():
            q_value = -child.total_action_value / (child.visit_count + 1e-8)
            u_value = c_puct * child.prior_probability * \
                      math.sqrt(self.visit_count) / (1 + child.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score, best_child = score, child
        return best_child

    def expand(self, policy_probs):
        """Expands the node by creating new children for all legal moves."""
        legal_moves = self.game.get_legal_moves()
        for move in legal_moves:
            if move not in self.children:
                temp_game = copy.deepcopy(self.game)
                temp_game.make_move(move)
                move_index = move[0] * self.game.board_size + move[1]
                self.children[move] = Node(temp_game, parent=self, move=move, prior_p=policy_probs[move_index])

    def backpropagate(self, value):
        """Updates the visit counts and action values of this node and all its ancestors."""
        self.visit_count += 1
        self.total_action_value += value
        if self.parent:
            self.parent.backpropagate(-value)

class MCTS:
    """The main Monte Carlo Tree Search class."""
    def __init__(self, model, num_simulations=100, c_puct=1.0, device='cpu'):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device

    def search(self, game):
        """Runs MCTS and returns the best move for gameplay."""
        root = self._run_simulations(game)
        most_visited_child = max(root.children.values(), key=lambda n: n.visit_count)
        return most_visited_child.move

    def search_for_training(self, game):
        """Runs MCTS and returns the root node for generating training data."""
        return self._run_simulations(game)

    def _run_simulations(self, game):
        """The core MCTS loop, shared by both search methods."""
        root = Node(game)
        
        for _ in range(self.num_simulations):
            node = root
            
            while node.children:
                node = node.select_child(self.c_puct)

            game_status = node.game.get_game_status()
            value = 0.0
            if game_status == 0: # Game is ongoing
                board_tensor = torch.tensor(node.game.board, dtype=torch.int).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    policy_logits, value_tensor = self.model(board_tensor)
                
                value = value_tensor.item()
                policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).detach().cpu().numpy()
                
                legal_moves_mask = np.zeros(game.board_size * game.board_size)
                for r, c in game.get_legal_moves():
                    legal_moves_mask[r * game.board_size + c] = 1
                policy_probs *= legal_moves_mask
                if np.sum(policy_probs) > 0:
                    policy_probs /= np.sum(policy_probs)
                
                node.expand(policy_probs)
            else: # Game has ended
                # ######################
                # ### START OF FIX ###
                # ######################
                # The value for a terminal node is always from the perspective of the player
                # whose turn it is. At a terminal node, that player has lost.
                value = -1.0
                # ####################
                # ### END OF FIX ###
                # ####################
            
            node.backpropagate(value)
        return root

if __name__ == '__main__':
    from engine import HexGame

    class DummyModel(torch.nn.Module):
        def __init__(self, board_size=5):
            super(DummyModel, self).__init__()
            self.board_size = board_size
        def forward(self, x):
            batch_size = x.shape[0]
            policy = torch.ones(batch_size, self.board_size * self.board_size)
            value = torch.rand(batch_size, 1) * 2 - 1
            return policy, value

    game = HexGame(board_size=5)
    dummy_model = DummyModel()
    mcts = MCTS(dummy_model, num_simulations=50)
    
    print("Initial Board:")
    game.display()
    
    best_move = mcts.search(game)
    print(f"\nMCTS decided the best move is: {best_move}")
    
    game.make_move(best_move)
    print("\nBoard after MCTS move:")
    game.display()