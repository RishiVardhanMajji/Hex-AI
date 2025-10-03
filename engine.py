import numpy as np

class HexGame:
    """
    A class to represent and manage the game of Hex.
    Board representation: 0=empty, 1=player1 (Horizontal), 2=player2 (Vertical)
    Player 1 wins by connecting the left and right edges.
    Player 2 wins by connecting the top and bottom edges.
    """
    def __init__(self, board_size=9):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.winner = 0  # 0: ongoing, 1: player 1 wins, 2: player 2 wins

    def get_legal_moves(self):
        """Returns a list of (row, col) tuples for all empty cells."""
        return list(zip(*np.where(self.board == 0)))

    def is_move_legal(self, move):
        """Checks if a move is legal."""
        return self.board[move[0], move[1]] == 0

    def make_move(self, move):
        """Places a piece on the board and switches the player."""
        if not self.is_move_legal(move):
            raise ValueError(f"Illegal move {move} attempted.")
        
        self.board[move[0], move[1]] = self.current_player
        if self._check_win(self.current_player):
            self.winner = self.current_player
        
        self.current_player = 3 - self.current_player # Switch player (1 -> 2, 2 -> 1)
        return self.get_game_status()

    def get_game_status(self):
        """Returns the current game status."""
        if self.winner != 0:
            return self.winner
        if not self.get_legal_moves():
            return -1 # Draw (technically impossible in Hex)
        return 0 # Ongoing

    def _check_win(self, player):
        """Checks if the given player has won using BFS."""
        if player == 1: # Player 1 connects left to right
            starts = [(r, 0) for r in range(self.board_size) if self.board[r, 0] == player]
        else: # Player 2 connects top to bottom
            starts = [(0, c) for c in range(self.board_size) if self.board[0, c] == player]

        for start_node in starts:
            q = [start_node]
            visited = {start_node}
            while q:
                r, c = q.pop(0)
                
                if (player == 1 and c == self.board_size - 1) or \
                   (player == 2 and r == self.board_size - 1):
                    return True

                # Check all 6 neighbors in a hex grid
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0), (1,-1), (-1,1)]:
                    nr, nc = r + dr, c + dc
                    
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size and \
                       self.board[nr, nc] == player and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return False

    def display(self):
        """Prints a human-readable representation of the board."""
        p1_char, p2_char, empty_char = 'X', 'O', '.'
        print("\nPlayer 1 (X) -> Horizontal | Player 2 (O) -> Vertical")
        for r in range(self.board_size):
            print(" " * (r + 2), end="")
            row_str = []
            for c in range(self.board_size):
                if self.board[r,c] == 1: row_str.append(p1_char)
                elif self.board[r,c] == 2: row_str.append(p2_char)
                else: row_str.append(empty_char)
            print(" - ".join(row_str))
        print("-" * (4 * self.board_size))