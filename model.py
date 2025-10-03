import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    """A standard residual block for the CNN."""
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class TransformerEncoderBlock(nn.Module):
    """A standard Transformer Encoder block."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        x = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_output))
        return x

class HexZeroBrain(nn.Module):
    """
    The main Conv->Transformer hybrid model for HexZero++.
    """
    def __init__(self, board_size=9, num_cnn_blocks=4, cnn_channels=64,
                 embed_dim=128, num_transformer_blocks=4, num_heads=4, ff_dim=256):
        super(HexZeroBrain, self).__init__()
        self.board_size = board_size
        self.embed_dim = embed_dim

        # --- CNN Backbone ---
        self.initial_conv = nn.Conv2d(3, cnn_channels, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(cnn_channels)
        self.cnn_blocks = nn.Sequential(*[ResidualBlock(cnn_channels) for _ in range(num_cnn_blocks)])

        # --- Bridge from CNN to Transformer ---
        self.conv_to_transformer = nn.Conv2d(cnn_channels, embed_dim, kernel_size=1)

        # --- Positional Embedding ---
        self.positional_embedding = nn.Parameter(torch.randn(1, board_size * board_size, embed_dim))

        # --- Transformer Encoder ---
        self.transformer_blocks = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, ff_dim) for _ in range(num_transformer_blocks)]
        )

        # --- Output Heads ---
        # Policy Head: Predicts the best move
        self.policy_head = nn.Linear(embed_dim, 1)

        # Value Head: Predicts the chance of winning
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x is a batch of board states, shape (batch_size, board_size, board_size)

        # --- Prepare Input Tensor ---
        p1_pieces = (x == 1).float().unsqueeze(1)
        p2_pieces = (x == 2).float().unsqueeze(1)
        empty_pieces = (x == 0).float().unsqueeze(1)
        x_conv = torch.cat([p1_pieces, p2_pieces, empty_pieces], dim=1)

        # --- CNN Backbone ---
        x_conv = F.relu(self.initial_bn(self.initial_conv(x_conv)))
        x_conv = self.cnn_blocks(x_conv)

        # --- Bridge to Transformer ---
        x_transformer = self.conv_to_transformer(x_conv)
        batch_size = x_transformer.shape[0]
        x_transformer = x_transformer.view(batch_size, self.embed_dim, -1).permute(0, 2, 1)

        # --- Transformer Encoder ---
        x_transformer += self.positional_embedding
        x_transformer = self.transformer_blocks(x_transformer)

        # --- Output Heads ---
        # Apply the policy head and squeeze the last dimension to get the correct shape
        policy_logits = self.policy_head(x_transformer).squeeze(-1) # Shape becomes (batch, seq_len)

        # Use the average of all token representations for the value prediction
        value_input = x_transformer.mean(dim=1)
        value = torch.tanh(self.value_head(value_input)) # tanh scales output to [-1, 1]

        return policy_logits, value

# --- Test the model with a dummy input ---
if __name__ == '__main__':
    BOARD_SIZE = 9
    model = HexZeroBrain(board_size=BOARD_SIZE)
    print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    dummy_input = torch.zeros(4, BOARD_SIZE, BOARD_SIZE, dtype=torch.int)
    dummy_input[0, 3, 3] = 1
    dummy_input[0, 4, 4] = 2

    model.eval()
    with torch.no_grad():
        policy, value = model(dummy_input)

    print("\n--- Output Shapes ---")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Policy logits shape: {policy.shape}") # Should be (4, 81)
    print(f"Value shape: {value.shape}")          # Should be (4, 1)

    print("\n--- Example Outputs ---")
    print(f"Example Policy (logits for board 0):\n{policy[0]}")
    print(f"Example Value (for board 0): {value[0].item():.4f}")