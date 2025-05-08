import chess
import chess.engine

# Path to your lc0 binary (compiled with GPU support)
LC0_PATH = "/home/simonari/build/lc0/build/release/lc0"

# Launch the engine
engine = chess.engine.SimpleEngine.popen_uci(LC0_PATH)

# Optional: Set lc0 options (like network weights, GPU backend, etc.)
# You can run `lc0` manually to see supported options
# Example (you may or may not need this depending on how lc0 is compiled):
engine.configure({
    "Threads": 1,
    "Backend": "cuda"
    "/path/to/network.pb.gz"
})

# Define your FEN
fen = "r1bqkbnr/pppppppp/n7/8/8/N7/PPPPPPPP/R1BQKBNR w KQkq - 0 1"
board = chess.Board(fen)

# Run analysis
info = engine.analyse(board, chess.engine.Limit(nodes=1000))

# Show the score and best move
print("Score:", info["score"])
print("Best move:", info["pv"][0])

engine.quit()