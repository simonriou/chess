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
    "Backend": "cuda-fp16",
    "WeightsFile": "/home/simonari/build/lc0/build/release/bt4-1740.pb"
})

all_fens = ["rnb1kbn1/pppppppp/8/8/2B1P3/8/PPPP1PPP/RNBQK1NR w KQq - 0 1", "4kn1r/pQ3ppp/8/8/1B6/8/PPPPPPPP/RN2KBNR w KQk - 0 1", "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"]
evals = []

for fen in all_fens:
    board = chess.Board(fen)
    turn = board.turn

    info = engine.analyse(board, chess.engine.Limit(nodes=1000))
    score = info["score"].pov(turn).score()

    evals.append(score)
engine.quit()

print("Evaluations:", evals)