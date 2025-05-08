import chess
import chess.engine
import csv
from tqdm import tqdm  # Progress bar

# Path to your lc0 binary (compiled with GPU support)
LC0_PATH = "/home/simonari/build/lc0/build/release/lc0"

# Launch the engine
engine = chess.engine.SimpleEngine.popen_uci(LC0_PATH)

# Optional: Set lc0 options (like network weights, GPU backend, etc.)
engine.configure({
    "Threads": 1,
    "Backend": "cuda-fp16",
    "WeightsFile": "/home/simonari/build/lc0/build/release/bt4-1740.pb"
})

# Import fens from input/sample_features.csv (header "FEN")
with open("input/features.csv", "r") as f:
    reader = csv.DictReader(f)
    all_fens = [row["FEN"] for row in reader]

with open("input/scores.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["evaluation"])
    for fen in tqdm(all_fens, desc="Evaluating FENs"):
        board = chess.Board(fen)
        turn = board.turn

        info = engine.analyse(board, chess.engine.Limit(nodes=500))
        raw_score = info['score']

        if raw_score.is_mate():
            score = 100000 if raw_score.pov(turn).mate() > 0 else -100000
        else:
            score = raw_score.pov(turn).score()

        writer.writerow([score])

engine.quit()
