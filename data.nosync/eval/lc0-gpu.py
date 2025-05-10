import chess
import chess.engine
import csv
from tqdm import tqdm
import os

# Path to your lc0 binary (compiled with GPU support)
LC0_PATH = "/home/simonari/build/lc0/build/release/lc0"
output_path = "../input/scores.csv"

# Launch the engine
engine = chess.engine.SimpleEngine.popen_uci(LC0_PATH)

# Optional: Set lc0 options (like network weights, GPU backend, etc.)
engine.configure({
    "Threads": 1,
    "Backend": "cuda-fp16",
    "WeightsFile": "/home/simonari/build/lc0/build/release/bt4-1740.pb"
})

# Import fens from input/sample_features.csv (header "FEN")
with open("../input/features.csv", "r") as f:
    reader = csv.DictReader(f)
    all_fens = [row["FEN"] for row in reader]

# Determine how many FENs have already been evaluated
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        evaluated_lines = sum(1 for _ in f) - 1  # exclude header
else:
    evaluated_lines = 0

with open(output_path, "a", newline="") as f:
    writer = csv.writer(f)

    if evaluated_lines == 0:
        print("Evaluating FENs from the beginning...")
        writer.writerow(["evaluation"])

    print(f"Evaluating FENs from {evaluated_lines} to {len(all_fens)}")

    for i in tqdm(range(evaluated_lines, len(all_fens)), desc="Evaluating FENs"):
        fen = all_fens[i]
        board = chess.Board(fen)
        turn = board.turn

        info = engine.analyse(board, chess.engine.Limit(nodes=300))
        raw_score = info["score"]

        if raw_score.is_mate():
            score = 100000 if raw_score.pov(turn).mate() > 0 else -100000
        else:
            score = raw_score.pov(turn).score()

        writer.writerow([score])

engine.quit()
