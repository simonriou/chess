import chess
import chess.engine
import pandas as pd
from tqdm import tqdm

stockfish_path = "/opt/homebrew/bin/stockfish"
input_csv = "input/sample_features.csv"
output_csv = "input/sample_eval.csv"
eval_depth = 15

df = pd.read_csv(input_csv)
if "FEN" not in df.columns:
    raise ValueError(f"Column 'FEN' not found in {input_csv}")

engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
evaluations = []

for fen in tqdm(df["FEN"], desc="Evaluating FENs"):
    board = chess.Board(fen)
    try:
        result = engine.analyse(board, chess.engine.Limit(depth=eval_depth))
        score = result["score"].relative

        if score.is_mate():
            val = 1000 if score.mate() > 0 else -1000
        else:
            val = score.score()

        evaluations.append(val)
    except Exception as e:
        print(f"Error evaluating FEN {fen}: {e}")
        evaluations.append(None)

engine.quit()

df["stockfish_eval"] = evaluations
df = df.dropna(subset=["stockfish_eval"])

df.to_csv(output_csv, index=False)
print(f"Evaluations saved to {output_csv}")