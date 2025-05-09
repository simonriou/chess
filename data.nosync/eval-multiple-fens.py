import os
import threading
import queue
import pandas as pd
import chess
import chess.engine
from tqdm import tqdm

# == Config ==
stockfish_path = "/opt/homebrew/bin/stockfish"
input_dir = "input/split_features"
output_suffix = "_eval.csv"
eval_depth = 15
num_threads = 4  # Number of threads to use for evaluation

# == Thread Worker ==
def evaluate_file_worker(file_queue):
    while True:
        try:
            input_file = file_queue.get_nowait()
        except queue.Empty:
            break

        try:
            output_file = os.path.join("input/split_labels", os.path.basename(input_file).replace(".csv", output_suffix))
            if os.path.exists(output_file):
                print(f"Skipping {output_file} (already exists)")
                continue

            df = pd.read_csv(input_file)
            if "FEN" not in df.columns:
                print(f"Column 'FEN' not found in {input_file}")
                continue

            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            evaluations = []

            for fen in tqdm(df["FEN"], desc=f"Evaluating {os.path.basename(input_file)}", leave=False):
                try:
                    board = chess.Board(fen)
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
            df.dropna(subset=["stockfish_eval"], inplace=True)
            df.to_csv(output_file, index=False)
            print(f"Evaluations saved to {output_file}")

        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
        finally:
            file_queue.task_done()
            print(f"✅ Finished processing {input_file}")

# == Main Logic ==
def main():
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv') and not f.endswith(output_suffix)]

    file_queue = queue.Queue()
    for file in input_files:
        file_queue.put(file)

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=evaluate_file_worker, args=(file_queue,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    
    print("All files processed.")

if __name__ == "__main__":
    main()