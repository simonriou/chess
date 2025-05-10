import os
import chess
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

active_files = []
lock = threading.Lock()

def log_status():
    print(f"Currently processing: {', '.join(active_files)}")

def is_valid_fen(fen):
    try:
        board = chess.Board(fen)
        return board.is_valid()
    except:
        return False

def clean_fens(input_file, output_file):
    file_name = os.path.basename(input_file)

    with lock:
        active_files.append(file_name)
        log_status()

    unique_fens = set()

    with open(input_file, 'r', encoding="utf-8") as f_in, open(output_file, 'w', encoding="utf-8") as f_out:
        total_lines = sum(1 for _ in f_in)
        f_in.seek(0)  # Reset file pointer to the beginning
        for line in tqdm(f_in, total=total_lines, desc="Cleaning FENs", unit="line"):
            fen = line.strip()
            if fen and fen not in unique_fens and is_valid_fen(fen):
                unique_fens.add(fen)
                f_out.write(f"{fen}\n")
    
    with lock:
        active_files.remove(file_name)
        print(f"Finished processing {file_name}")
        log_status()

if __name__ == "__main__":
    input_dir = "input/split_fens"
    output_dir = "input/cleaned_fens"
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    files_nb = len(files)

    print(f"Starting multithread FEN cleaning on {files_nb} files")

    max_threads = min(4, os.cpu_count())

    print(f"Using {max_threads} threads")

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(clean_fens, os.path.join(input_dir, f), os.path.join(output_dir, f)): f
            for f in files
        }

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file {futures[future]}: {e}")