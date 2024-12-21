from utils import clean_pgn, export_features_labels, split_pgn
from data_process import get_raw_fen, process_game, process_file

file_path = 'datasets/blitz_2000_23.pgn'
file_name = file_path.split('/')[-1].split('.')[0]
stockfish_path = 'stockfish/stockfish-macos-m1-apple-silicon'

game_fens = get_raw_fen(clean_pgn(file_path))

# Splitting the PGN file
for file_path in split_pgn(f"cleaned/{file_name}_cleaned.pgn", 100):
    file_nb = file_path.split('/')[2].split('split_')[1].split('.')[0]
    print(f"Processing file {file_nb}")
    process_file(file_path, stockfish_path, file_name, 0.5, max_workers=6)