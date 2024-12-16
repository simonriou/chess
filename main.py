from utils import clean_pgn, export_features_labels
from data_process import get_raw_fen, process_game, process_file

file_path = 'datasets/test_double.pgn'
file_name = file_path.split('/')[-1].split('.')[0]
stockfish_path = 'stockfish/stockfish-macos-m1-apple-silicon'

game_fens = get_raw_fen(clean_pgn(file_path))
process_file(file_path, stockfish_path, file_name, 0.5)