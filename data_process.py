import chess
import chess.pgn
import chess.engine
from evaluate import evaluate_position
from utils import export_features_labels

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import tqdm

def get_raw_fen(file_path):
    """
    Reads a PGN file and returns a list of games, where each game is a list of FENs after each move.

    Args:
    - file_path: str, path to the PGN file to read

    Returns:
    - games: list of lists of strings, where each string is a FEN
    """
    game_fens = []

    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            # Parse each game in the PGN file
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Create a new board for each game and process moves
            board = game.board()
            fens = []
            for move in game.mainline_moves():
                board.push(move)
                fens.append(board.fen())

            game_fens.append(fens)
        
        return game_fens

def process_game(fens, stockfish_path, game_idx=1, time_limit=0.1, tqdm_dict=None):
    """
    Processes a game given a list of FENs and returns a list of evaluations.

    Args:
    - fens: list of strings, where each string is a FEN
    - stockfish_path: str, path to the Stockfish executable
    - game_idx: int, index of the current game
    - time_limit: float, time limit for Stockfish to evaluate each position
    - tqdm_dict: dict, a shared dictionary for managing progress bars across threads

    Returns:
    - evaluations: list of dicts, where each dict contains detailed evaluation info
    """
    evaluations = []

    # Create a unique progress bar for the current game
    tqdm_bar = tqdm.tqdm(
        total=len(fens),
        desc=f"Game {game_idx + 1} - Processing FENs",
        position=game_idx,  # Each game gets a unique position
        leave=True
    )

    # Register the progress bar in the shared dictionary
    if tqdm_dict is not None:
        tqdm_dict[game_idx] = tqdm_bar

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        for fen in fens:
            evaluation = evaluate_position(fen, engine, time_limit)
            evaluations.append(evaluation)
            tqdm_bar.update(1)  # Update progress bar

    # Close the progress bar when done
    tqdm_bar.close()
    if tqdm_dict is not None:
        del tqdm_dict[game_idx]  # Clean up the shared dictionary

    return evaluations

def process_file(file_path, stockfish_path, set_name, time_limit=0.1, max_workers=4):
    """
    Processes a PGN file in parallel and saves features and labels to JSON or HDF5 files.

    Args:
    - file_path: str, path to the PGN file to process
    - stockfish_path: str, path to the Stockfish executable
    - time_limit: float, maximum time Stockfish can take for each evaluation
    - max_workers: int, maximum number of threads to use for parallel processing
    """     

    game_fens = get_raw_fen(file_path)  # Extract FENs for each game
    all_evaluations = [None] * len(game_fens)  # Pre-allocate results for order
    tqdm_dict = {}  # Shared dictionary for progress bar management

    def worker(game_idx):
        """Worker function to process a single game."""
        return game_idx, process_game(
            game_fens[game_idx],
            stockfish_path,
            game_idx=game_idx,
            time_limit=time_limit,
            tqdm_dict=tqdm_dict
        )

    # Use ThreadPoolExecutor to process games in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, i): i for i in range(len(game_fens))}
        for future in as_completed(futures):
            game_idx, evaluations = future.result()
            all_evaluations[game_idx] = evaluations  # Store evaluations in the correct order

    # Prepare features and labels
    features = game_fens
    labels = [[{'type': move['type'], 'value': move['value']} for move in game] for game in all_evaluations]

    # File number
    file_nb = file_path.split('/')[2].split('split_')[1].split('.')[0]

    # Export features and labels
    export_features_labels(set_name, file_nb, features, labels, unroll=False)