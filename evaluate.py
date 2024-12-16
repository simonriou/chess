import chess
import chess.engine
import time

def evaluate_position(fen, engine, time_limit=0.1):
    """
    Evaluates a chess position given its FEN using Stockfish and provides detailed information.

    Args:
        fen (str): The FEN string of the chess position.
        stockfish_path (str): Path to the Stockfish executable.
        time_limit (float): Time in seconds to allow Stockfish to calculate.

    Returns:
        dict: A dictionary containing detailed evaluation info.
    """
    
    # Create a board from the FEN
    board = chess.Board(fen)

    # Analyze the position
    result = engine.analyse(board, chess.engine.Limit(time=time_limit))

    evaluation = {"fen": fen}

    # Extract evaluation details
    if 'score' in result:
        score = result['score'].white()
        if score.is_mate():
            mate_in = score.mate()  # Number of moves to mate
            evaluation["type"] = "mate"
            evaluation["value"] = f"M{mate_in}" if mate_in > 0 else f"M{mate_in}"
            evaluation["description"] = (
                f"Mate in {abs(mate_in)} moves for {'white' if mate_in > 0 else 'black'}"
            )
        else:
            centipawns = score.score()  # Centipawn evaluation
            evaluation["type"] = "centipawn"
            evaluation["value"] = centipawns
            evaluation["description"] = f"Centipawn evaluation: {centipawns}"
    else:
        evaluation["type"] = "unknown"
        evaluation["value"] = None
        evaluation["description"] = "Unknown evaluation."
    # Add board state and other metadata
    evaluation["turn"] = "white" if board.turn else "black"
    evaluation["fullmove_number"] = board.fullmove_number
    evaluation["is_check"] = board.is_check()
    evaluation["is_game_over"] = board.is_game_over()
    evaluation["legal_moves"] = board.legal_moves.count()
    
    return evaluation