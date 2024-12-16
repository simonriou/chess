# Chess Bot

The aim of this project is to develop a neural network capable of evaluating chess positions. The project is still in its infancy and is currently under development.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
  - [clean_pgn](#clean_pgn)
  - [export_features_labels](#export_features_labels)
  - [evaluate_position](#evaluate_position)
  - [get_raw_fen](#get_raw_fen)
  - [process_game](#process_game)
  - [process_file](#process_file)
  - [fen_to_matrix](#fen_to_matrix)
  - [flatten_data](#flatten_data)
  - [normalize_evaluation](#normalize_evaluation)
  - [split_pgn](#split_pgn)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/simonriou/chess.git
    cd chess
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download and install [Stockfish](https://stockfishchess.org/download/) (or use the .zip provided here)

## Current Features

Data formatting has been solved. Each input feature is a 8x8x12 matrix. (8x8 is the chessboard, 12 is 1 channel / piece, 1 if there's a piece, 0 else). My first try is a CNN + 2 Dense layers network. Still in the process of testing it on different datasets. Will update when I come up with something.

## Usage

1. Clean a PGN file:
    ```python
    from utils import clean_pgn

    cleaned_pgn_path = clean_pgn('path/to/your/file.pgn')
    print(f"Cleaned PGN file saved to: {cleaned_pgn_path}")
    ```

2. Process a PGN file and export features and labels:
    ```python
    from data_process import process_file

    process_file('path/to/your/file.pgn', 'path/to/stockfish', 'dataset_name')
    ```

3. Split a PGN file into smaller files:
    ```python
    from utils import split_pgn

    split_files = split_pgn('path/to/your/file.pgn', num_files=10)
    print(f"Split files: {split_files}")
    ```

4. Convert FEN to matrix and normalize evaluations:
    ```python
    from utils import fen_to_matrix, normalize_evaluation

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    matrix = fen_to_matrix(fen)
    print(matrix)

    evaluation = {"type": "centipawn", "value": 34}
    normalized_value = normalize_evaluation(evaluation)
    print(normalized_value)
    ```

## Functions

### clean_pgn

Removes time annotations from a PGN file and writes the cleaned file to a new file.

**Args:**
- `file_path` (str): Path to the PGN file to clean.

**Returns:**
- `output_path` (str): Path to the cleaned PGN file.

### export_features_labels

Exports the features (board positions) and the labels (evaluation scores) to JSON (or HDF5 files in the future).

**Args:**
- `set_name` (str): Name of the dataset.
- `file_nb` (int): Number of the file being processed.
- `features` (list of lists): List of board positions.
- `labels` (list of lists): List of evaluation scores.
- `unroll` (bool): Whether to unroll the features and labels into a single list.

**Returns:**
- `features_path` (str): Path to the file containing the features.
- `labels_path` (str): Path to the file containing the labels.

### evaluate_position

Evaluates a chess position given its FEN using Stockfish and provides detailed information.

**Args:**
- `fen` (str): The FEN string of the chess position.
- `engine` (chess.engine.SimpleEngine): The Stockfish engine instance.
- `time_limit` (float): Time in seconds to allow Stockfish to calculate.

**Returns:**
- `dict`: A dictionary containing detailed evaluation info.

### get_raw_fen

Reads a PGN file and returns a list of games, where each game is a list of FENs after each move.

**Args:**
- `file_path` (str): Path to the PGN file to read.

**Returns:**
- `games` (list of lists of strings): List of games with FEN strings.

### process_game

Processes a game given a list of FENs and returns a list of evaluations.

**Args:**
- `fens` (list of strings): List of FEN strings.
- `stockfish_path` (str): Path to the Stockfish executable.
- `game_idx` (int): Index of the current game.
- `time_limit` (float): Time limit for Stockfish to evaluate each position.
- `tqdm_dict` (dict): A shared dictionary for managing progress bars across threads.

**Returns:**
- `evaluations` (list of dicts): List of evaluations.

### process_file

Processes a PGN file in parallel and saves features and labels to JSON (or HDF5 files in the future).

**Args:**
- `file_path` (str): Path to the PGN file to process.
- `stockfish_path` (str): Path to the Stockfish executable.
- `set_name` (str): Name of the dataset.
- `time_limit` (float): Maximum time Stockfish can take for each evaluation.
- `max_workers` (int): Maximum number of threads to use for parallel processing.

### fen_to_matrix

Convert a FEN string into an 8x8x12 matrix representation.

**Args:**
- `fen` (str): FEN string describing the board state.

**Returns:**
- `np.ndarray`: 8x8x12 tensor representing the board.

### flatten_data

Flattens the features and labels.

**Args:**
- `features` (list of lists): List of board positions.
- `labels` (list of lists): List of evaluation scores.

**Returns:**
- `flat_features` (np.array): Flattened features.
- `flat_labels` (np.array): Flattened labels.

### normalize_evaluation

Normalize evaluation values to the range [-1, 1].

**Args:**
- `evaluation` (dict): The evaluation dictionary from evaluate_position.
- `max_centipawn` (int): Maximum centipawn value for normalization.
- `max_mate_distance` (int): Maximum mate distance for normalization.

**Returns:**
- `float`: Normalized evaluation value.

### split_pgn

Splits a PGN file into smaller files, each containing an equal number of games.

**Args:**
- `input_pgn_file` (str): Path to the input PGN file.
- `num_files` (int): Number of files to split the games into.

**Returns:**
- `list of str`: Paths to the split PGN files.

## Examples

The `processed/` directory contains the processed versions of the `datasets/classical_2000_0124.pgn` and `datasets/test_double.pgn` files. The first one contains all the classical games of 2000+ rated players played on Lichess during January 2024. The second one is just a sample of that file that I used for tests.

It took approximately 20 minutes to process a bit more than 1000 games using 4 workers. It is still relatively slow, but keep in mind that this process will be needed once for each dataset.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.