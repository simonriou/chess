# Chess

The aim of this project is to create a neural network capable of evaluating chess positions. The workflow involves several steps, including data preprocessing, dataset creation, and model training.

## Workflow Overview

1. **Preprocessing PGN Files**:
   - PGN files are processed using [`preprocess-pgn.py`](data.nosync/preprocess-pgn.py) to clean and extract relevant moves.

2. **Extracting FENs**:
   - FENs (Forsyth-Edwards Notation) are extracted from the preprocessed PGN files using [`extract-fen.py`](data.nosync/extract-fen.py).

3. **Filtering FENs**:
   - Invalid or duplicate FENs are removed using [`filter-fen.py`](data.nosync/filter-fen.py).

4. **Merging FEN Files**:
   - Multiple FEN files are merged into a single dataset using [`merge-fen.py`](data.nosync/merge-fen.py).

5. **Evaluating FENs**:
   - FENs are evaluated using chess engines like Stockfish or LCZero via scripts such as [`eval-fen.py`](data.nosync/eval-fen.py) and [`lc0-gpu.py`](data.nosync/lc0-gpu.py).

6. **Normalizing Evaluations**:
   - Evaluation scores are normalized to a range of [-1.0, 1.0] using [`normalize.py`](data.nosync/normalize.py).

7. **Encoding Data**:
   - FENs and their evaluations are converted into tensors and saved as TFRecord files using [`encode.py`](data.nosync/encode.py).

8. **Merging Features and Evaluations**:
   - FENs and their normalized evaluations are combined into a single dataset using [`merge-features.py`](temp/merge-features.py).

9. **Visualization**:
   - Tensor data is visualized using [`visualise.py`](temp/visualise.py).

10. **Model Training**:
    - The neural network is trained using scripts in the [`network`](network/) folder, such as [`train.py`](network/train.py).

## Directory Structure

- `models/`: Contains the trained neural network models.
- `data.nosync/`: Includes scripts for data preprocessing, evaluation, and encoding.
- `network/`: Contains scripts for building and training the neural network.
- `temp/`: Temporary files and scripts for merging and visualizing data.

## Requirements

- Python 3.8+
- TensorFlow
- Stockfish or LCZero chess engine
- tqdm
- pandas
- numpy
- matplotlib