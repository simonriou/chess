import chess.pgn
import csv

def extract_positions(input_pgn, output_csv):
    with open(input_pgn, "r") as infile, open(output_csv, "w", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["FEN"]) # Header

        game_count = 0
        while True:
            game = chess.pgn.read_game(infile)
            if game is None:
                break

            game_count += 1
            print(f"Processing game {game_count}...")

            board = chess.Board()

            for move in game.mainline_moves():
                writer.writerow([board.fen()])
                board.push(move)          
                 
    print(f"Position dataset saved to {output_csv}")

if __name__ == "__main__":
    input_pgn = "preprocessed.pgn"
    output_csv = "fens.csv"
    extract_positions(input_pgn, output_csv)