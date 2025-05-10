import chess.pgn

def lean_pgn(input_pgn, output_pgn):
    """
    This function preprocesses a PGN file by removing unnecessary tags and comments,
    and keeping only the moves.
    """
    with open(input_pgn, "r") as infile, open(output_pgn, "w") as outfile:
        game_count = 0
        while True:
            game = chess.pgn.read_game(infile)
            if game is None:
                break

            game_count += 1
            print(f"Processing game {game_count}...")

            board = chess.Board()
            move_list = []

            for move in game.mainline_moves():
                san_move = board.san(move)
                move_list.append(san_move)
                board.push(move)
            
            outfile.write(f"[Game number: {game_count}]\n")
            outfile.write(f"[Outcome: {game.headers['Result']}]\n")
            outfile.write(" ".join(move_list) + "\n\n")

    print(f"Processed {input_pgn} and saved to {output_pgn}")

if __name__ == "__main__":
    input_pgn = "2021-08-elite.pgn"
    output_pgn = "preprocessed.pgn"
    lean_pgn(input_pgn, output_pgn)