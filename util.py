import pandas as pd
import os

def get_sample(input_file, output_file, n=10):
    df = pd.read_csv(input_file)
    sample = df.head(n)
    sample.to_csv(output_file, index=False)
    print(f"Sample saved to {output_file}")

def add_fen_header_if_missing(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            path = os.path.join(directory, filename)

            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline()

            # Check if header is already "FEN"
            if first_line.strip().upper() == "FEN":
                continue

            print(f"🛠 Adding header to: {filename}")

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            with open(path, "w", encoding="utf-8") as f:
                f.write("FEN\n" + content)

    print("✅ Done.")

# Example usage
if __name__ == "__main__":
    add_fen_header_if_missing("data.nosync/input/split_features")