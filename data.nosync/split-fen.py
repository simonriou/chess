import os
from tqdm import tqdm

def split_csv(filename, output_dir, num_splits=100):
    os.makedirs(output_dir, exist_ok=True)

    print("Counting lines in the CSV file...")
    with open(filename, 'r', encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    lines_per_file = total_lines // num_splits
    remainder = total_lines % num_splits
    print(f"Total lines: {total_lines}, Lines per file: {lines_per_file}, Remainder: {remainder}")

    with open(filename, 'r', encoding="utf-8") as f:
        with tqdm(total=total_lines, desc="Splitting CSV") as pbar:
            for i in range(num_splits):
                out_file = os.path.join(output_dir, f"split_{i+1}.csv")
                with open(out_file, 'w', encoding="utf-8") as out_f:
                    num_lines = lines_per_file + (1 if i < remainder else 0)
                    for _ in range(num_lines):
                        line = f.readline()
                        if not line:
                            break
                        out_f.write(line)
                        pbar.update(1)
    print(f"Splitting complete. Files saved in {output_dir}")

if __name__ == "__main__":
    input_csv = "input/fens.csv"
    output_dir = "input/split_fens"
    num_splits = 100
    split_csv(input_csv, output_dir, num_splits)