import hashlib
import os
import glob

def merge_csv_files(input_dir, output_filename):
    seen = set()
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

    with open(output_filename, 'w') as outfile:
        file_count = 0
        for csv_file in csv_files:
            file_count += 1
            print(f"Processing file {file_count}")
            with open(csv_file, 'r') as infile:
                for line in infile:
                    stripped_line = line.strip()
                    if stripped_line:
                        line_hash = hashlib.sha256(stripped_line.encode()).hexdigest()
                        if line_hash not in seen:
                            seen.add(line_hash)
                            outfile.write(stripped_line + '\n')

if __name__ == "__main__":
    input_directory = 'input/cleaned_fens'
    output_file = 'input/cleaned.csv'

    # Ensure the output file does not exist before merging
    if os.path.exists(output_file):
        os.remove(output_file)

    merge_csv_files(input_directory, output_file)
    print(f"Merged CSV files into {output_file}")