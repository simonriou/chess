import os

def split_csv(input_file, output_prefix='output', lines_per_file=100000):
    file_count = 1
    line_count = 0
    output_file = open(f"input/split_features/{output_prefix}_{file_count}.csv", 'w', encoding='utf-8')

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line_count >= lines_per_file:
                output_file.close()
                file_count += 1
                line_count = 0
                output_file = open(f"input/split_features/{output_prefix}_{file_count}.csv", 'w', encoding='utf-8')
            
            output_file.write(line)
            line_count += 1

    output_file.close()
    print(f"✅ Split complete: {file_count} files created.")

# Example usage
if __name__ == "__main__":
    split_csv("input/features.csv", output_prefix="split_sample", lines_per_file=1000000)