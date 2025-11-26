"""
Author: Shams
Description:
This script takes one massive corpus file and splits it into smaller parts.
It calculates how many lines to put in each part (roughly 10% chunks here)
and saves them into a new folder so they are easier to work with.
"""

import os
import math

def process_corpus_file(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: '{file_path}' not found. Skipping.")
        return

    print(f"Processing '{file_path}'...")

    # Create a folder named after the file (minus .txt)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.getcwd(), base_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"  -> Created folder: '{output_dir}'")

    try:
        with open(file_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not lines:
        print(f"  -> File is empty.")
        return

    total_lines = len(lines)
    # Calculate split size (10 parts)
    lines_per_part = math.ceil(total_lines / 10)

    # Slice the big list of lines into chunks
    parts = [
        lines[0 : lines_per_part],
        lines[lines_per_part : 2 * lines_per_part],
        lines[2 * lines_per_part : 3 * lines_per_part],
        lines[3 * lines_per_part : 4 * lines_per_part],
        lines[4 * lines_per_part : 5 * lines_per_part],
        lines[5 * lines_per_part : 6 * lines_per_part],
        lines[6 * lines_per_part : 7 * lines_per_part],
        lines[7 * lines_per_part : 8 * lines_per_part],
        lines[8 * lines_per_part : 9 * lines_per_part],
        lines[9 * lines_per_part : 10 * lines_per_part],
    ]

    line_counter = 1
    for i, part_lines in enumerate(parts):
        part_num = i + 1
        output_filename = os.path.join(output_dir, f"part_{part_num}.txt")

        with open(output_filename, 'w', encoding='utf-8') as f_out:
            for line in part_lines:
                clean_line = line.strip()

                # Remove old numbering if it exists so we can renumber cleanly
                if '.' in clean_line:
                    sentence = clean_line.split('. ', 1)[-1]
                else:
                    sentence = clean_line
                
                if sentence:
                    f_out.write(f"{line_counter}. {sentence}\n")
                    line_counter += 1
        
        print(f"  -> Wrote '{output_filename}'")

if __name__ == "__main__":
    # Loops through files named egyptian_arabic_corpus_0001.txt to 0012.txt
    for i in range(1, 13):
        filename = f"egyptian_arabic_corpus_{i:04d}.txt"
        process_corpus_file(filename)
    
    print("\nAll files split successfully.")
