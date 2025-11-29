"""
Author: Shams
Description:
This script cleans a text file IN-PLACE by removing lines that contain a high
ratio of "junk" or mojibake characters. It defines a set of allowed characters
(Arabic, English, numbers) and deletes any line where the percentage of
unwanted characters exceeds a specified threshold (default 25%).
It writes to a temp file first for safety, then replaces the original.
"""

import argparse
import re
import os
import shutil
from tqdm import tqdm

def clean_file_by_junk_ratio_inplace(input_file, junk_threshold, encoding='utf-8'):
    print("--- Starting In-Place Junk Character Ratio Cleaning Process ---")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return

    print(f"Configuration:")
    print(f"  - Target File: {input_file}")
    print(f"  - Junk Ratio Threshold: {junk_threshold:.2%} ({junk_threshold})")
    print("-" * 30)
    
    temp_file = input_file + '.tmp'

    allowed_arabic = r'\u0621-\u064A'
    allowed_tashkeel = r'\u064B-\u065F'
    allowed_numbers = r'0-9\u0660-\u0669'
    allowed_english = r'a-zA-Z'
    allowed_punctuation = r'\s.,?!-_()\[\]{}:;\'"\/\\«»“”…' 

    allowed_chars_set = (
        allowed_arabic + allowed_tashkeel + allowed_numbers +
        allowed_english + allowed_punctuation
    )
    JUNK_CHARS_PATTERN = re.compile(f'[^{allowed_chars_set}]')

    lines_read = 0
    lines_written = 0
    lines_removed = 0

    try:
        print("--- Writing cleaned content to a temporary file... ---")
        with open(input_file, 'r', encoding=encoding, errors='ignore') as infile, \
             open(temp_file, 'w', encoding=encoding) as outfile:
            
            for line in tqdm(infile, desc="Filtering by junk ratio"):
                lines_read += 1
                clean_line = line.strip()

                if not clean_line:
                    continue

                total_length = len(clean_line)
                junk_count = len(JUNK_CHARS_PATTERN.findall(clean_line))
                junk_ratio = junk_count / total_length if total_length > 0 else 0

                if junk_ratio > junk_threshold:
                    lines_removed += 1
                    continue
                else:
                    outfile.write(clean_line + '\n')
                    lines_written += 1

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print("Operation aborted. The original file has not been changed.")
        return

    print("\n--- Finalizing: Replacing original file with cleaned version... ---")
    try:
        shutil.move(temp_file, input_file)
        print("File replacement successful.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to replace the original file.")
        print(f"The cleaned data is safe in the temporary file: '{temp_file}'")
        print(f"Please manually rename this file to '{input_file}' to complete the process.")
        print(f"Error details: {e}")
        return

    print("\n--- In-Place Cleaning Process Complete ---")
    print("\n--- Summary Report ---")
    print(f"Total lines read from original file: {lines_read:,}")
    print("-" * 25)
    print(f"Lines removed (high junk ratio):   {lines_removed:,}")
    print(f"Lines kept in the final file:      {lines_written:,}")
    print("-" * 25)
    print(f"File '{input_file}' has been successfully updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to clean a text file in-place by removing lines with a high ratio of junk/mojibake characters.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the text file to be cleaned in-place."
    )
    parser.add_argument(
        "--junk-threshold",
        type=float,
        default=0.1,
        help="The ratio of junk characters required to remove a sentence (e.g., 0.25 for 25%). (Default: 0.25)"
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="File encoding to use. (Default: utf-8)"
    )

    args = parser.parse_args()

    if not 0.0 <= args.junk_threshold <= 1.0:
        parser.error("--junk-threshold must be a value between 0.0 and 1.0.")

    clean_file_by_junk_ratio_inplace(
        input_file=args.input_file,
        junk_threshold=args.junk_threshold,
        encoding=args.encoding
    )
