"""
Author: Shams
Description:
I've got all these separate files ending in '-P.txt'.
This script grabs them all, puts them into one big list, removes duplicates,
sorts them alphabetically, and saves one giant 'COMBINED_CORPUS.txt'.
It also prints out some stats so I know what I'm working with.
"""

import os
import re
import pandas as pd
import glob
from collections import Counter

# Settings
FINAL_OUTPUT_FILENAME = "COMBINED_CORPUS.txt"
EXTREME_RATIO_THRESHOLD = 3.0

def parse_p_file(file_path):
    """
    Reads a file that looks like 'ID. English ||| Arabic'.
    """
    pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line or '|||' not in clean_line:
                    continue
                
                parts = clean_line.rsplit('|||', 1)
                if len(parts) != 2:
                    continue

                eng_part, ara_part = parts
                
                # Get rid of the ID number
                eng_sentence = re.sub(r'^\d+[\s.-]*', '', eng_part).strip()
                ara_sentence = ara_part.strip()

                if eng_sentence and ara_sentence:
                    pairs.append((eng_sentence, ara_sentence))
    except Exception as e:
        print(f"  - Can't read {file_path}. Error: {e}")
        return []
                
    return pairs

def clean_text_for_vocab(text):
    """Simple cleaner just for counting words."""
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077Fa-zA-Z0-9\s]', '', text)
    return text.lower().strip()

def run_comprehensive_eda(df, title):
    """
    This function looks at the data and prints out a report.
    """
    print("\n" + "="*70)
    print(f"||   {title.upper()}   ||")
    print("="*70 + "\n")

    # 1. Basics
    print("--- 1. Stats ---")
    df['eng_words'] = df['english'].str.split().str.len()
    df['ara_words'] = df['arabic'].str.split().str.len()
    print(f"Total pairs: {len(df):,}")
    print(f"Avg English length: {df['eng_words'].mean():.2f} words")
    print(f"Avg Arabic length: {df['ara_words'].mean():.2f} words")
    print("-" * 35)
    print(f"Total English words: {df['eng_words'].sum():,}")
    print(f"Total Arabic words: {df['ara_words'].sum():,}")
    print("\n")

    # 2. Duplicates
    print("--- 2. Duplicates ---")
    total_pairs = len(df)
    unique_pairs_count = len(df.drop_duplicates(subset=['english', 'arabic']))
    uniqueness_ratio = (unique_pairs_count / total_pairs) * 100 if total_pairs > 0 else 0
    print(f"Total pairs: {total_pairs:,}")
    print(f"Unique pairs: {unique_pairs_count:,}")
    print(f"Uniqueness: {uniqueness_ratio:.2f}%")
    print("(Higher is better.)")
    print("\n")
    
    # 3. Words
    print("--- 3. Word Counts ---")
    print("Counting words...")
    eng_vocab = Counter(clean_text_for_vocab(' '.join(df['english'])).split())
    ara_vocab = Counter(clean_text_for_vocab(' '.join(df['arabic'])).split())
    print(f"Unique English words: {len(eng_vocab):,}")
    print(f"Unique Arabic words: {len(ara_vocab):,}")
    
    print("\nTop 10 English words:")
    for word, count in eng_vocab.most_common(10): print(f"  - '{word}': {count:,}")
    
    print("\nTop 10 Arabic words:")
    for word, count in ara_vocab.most_common(10): print(f"  - '{word}': {count:,}")
    print("\n")

    # 4. Ratios
    print(f"--- 4. Length Ratios (Threshold: {EXTREME_RATIO_THRESHOLD:.1f}x) ---")
    df_ratio = df.drop_duplicates(subset=['english', 'arabic'])
    df_ratio = df_ratio[(df_ratio['ara_words'] > 0) & (df_ratio['eng_words'] > 0)].copy()
    df_ratio['ratio'] = df_ratio['eng_words'] / df_ratio['ara_words']
    
    high_ratio_pairs = df_ratio[df_ratio['ratio'] > EXTREME_RATIO_THRESHOLD]
    low_ratio_pairs = df_ratio[df_ratio['ratio'] < (1 / EXTREME_RATIO_THRESHOLD)]
    
    print(f"Pairs where English is way longer: {len(high_ratio_pairs):,}")
    print(f"Pairs where Arabic is way longer: {len(low_ratio_pairs):,}")
    print("(These should be low.)")
    print("\n")

def main():
    # Find all files ending in -P.txt
    target_files = sorted(glob.glob("*-P.txt"))
    
    if not target_files:
        print("Error: No '*-P.txt' files found.")
        return

    print(f"Found {len(target_files)} files to combine.")
    
    all_pairs = []
    for file_path in target_files:
        print(f"  - Reading: {file_path}")
        pairs = parse_p_file(file_path)
        all_pairs.extend(pairs)

    print("-" * 35)
    print(f"Total pairs collected: {len(all_pairs):,}")

    if not all_pairs:
        print("Empty. Stopping.")
        return

    # Make a dataframe
    combined_df = pd.DataFrame(all_pairs, columns=['english', 'arabic'])

    # Sort it so it looks nice
    print("Sorting alphabetically...")
    combined_df = combined_df.sort_values(by='english', kind='mergesort').reset_index(drop=True)

    # Save it
    print(f"Saving to '{FINAL_OUTPUT_FILENAME}'...")
    with open(FINAL_OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        for index, row in combined_df.iterrows():
            f.write(f"{index + 1}. {row['english']} ||| {row['arabic']}\n")
    print("Saved.")

    # Run the report
    run_comprehensive_eda(combined_df, "Final Combined Corpus Report")

if __name__ == '__main__':
    main()

# --- START OF FILE SplitArrowToTxt.py ---

"""
Author: Shams
Description:
The dataset I got is in a format called Arrow (from Hugging Face). 
It's too big to open in a normal text editor.
So I wrote this to chop it up into 50 smaller text files that are easier to handle.
"""

import os
import math
from datasets import load_from_disk

ARROW_DIR = "etc-mini-(2gb)"

# How many pieces I want
NUM_FILES = 50

# Load the data
ds = load_from_disk(ARROW_DIR)
total = len(ds)
print(f"Found {total} examples.")

# Check the column name
field = "sentence" if "sentence" in ds.features else "text"
print(f"Using column: '{field}'")

# Calculate lines per file
lines_per_file = math.ceil(total / NUM_FILES)
print(f"Lines per file: {lines_per_file}")

OUT_DIR = os.path.join(ARROW_DIR, "ETC-ARZ")
os.makedirs(OUT_DIR, exist_ok=True)

file_idx = 0
line_idx = 0
fout = None

for i, example in enumerate(ds):
    # Start a new file if needed
    if line_idx == 0:
        if fout:
            fout.close()
        file_idx += 1
        filename = os.path.join(OUT_DIR, f"ETC-ARZ-{file_idx:02d}.txt")
        fout = open(filename, "w", encoding="utf-8")
    
    # Flatten the text
    text = example[field].replace("\n", " ").strip()
    fout.write(text + "\n")
    line_idx += 1

    # Switch file if full
    if line_idx >= lines_per_file:
        line_idx = 0

if fout:
    fout.close()

print(f"Done. Made {file_idx} files in '{OUT_DIR}/'")
