"""
Author: Shams
Description:
This script checks the basic health of my processed text files.
It looks for duplicates, weird length ratios, and counts words.
It's a quick sanity check to see if the cleaning worked.
"""

import os
import re
import time
from collections import Counter
import pandas as pd
import glob

# Settings
EXTREME_RATIO_THRESHOLD = 3.0 
TOP_N_DUPLICATES = 50         
DELAY_BETWEEN_FILES = 20      

# --- Functions ---

def clean_text_for_vocab(text):
    """
    Just scrubbing text so I can count unique words properly.
    """
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077Fa-zA-Z0-9\s]', '', text)
    return text.lower().strip()

def parse_corpus(file_path):
    """
    I need to split the lines carefully. I use the first Arabic character
    as the cutting point between English and Arabic.
    """
    pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line: continue
                
                # Find where Arabic starts
                match = re.search(r'[\u0600-\u06FF]', clean_line)
                if not match: continue
                
                first_arabic_index = match.start()
                eng_part_raw = clean_line[:first_arabic_index]
                ara_part_raw = clean_line[first_arabic_index:]

                # Clean up the English side (remove IDs)
                eng_part_cleaned = re.sub(r'^\d+[\s.-]*', '', eng_part_raw).strip()
                eng_sentence = eng_part_cleaned.rstrip(' ->"\'|')
                ara_sentence = ara_part_raw.strip()

                if eng_sentence and ara_sentence:
                    pairs.append((eng_sentence, ara_sentence))
    except FileNotFoundError:
        print(f"Error: Can't find {file_path}")
        return pd.DataFrame() 
                
    return pd.DataFrame(pairs, columns=['english', 'arabic'])

def run_analysis(df, title="Analysis Report"):
    """
    This runs all the math on the data.
    """
    if df.empty:
        print("Empty dataframe. Can't analyze.")
        return

    print("\n" + "="*70)
    print(f"||   {title.upper()}   ||")
    print("="*70 + "\n")

    # 1. Stats
    print("--- 1. Stats ---")
    df['eng_len_word'] = df['english'].str.split().str.len()
    df['ara_len_word'] = df['arabic'].str.split().str.len()
    print(f"Total pairs: {len(df):,}")
    print(f"Avg English length: {df['eng_len_word'].mean():.2f} words")
    print(f"Avg Arabic length: {df['ara_len_word'].mean():.2f} words")
    print("\n")

    # 2. Duplicates
    print("--- 2. Duplicate Analysis ---")
    total_pairs = len(df)
    
    duplicate_counts = df.groupby(['english', 'arabic']).size()
    unique_pairs_count = len(duplicate_counts)

    print(f"Scenario A: Unique pairs only.")
    print(f"  - Count: {unique_pairs_count:,}")
    
    # Cap duplicates at 2
    capped_counts = duplicate_counts.clip(upper=2)
    size_with_one_duplicate = capped_counts.sum()
    print(f"Scenario B: Allow max 1 duplicate.")
    print(f"  - Count: {size_with_one_duplicate:,}")
    
    if unique_pairs_count < total_pairs:
        print(f"\nTop {TOP_N_DUPLICATES} frequent duplicates:")
        for (eng, ara), count in duplicate_counts.sort_values(ascending=False).head(TOP_N_DUPLICATES).items():
            print(f"  - Count: {count} | Eng: '{eng}' | Ara: '{ara}'")
    else:
        print("\nNo duplicates found.")
    print("\n")

    # 3. Ratios
    print(f"--- 3. Length Ratios (Limit: {EXTREME_RATIO_THRESHOLD:.1f}x) ---")
    df_ratio = df.drop_duplicates(subset=['english', 'arabic']) 
    df_ratio = df_ratio[(df_ratio['ara_len_word'] > 0) & (df_ratio['eng_len_word'] > 0)].copy()
    df_ratio['ratio'] = df_ratio['eng_len_word'] / df_ratio['ara_len_word']
    
    high_ratio_pairs = df_ratio[df_ratio['ratio'] > EXTREME_RATIO_THRESHOLD]
    low_ratio_pairs = df_ratio[df_ratio['ratio'] < (1 / EXTREME_RATIO_THRESHOLD)]
    
    print(f"English way longer than Arabic: {len(high_ratio_pairs):,} pairs")
    print(f"Arabic way longer than English: {len(low_ratio_pairs):,} pairs")

    if not high_ratio_pairs.empty:
        print("\nExamples (English is long):")
        for _, row in high_ratio_pairs.head(5).iterrows():
            print(f"  - Ratio: {row['ratio']:.1f}x | Eng({row['eng_len_word']}w): '{row['english']}' | Ara({row['ara_len_word']}w): '{row['arabic']}'")
    
    if not low_ratio_pairs.empty:
        print("\nExamples (Arabic is long):")
        for _, row in low_ratio_pairs.head(5).iterrows():
            print(f"  - Ratio: {1/row['ratio']:.1f}x | Ara({row['ara_len_word']}w): '{row['arabic']}' | Eng({row['eng_len_word']}w): '{row['english']}'")
    print("\n")

def main():
    # Grab all files ending in -F.txt
    target_files = sorted(glob.glob("*-F.txt"))
    
    if not target_files:
        print("Error: No '*-F.txt' files found.")
        return

    print(f"Found {len(target_files)} files to analyze.")
    all_dataframes = []

    for i, file_path in enumerate(target_files):
        print("\n" + "#"*70)
        print(f"# Processing: {file_path} ({i+1}/{len(target_files)})")
        print("#"*70)

        df = parse_corpus(file_path)
        if not df.empty:
            run_analysis(df, title=f"Analysis for {os.path.basename(file_path)}")
            all_dataframes.append(df)
        
        # Taking a breath between files
        if i < len(target_files) - 1:
            print(f"\n--- Waiting {DELAY_BETWEEN_FILES} seconds... ---")
            time.sleep(DELAY_BETWEEN_FILES)

    # --- Combined Stats ---
    if not all_dataframes:
        print("No data parsed.")
        return
        
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    run_analysis(combined_df, title="Final Combined Analysis")

if __name__ == '__main__':
    main()
