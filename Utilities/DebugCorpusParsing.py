"""
Author: Shams
Description:
This script is for debugging my dataset files.
It uses a very robust "Language Boundary Slice" method to separate English from Arabic.
It doesn't rely on a specific separator like "|||". Instead, it looks for the first
Arabic character on the line and splits the string there.
"""

import re
import os
from collections import Counter
import pandas as pd

def clean_text(text):
    # Removes special chars so I can count words accurately
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077Fa-zA-Z0-9\s]', '', text)
    return text.lower().strip()

def parse_corpus(file_path):
    """
    Reads the file line by line. It finds where the English ends and Arabic begins
    by looking for the first Arabic letter.
    """
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        return [], {}

    pairs = []
    skipped_details = { 'blank_lines': [], 'no_arabic_found': [] }

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            clean_line = line.strip()

            if not clean_line:
                skipped_details['blank_lines'].append((line_num, clean_line))
                continue
            
            # Find the first Arabic letter
            match = re.search(r'[\u0600-\u06FF]', clean_line)
            
            if not match:
                skipped_details['no_arabic_found'].append((line_num, clean_line))
                continue
            
            # Slice the string at that index
            first_arabic_index = match.start()
            eng_part_raw = clean_line[:first_arabic_index]
            ara_part_raw = clean_line[first_arabic_index:]

            # Clean up English part (remove ID numbers like "100. ")
            eng_part_cleaned = re.sub(r'^\d+[\s.-]*', '', eng_part_raw).strip()
            eng_sentence = eng_part_cleaned.rstrip(' ->"\'')
            
            ara_sentence = ara_part_raw.strip()

            if eng_sentence and ara_sentence:
                pairs.append((eng_sentence, ara_sentence))
            else:
                skipped_details['no_arabic_found'].append((line_num, clean_line))
                
    return pairs, skipped_details

def analyze_corpus(file_path):
    print(f"--- Analyzing: {file_path} ---\n")

    pairs, skipped_details = parse_corpus(file_path)
    total_skipped = sum(len(lines) for lines in skipped_details.values())
    
    df = pd.DataFrame(pairs, columns=['english', 'arabic'])
    
    print("1. Parsing Stats")
    print(f"Loaded Pairs: {len(df)}")
    print(f"Skipped Lines: {total_skipped}")
    print("\n")

    print("2. Length Stats")
    df['eng_len_char'] = df['english'].str.len()
    df['ara_len_char'] = df['arabic'].str.len()
    df['eng_len_word'] = df['english'].str.split().str.len()
    df['ara_len_word'] = df['arabic'].str.split().str.len()
    print(f"Avg English Length: {df['eng_len_word'].mean():.2f} words")
    print(f"Avg Arabic Length: {df['ara_len_word'].mean():.2f} words")
    print("\n")

    print("3. Duplicates")
    duplicate_counts = df.groupby(['english', 'arabic']).size().sort_values(ascending=False)
    unique_pairs = len(duplicate_counts)
    print(f"Unique Pairs: {unique_pairs}")
    print(f"Duplicates to remove: {len(df) - unique_pairs}")
    print("\n")

    print("4. Vocabulary")
    eng_text = ' '.join(df['english'].apply(clean_text))
    ara_text = ' '.join(df['arabic'].apply(clean_text))
    eng_vocab = Counter(eng_text.split())
    ara_vocab = Counter(ara_text.split())
    print(f"English Vocab Size: {len(eng_vocab)}")
    print(f"Arabic Vocab Size: {len(ara_vocab)}")
    print("\n")

    print("5. Extreme Ratios (Sanity Check)")
    # Identify sentences where one language is way longer than the other (likely bad data)
    df_ratio = df[(df['ara_len_word'] > 0) & (df['eng_len_word'] > 0)].copy()
    df_ratio['length_ratio'] = df_ratio['eng_len_word'] / df_ratio['ara_len_word']
    
    print("Highest EN/AR Ratios (English much longer):")
    for _, row in df_ratio.nlargest(5, 'length_ratio').iterrows():
        print(f"  - Ratio: {row['length_ratio']:.2f} | En: '{row['english']}' | Ar: '{row['arabic']}'")
    
    print("\n")
    
    print("6. Skipped Line Details")
    if total_skipped == 0:
        print("No errors found.")
    else:
        for category, lines in skipped_details.items():
            if not lines: continue
            print(f"Category: {category} ({len(lines)} lines)")
            for i, (line_num, content) in enumerate(lines):
                if i >= 5: break
                print(f"  Line {line_num}: '{content}'")
            print()

    print("done")

if __name__ == '__main__':
    file_path = 'Parallel-2.txt'
    analyze_corpus(file_path)
