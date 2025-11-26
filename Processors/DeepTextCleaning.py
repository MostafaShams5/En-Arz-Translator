"""
Author: Shams
Description:
This is the heavy-duty cleaner.
It removes things like subtitles (e.g. "She sighs"), repeated punctuation "...",
and filters out lines based on specific rules to reduce noise.
"""

import os
import re
import pandas as pd
import glob

def deep_clean_text(text, is_arabic=False):
    """
    Scrubbing the text clean.
    """
    # Remove # and ???
    text = re.sub(r'#|\?{3,}', '', text)
    
    # Remove things in brackets like (laughing)
    if is_arabic:
        text = re.sub(r'\([\u0600-\u06FF\s.]+\)', '', text)
    else:
        text = re.sub(r"''", "'", text)
        text = re.sub(r'\([\w\s.\']+\)', '', text)
        
    # Fix ellipses and strip junk
    text = re.sub(r'\s*\.{2,}\s*', ' ... ', text)
    text = text.strip(' "-*|.')

    # Fix spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def parse_and_clean_corpus(file_path):
    """
    Reads the file and cleans each line.
    """
    pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line or '|||' not in clean_line:
                    continue
                
                # Split at the last separator just in case
                parts = clean_line.rsplit('|||', 1)
                if len(parts) != 2:
                    continue

                eng_raw, ara_raw = parts

                # Remove ID
                eng_no_id = re.sub(r'^\d+[\s.-]*', '', eng_raw).strip()

                # Clean both sides
                eng_sentence = deep_clean_text(eng_no_id, is_arabic=False)
                ara_sentence = deep_clean_text(ara_raw, is_arabic=True)

                if eng_sentence and ara_sentence:
                    pairs.append((eng_sentence, ara_sentence))
    except FileNotFoundError:
        print(f"Error: Can't find {file_path}")
        return pd.DataFrame()
                
    return pd.DataFrame(pairs, columns=['english', 'arabic'])

def apply_processing_rules(df, filename):
    """
    Filtering out bad ratios and limiting duplicates.
    """
    initial_count = len(df)
    initial_unique_count = len(df.drop_duplicates(subset=['english', 'arabic']))
    
    print("--- Before Processing ---")
    print(f"  - Total: {initial_count:,}")
    print(f"  - Unique: {initial_unique_count:,}")

    df['eng_words'] = df['english'].str.split().str.len()
    df['ara_words'] = df['arabic'].str.split().str.len()
    
    df_ratio = df[(df['eng_words'] > 0) & (df['ara_words'] > 0)].copy()
    
    df_ratio['eng_ara_ratio'] = df_ratio['eng_words'] / df_ratio['ara_words']
    df_ratio['ara_eng_ratio'] = df_ratio['ara_words'] / df_ratio['eng_words']
    
    count_before_filter = len(df_ratio)
    
    # Specific rules for specific files
    if "ArzEn-Corpus" in filename:
        df_ratio = df_ratio[~(df_ratio['ara_eng_ratio'] > 3.0)]
        df_ratio = df_ratio[~((df_ratio['eng_ara_ratio'] >= 3.0) & (df_ratio['ara_words'] > 3))]
    elif "ArzEn-MultiGenre" in filename:
        df_ratio = df_ratio[~(df_ratio['ara_eng_ratio'] > 3.0)]
        df_ratio = df_ratio[~((df_ratio['eng_ara_ratio'] >= 3.0) & (df_ratio['ara_words'] > 3))]
        df_ratio = df_ratio[~(df_ratio['eng_ara_ratio'] > 5.0)]
    else:
        df_ratio = df_ratio[~(df_ratio['ara_eng_ratio'] > 3.0)]
        df_ratio = df_ratio[~((df_ratio['eng_ara_ratio'] >= 3.0) & (df_ratio['ara_words'] > 3))]
    
    count_after_filter = len(df_ratio)
    
    # Downsampling (don't keep too many copies of the same sentence)
    pair_counts = df_ratio.groupby(['english', 'arabic']).size()
    
    processed_pairs = []
    
    for (eng, ara), count in pair_counts.items():
        target_count = count
        
        if "ArzEn-MultiGenre" in filename:
            if count > 10: target_count = 10
        else:
            if count < 5: target_count = 1
            elif 5 <= count <= 100: target_count = 1
            elif 101 <= count <= 300: target_count = 1
            elif count > 300: target_count = 1
        
        processed_pairs.extend([(eng, ara)] * target_count)
        
    final_df = pd.DataFrame(processed_pairs, columns=['english', 'arabic'])
    
    final_count = len(final_df)
    final_unique_count = len(final_df.drop_duplicates(subset=['english', 'arabic']))
    
    pairs_removed_by_ratio = count_before_filter - count_after_filter
    pairs_removed_by_downsampling = count_after_filter - final_count

    print("\n--- After Processing ---")
    print(f"  - Removed by ratio: {pairs_removed_by_ratio:,}")
    print(f"  - Removed by capping duplicates: {pairs_removed_by_downsampling:,}")
    print(f"  - Total removed: {pairs_removed_by_ratio + pairs_removed_by_downsampling:,}")
    print("-" * 35)
    print(f"  - Final total: {final_count:,}")
    print(f"  - Final unique: {final_unique_count:,}")
    
    return final_df

def main():
    target_files = sorted(glob.glob("*-F.txt"))
    
    if not target_files:
        print("Error: No '*-F.txt' files found.")
        return

    print(f"Found {len(target_files)} files to deep clean.")

    for file_path in target_files:
        base_name = os.path.basename(file_path).replace("-F.txt", "")
        output_filename = f"{base_name}-P.txt"
        
        print("\n" + "="*70)
        print(f"Processing: {file_path}")
        print("="*70)

        df = parse_and_clean_corpus(file_path)
        if df.empty:
            print(f"  - Skipping {file_path}, it's empty.")
            continue

        if "arzen-llm" in file_path:
            print("  - Rule: Just deep clean this one, no filtering.")
        else:
            df = apply_processing_rules(df, base_name)

        print(f"\nSaving {len(df):,} pairs to {output_filename}...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            # Sort it so it's tidy
            df = df.sort_values(by=['english', 'arabic']).reset_index(drop=True)
            for index, row in df.iterrows():
                f.write(f"{index + 1}. {row['english']} ||| {row['arabic']}\n")
        
        print(f"Saved: {output_filename}")

    print("\n" + "="*70)
    print("All done.")

if __name__ == '__main__':
    main()
