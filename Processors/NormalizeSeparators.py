"""
Author: Shams
Description:
I have a bunch of text files where English and Arabic are separated by different things.
Some use " -> ", some use " - ".
I want them all to look the same: "ID. English ||| Arabic".
This script fixes that.
"""

import os
import re
import sys
import argparse
from collections import Counter

# The separator I want to use
UNIQUE_SEPARATOR = "|||"
# The separators I might find in the wild
POSSIBLE_SEPARATORS = [' -> ', ' - ']
DEFAULT_FILES_TO_PROCESS = [
    "arzen-llm.txt",
    "ArzEn-MultiGenre.txt",
    "Parallel-2.txt",
    "ar-eng6.txt",
    "ar-eng4.txt",
    "ar-eng5.txt",
    "Parallel-1.txt",
    "ar-eng3.txt",
    "ar-eng2.txt",
    "ar-eng1.txt",
    "ArzEn-Corpus.txt",
]

def calculate_text_stats(sentences):
    """Just counting words and letters to see what the data looks like."""
    if not sentences:
        return {'avg_words': 0, 'avg_chars': 0, 'vocab_size': 0, 'word_count': 0}
    
    total_words, total_chars, all_words = 0, 0, []
    for s in sentences:
        words = s.split()
        total_words += len(words)
        total_chars += len(s)
        cleaned_words = [re.sub(r'[^\w\s]', '', w).lower() for w in words if w]
        all_words.extend(cleaned_words)
    
    num_sentences = len(sentences)
    avg_words = total_words / num_sentences if num_sentences > 0 else 0
    avg_chars = total_chars / num_sentences if num_sentences > 0 else 0
    vocab_size = len(set(all_words))

    return {
        'avg_words': avg_words, 'avg_chars': avg_chars,
        'vocab_size': vocab_size, 'word_count': total_words
    }

def process_file(input_path):
    print(f"\n{'='*20}\nWorking on: {input_path}\n{'='*20}")

    base_name, ext = os.path.splitext(input_path)
    output_path = f"{base_name}-F{ext}"
    
    skipped_lines = {"no_separator": [], "empty_after_clean": []}
    original_eng, original_arz, processed_eng, processed_arz = [], [], [], []
    line_count_original, new_id = 0, 1

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                line_count_original += 1
                original_line = line.strip()
                if not original_line: continue

                parts = None
                # Try to find which separator is being used
                for sep in POSSIBLE_SEPARATORS:
                    if sep in original_line:
                        temp_parts = original_line.rsplit(sep, 1)
                        if len(temp_parts) == 2:
                            parts = temp_parts
                            break
                
                if parts:
                    eng_part, arz_part = parts
                    # Clean up the English part (remove old numbers)
                    cleaned_eng = re.sub(r'^\d+[\.\-]?\s*', '', eng_part)
                    eng_sentence = cleaned_eng.strip().strip('"- ')
                    arz_sentence = arz_part.strip()
                    
                    if eng_sentence and arz_sentence:
                        original_eng.append(eng_part)
                        original_arz.append(arz_part)
                        # Write it in the new format
                        new_line = f"{new_id}. {eng_sentence} {UNIQUE_SEPARATOR} {arz_sentence}\n"
                        outfile.write(new_line)
                        processed_eng.append(eng_sentence)
                        processed_arz.append(arz_sentence)
                        new_id += 1
                    else:
                        skipped_lines["empty_after_clean"].append(original_line)
                else:
                    skipped_lines["no_separator"].append(original_line)
        print(f"Made new file: {output_path}")

    except FileNotFoundError:
        print(f"Error: Can't find {input_path}")
        return None
    except Exception as e:
        print(f"Weird error processing {input_path}: {e}")
        return None
        
    stats_original_eng = calculate_text_stats(original_eng)
    stats_original_arz = calculate_text_stats(original_arz)
    stats_processed_eng = calculate_text_stats(processed_eng)
    stats_processed_arz = calculate_text_stats(processed_arz)

    file_stats = {
        "input_file": input_path, "output_file": output_path,
        "skipped_lines_list": skipped_lines,
        "counts": {
            "original_total_lines": line_count_original, "processed_pairs": len(processed_eng),
            "skipped_no_separator": len(skipped_lines["no_separator"]),
            "skipped_empty_after_clean": len(skipped_lines["empty_after_clean"])
        },
        "original_stats": {"eng": stats_original_eng, "arz": stats_original_arz},
        "processed_stats": {"eng": stats_processed_eng, "arz": stats_processed_arz}
    }
    return file_stats

def print_eda_report(stats):
    if not stats: return
    
    counts = stats['counts']
    total_skipped = counts['skipped_no_separator'] + counts['skipped_empty_after_clean']
    
    print("\n--- Report ---")
    print(f"Input: {stats['input_file']} | Output: {stats['output_file']}")
    
    print("\n[Summary]")
    print(f"  - Original lines: {counts['original_total_lines']:,}")
    print(f"  - Kept lines: {counts['processed_pairs']:,}")
    print(f"  - Skipped lines: {total_skipped:,}")
    print(f"    - No separator found: {counts['skipped_no_separator']:,}")
    print(f"    - Empty after clean: {counts['skipped_empty_after_clean']:,}")

    print("\n[Original Stats]")
    o_eng, o_arz = stats['original_stats']['eng'], stats['original_stats']['arz']
    print(f"  English: Avg Words: {o_eng['avg_words']:.2f}, Vocab: {o_eng['vocab_size']:,}")
    print(f"  Arabic:  Avg Words: {o_arz['avg_words']:.2f}, Vocab: {o_arz['vocab_size']:,}")

    print("\n[New Stats]")
    p_eng, p_arz = stats['processed_stats']['eng'], stats['processed_stats']['arz']
    print(f"  English: Avg Words: {p_eng['avg_words']:.2f}, Vocab: {p_eng['vocab_size']:,}")
    print(f"  Arabic:  Avg Words: {p_arz['avg_words']:.2f}, Vocab: {p_arz['vocab_size']:,}")

def main():
    parser = argparse.ArgumentParser(
        description="Fixes separators in text files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'input_files', metavar='FILE', type=str, nargs='*',
        help='Files to process. If empty, uses default list.'
    )
    args = parser.parse_args()
    
    files_to_process = args.input_files if args.input_files else DEFAULT_FILES_TO_PROCESS
    print(f"--- Processing {len(files_to_process)} file(s) ---")

    overall_stats = {
        "total_files": 0, "total_lines_original": 0, "total_lines_processed": 0,
        "total_skipped_no_separator": 0, "total_skipped_empty": 0,
        "all_skipped_lines": {}, "total_words_eng": 0, "total_words_arz": 0
    }
    
    for file_path in files_to_process:
        stats = process_file(file_path)
        if stats:
            print_eda_report(stats)
            overall_stats["total_files"] += 1
            overall_stats["total_lines_original"] += stats["counts"]["original_total_lines"]
            overall_stats["total_lines_processed"] += stats["counts"]["processed_pairs"]
            overall_stats["total_skipped_no_separator"] += stats["counts"]["skipped_no_separator"]
            overall_stats["total_skipped_empty"] += stats["counts"]["skipped_empty_after_clean"]
            overall_stats["all_skipped_lines"][file_path] = stats["skipped_lines_list"]
            overall_stats["total_words_eng"] += stats["processed_stats"]["eng"]["word_count"]
            overall_stats["total_words_arz"] += stats["processed_stats"]["arz"]["word_count"]

    total_skipped_overall = overall_stats["total_skipped_no_separator"] + overall_stats["total_skipped_empty"]
    print(f"\n\n{'='*25}\n|| TOTAL SUMMARY ||\n{'='*25}")
    if overall_stats["total_files"] > 0:
        print(f"Total Lines Original: {overall_stats['total_lines_original']:,}")
        print(f"Total Lines Processed: {overall_stats['total_lines_processed']:,}")
        print("-" * 25)
        
        print(f"Files Processed: {overall_stats['total_files']}")
        print(f"Total Skipped: {total_skipped_overall:,}")
        print(f"  - No separator: {overall_stats['total_skipped_no_separator']:,}")
        print(f"  - Empty: {overall_stats['total_skipped_empty']:,}")
        print("-" * 25)
        print(f"Total English Words: {overall_stats['total_words_eng']:,}")
        print(f"Total Arabic Words: {overall_stats['total_words_arz']:,}")
    else:
        print("No files processed.")

    if total_skipped_overall > 0:
        print(f"\n\n{'!'*25}\n|| SKIPPED LINES ||\n{'!'*25}")
        for filename, skips in overall_stats["all_skipped_lines"].items():
            printed_filename = False
            if skips['no_separator']:
                print(f"\n--- From file: {filename} (No Separator) ---")
                printed_filename = True
                for line in skips['no_separator']:
                    print(f"  -> {line}")
            if skips['empty_after_clean']:
                if not printed_filename: print(f"\n--- From file: {filename} ---")
                print(f"--- (Empty After Clean) ---")
                for line in skips['empty_after_clean']:
                    print(f"  -> {line}")
    else:
        print("\n\nNice! Nothing was skipped.")

if __name__ == "__main__":
    main()
