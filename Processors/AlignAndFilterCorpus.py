"""
Author: Shams
Description:
I made this script to go through my folders and clean up the translation pairs.
It looks at the length of the sentences. If one is super long and the other is short, 
it's probably a bad translation, so I throw it out.
I also separate really long sentences into a different file so they don't mess up my main training data.
"""

import os
import re

# Settings
FOLDER_BASE_NAME = "egyptian_arabic_corpus"
SRC_FILENAME_BASE = "part"
TGT_FILENAME_BASE = "en"
FOLDER_START_NUM = 1
FOLDER_END_NUM = 12
FILE_START_NUM = 1
FILE_END_NUM = 10

# Where I'm saving stuff
ARABIC_OUTPUT_FILENAME = "corpus.ar"
ENGLISH_OUTPUT_FILENAME = "corpus.en"
LARGE_ARABIC_FILENAME = "large_corpus.ar"
LARGE_ENGLISH_FILENAME = "large_corpus.en"
REJECTED_LOG_FILENAME = "untranslated_log.txt"

# If a sentence is longer than this, I move it to the 'large' file
MAX_ACCEPTABLE_LENGTH = 750

line_pattern = re.compile(r'^(\d+)\.\s*(.*)')

# Keeping track of what I keep and what I toss
total_parallel_sentences = 0
total_large_sentences_isolated = 0
total_no_translation = 0
total_tier_p_rejected = 0 
total_tier_a_rejected = 0 
total_tier_b_rejected = 0 
total_tier_c_rejected = 0 
total_tier_d_rejected = 0 
total_tier_e_rejected = 0 
total_tier_f_rejected = 0 
total_single_word_rejected = 0 
total_empty_rejected = 0 

print("Starting the filtering process...")
print(f"Good data goes to '{ARABIC_OUTPUT_FILENAME}' and '{ENGLISH_OUTPUT_FILENAME}'.")
print(f"Really long stuff goes to '{LARGE_ARABIC_FILENAME}'.")
print(f"Bad matches go to '{REJECTED_LOG_FILENAME}'.\n")


def get_rejection_reason(ar_len, en_len, ratio):
    """
    I use this to figure out if the length difference between Arabic and English is too weird.
    Different lengths allow for different ratios.
    """
    # If it's huge, just flag it as too long
    if ar_len > MAX_ACCEPTABLE_LENGTH or en_len > MAX_ACCEPTABLE_LENGTH:
        return "too_long", f"Sentence exceeds max length ({MAX_ACCEPTABLE_LENGTH} words)"

    # Checking ratios based on how many words there are
    if ar_len == 0:
        return "empty", "Empty sentence"
    elif ar_len == 1:
        if ratio > 4.0 or ratio < (1.0/4.0):
            return "single_word", f"Single word, extreme ratio ({ratio:.2f})"
    elif ar_len <= 3:
        if ratio > 3.5 or ratio < (1.0/3.5):
            return "tier_f", f"Tier F (2-3w), ratio issue ({ratio:.2f})"
    elif ar_len <= 7:
        if ratio > 3.0 or ratio < (1.0/3.0):
            return "tier_e", f"Tier E (4-7w), ratio issue ({ratio:.2f})"
    elif ar_len <= 15:
        if ratio > 2.4 or ratio < (1.0/2.4):
            return "tier_d", f"Tier D (8-15w), ratio issue ({ratio:.2f})"
    elif ar_len <= 40:
        if ratio > 2.0 or ratio < (1.0/2.0):
            return "tier_c", f"Tier C (16-40w), ratio issue ({ratio:.2f})"
    elif ar_len <= 100:
        if ratio > 1.8 or ratio < (1.0/1.8):
            return "tier_b", f"Tier B (41-100w), ratio issue ({ratio:.2f})"
    elif ar_len <= 200:
        if ratio > 1.6 or ratio < (1.0/1.6):
            return "tier_a", f"Tier A (101-200w), ratio issue ({ratio:.2f})"
    else:
        # This is for the really long paragraphs
        if ratio > 1.5 or ratio < (1.0/1.5):
            return "tier_p", f"Tier P (201-{MAX_ACCEPTABLE_LENGTH}w), ratio issue ({ratio:.2f})"

    return None, None


try:
    with open(ARABIC_OUTPUT_FILENAME, 'w', encoding='utf-8') as ar_out, \
         open(ENGLISH_OUTPUT_FILENAME, 'w', encoding='utf-8') as en_out, \
         open(LARGE_ARABIC_FILENAME, 'w', encoding='utf-8') as ar_large_out, \
         open(LARGE_ENGLISH_FILENAME, 'w', encoding='utf-8') as en_large_out, \
         open(REJECTED_LOG_FILENAME, 'w', encoding='utf-8') as log_out:

        for i in range(FOLDER_START_NUM, FOLDER_END_NUM + 1):
            folder_name = f"{FOLDER_BASE_NAME}_{i:04d}"
            folder_path = os.path.join(os.getcwd(), folder_name)
            if not os.path.isdir(folder_path): continue
            print(f"Processing folder: '{folder_name}'")
            
            # Read the English files first to build a map
            translations_map = {}
            for j in range(FILE_START_NUM, FILE_END_NUM + 1):
                tgt_filepath = os.path.join(folder_path, f"{TGT_FILENAME_BASE}{j}.txt")
                if not os.path.exists(tgt_filepath): continue
                with open(tgt_filepath, 'r', encoding='utf-8') as tgt_file:
                    for line in tgt_file:
                        match = line_pattern.match(line)
                        if match: translations_map[int(match.group(1))] = match.group(2).strip()
            
            # Now match them with the Arabic files
            for k in range(FILE_START_NUM, FILE_END_NUM + 1):
                src_filepath = os.path.join(folder_path, f"{SRC_FILENAME_BASE}_{k}.txt")
                if not os.path.exists(src_filepath): continue
                
                with open(src_filepath, 'r', encoding='utf-8') as src_file:
                    for line in src_file:
                        match = line_pattern.match(line)
                        if not match: continue
                        
                        local_line_num = int(match.group(1))
                        arabic_text = match.group(2).strip()
                        english_text = translations_map.get(local_line_num)
                        
                        # If I can't find the English version, log it and skip
                        if not english_text:
                            log_out.write(f"NO_TRANSLATION: {arabic_text}\n")
                            total_no_translation += 1
                            continue

                        ar_len = len(arabic_text.split())
                        en_len = len(english_text.split())
                        
                        # Move giant sentences to the separate file
                        if ar_len > MAX_ACCEPTABLE_LENGTH or en_len > MAX_ACCEPTABLE_LENGTH:
                            ar_large_out.write(f"{arabic_text}\n")
                            en_large_out.write(f"{english_text}\n")
                            total_large_sentences_isolated += 1
                            continue

                        # Check if the length ratio is suspicious
                        ratio = (en_len / ar_len) if ar_len > 0 else float('inf')
                        rejection_type, reason = get_rejection_reason(ar_len, en_len, ratio)
                        
                        if rejection_type:
                            log_out.write(f"{rejection_type.upper()}: [AR] {arabic_text} || [EN] {english_text} | Reason: {reason}\n")
                            if rejection_type == "empty": total_empty_rejected += 1
                            elif rejection_type == "single_word": total_single_word_rejected += 1
                            elif rejection_type == "tier_f": total_tier_f_rejected += 1
                            elif rejection_type == "tier_e": total_tier_e_rejected += 1
                            elif rejection_type == "tier_d": total_tier_d_rejected += 1
                            elif rejection_type == "tier_c": total_tier_c_rejected += 1
                            elif rejection_type == "tier_b": total_tier_b_rejected += 1
                            elif rejection_type == "tier_a": total_tier_a_rejected += 1
                            elif rejection_type == "tier_p": total_tier_p_rejected += 1
                        else:
                            # It's good, save it
                            ar_out.write(f"{arabic_text}\n")
                            en_out.write(f"{english_text}\n")
                            total_parallel_sentences += 1

    print("\n" + "="*60)
    print("Done Filtering")
    print("="*60)
    print(f"\nTotal good sentences: {total_parallel_sentences}")
    print(f"Sentences moved because they were too big: {total_large_sentences_isolated}")

    print("\n--- Why sentences got rejected ---")
    print(f"  No translation: {total_no_translation}")
    print(f"  Empty: {total_empty_rejected}")
    print(f"  Single word weirdness: {total_single_word_rejected}")
    print(f"  Tier F (2-3 words): {total_tier_f_rejected}")
    print(f"  Tier E (4-7 words): {total_tier_e_rejected}")
    print(f"  Tier D (8-15 words): {total_tier_d_rejected}")
    print(f"  Tier C (16-40 words): {total_tier_c_rejected}")
    print(f"  Tier B (41-100 words): {total_tier_b_rejected}")
    print(f"  Tier A (101-200 words): {total_tier_a_rejected}")
    print(f"  Tier P (201+ words): {total_tier_p_rejected}")
    
    total_rejected = (total_no_translation + total_empty_rejected + total_single_word_rejected + 
                     total_tier_a_rejected + total_tier_b_rejected + total_tier_c_rejected + 
                     total_tier_d_rejected + total_tier_e_rejected + total_tier_f_rejected +
                     total_tier_p_rejected)

    print(f"Total rejected: {total_rejected}")
    
    total_processed = total_parallel_sentences + total_large_sentences_isolated + total_rejected
    if total_processed > 0:
        acceptance_rate = (total_parallel_sentences / total_processed) * 100
        print(f"Acceptance rate: {acceptance_rate:.2f}%")
    
    print("="*60)

except Exception as e:
    print(f"\nSomething went wrong: {e}")
    import traceback
    traceback.print_exc()
