"""
Author: Shams
Description:
This script is an advanced processing pipeline for Egyptian Arabic text.
It reads a raw text file line-by-line and applies spelling corrections (using a custom dictionary),
normalizes numbers (Eastern to Western), removes diacritics, and filters out junk phrases.
It produces three outputs: the final clean text, a modification log (showing exactly what changed),
and a basic statistical analysis report.
"""

import re
import sys
from collections import Counter
from pathlib import Path
from corrections_arz import ARZ_SAFE_CORRECTIONS

INPUT_FILE = "cleaned_ETC-ARZ.txt"
FINAL_CLEAN_FILE = "ETC-ARZ_final.txt"
MODIFIED_SENTENCES_REPORT = "modification_ETC-ARZ.txt"
EDA_REPORT = "eda_analysis_ETC-ARZ.txt"

ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF]')
ARABIC_WORD_PATTERN = re.compile(r'[\u0621-\u064A]+')

ARABIC_JUNK_PHRASES = [
    "فعل زر الجرس", 
    "تشترك في القناة",
    "اضغط على زر الإعجاب",
    "شير للفيديو"
]

MIN_WORD_FREQUENCY = 2
TOP_N_WORDS = 50

DELETE_SINGLE_WORD_SENTENCES = True
DELETE_NON_ARABIC_SENTENCES = True

class ArabicTextEDA:
    def __init__(self):
        self.stats = {
            'total_lines': 0,
            'empty_lines': 0,
            'lines_with_arabic': 0,
            'lines_without_arabic': 0,
            'average_line_length': 0,
            'total_words': 0,
            'unique_words': set(),
            'word_frequency': Counter(),
            'sentence_length_distribution': Counter(),
            'min_sentence_length': float('inf'),
            'max_sentence_length': 0,
            'total_words_for_avg': 0,
            'total_lines_for_avg': 0,
            'spelling_corrections_applied': Counter(),
            'deletion_reasons': Counter(),
            'lines_modified': 0,
            'lines_deleted': 0,
            'lines_kept': 0
        }
    
    def analyze_line(self, line: str, is_original: bool = True):
        if not line.strip():
            if is_original:
                self.stats['empty_lines'] += 1
            return
        
        if is_original:
            self.stats['total_lines'] += 1
        
        has_arabic = bool(ARABIC_PATTERN.search(line))
        if is_original:
            if has_arabic:
                self.stats['lines_with_arabic'] += 1
            else:
                self.stats['lines_without_arabic'] += 1
        
        words = ARABIC_WORD_PATTERN.findall(line)
        if is_original:
            self.stats['total_words'] += len(words)
            self.stats['unique_words'].update(words)
            self.stats['word_frequency'].update(words)
        
        word_count = len(line.split())
        if is_original and word_count > 0:
            self.stats['sentence_length_distribution'][word_count] += 1
            self.stats['total_words_for_avg'] += word_count
            self.stats['total_lines_for_avg'] += 1
            if word_count < self.stats['min_sentence_length']:
                self.stats['min_sentence_length'] = word_count
            if word_count > self.stats['max_sentence_length']:
                self.stats['max_sentence_length'] = word_count
    
    def record_spelling_changes(self, changes: dict):
        for original, correction in changes.items():
            self.stats['spelling_corrections_applied'][f"{original} -> {correction}"] += 1
    
    def record_deletion(self, reason: str):
        self.stats['deletion_reasons'][reason] += 1
        self.stats['lines_deleted'] += 1
    
    def record_modification(self):
        self.stats['lines_modified'] += 1
    
    def record_kept_line(self):
        self.stats['lines_kept'] += 1
    
    def calculate_final_stats(self):
        if self.stats['total_lines_for_avg'] > 0:
            self.stats['average_line_length'] = self.stats['total_words_for_avg'] / self.stats['total_lines_for_avg']
        self.stats['unique_word_count'] = len(self.stats['unique_words'])
        self.stats['unique_words'] = list(self.stats['unique_words'])
    
    def generate_report(self, output_file: str):
        self.calculate_final_stats()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== ARABIC TEXT PROCESSING - EXPLORATORY DATA ANALYSIS ===\n\n")
            f.write("--- BASIC STATISTICS ---\n")
            f.write(f"Total lines processed: {self.stats['total_lines']}\n")
            f.write(f"Empty lines skipped: {self.stats['empty_lines']}\n")
            f.write(f"Lines with Arabic content: {self.stats['lines_with_arabic']}\n")
            f.write(f"Lines without Arabic content: {self.stats['lines_without_arabic']}\n")
            f.write(f"Average sentence length (words): {self.stats['average_line_length']:.2f}\n")
            f.write(f"Total words found: {self.stats['total_words']}\n")
            f.write(f"Unique words found: {self.stats['unique_word_count']}\n\n")
            f.write("--- PROCESSING RESULTS ---\n")
            f.write(f"Lines kept: {self.stats['lines_kept']}\n")
            f.write(f"Lines deleted: {self.stats['lines_deleted']}\n")
            f.write(f"Lines modified (but kept): {self.stats['lines_modified']}\n")
            
            if self.stats['lines_kept'] + self.stats['lines_deleted'] > 0:
                retention_rate = (self.stats['lines_kept'] / (self.stats['lines_kept'] + self.stats['lines_deleted'])) * 100
                f.write(f"Retention rate: {retention_rate:.1f}%\n\n")
            
            if self.stats['deletion_reasons']:
                f.write("--- DELETION REASONS ---\n")
                for reason, count in self.stats['deletion_reasons'].most_common():
                    f.write(f"{reason}: {count} lines\n")
                f.write("\n")
            
            if self.stats['spelling_corrections_applied']:
                f.write("--- TOP SPELLING CORRECTIONS ---\n")
                for correction, count in self.stats['spelling_corrections_applied'].most_common(2000):
                    f.write(f"{correction}: {count} times\n")
                f.write("\n")
            
            if self.stats['word_frequency']:
                f.write(f"--- TOP {TOP_N_WORDS} MOST FREQUENT WORDS ---\n")
                for word, count in self.stats['word_frequency'].most_common(TOP_N_WORDS):
                    if count >= MIN_WORD_FREQUENCY:
                        f.write(f"{word}: {count} occurrences\n")
                f.write("\n")
            
            if self.stats['sentence_length_distribution']:
                length_dist = self.stats['sentence_length_distribution']
                f.write("--- SENTENCE LENGTH DISTRIBUTION ---\n")
                min_len = self.stats['min_sentence_length'] if self.stats['min_sentence_length'] != float('inf') else 0
                max_len = self.stats['max_sentence_length']
                f.write(f"Min length: {min_len} words\n")
                f.write(f"Max length: {max_len} words\n")
                f.write(f"Most common lengths:\n")
                for length, count in length_dist.most_common(10):
                    f.write(f"  {length} words: {count} sentences\n")

def correct_and_normalize(line: str, corrections_dict: dict) -> tuple[str, dict]:
    spelling_changes = {}
    temp_line = line.strip()
    
    if not temp_line:
        return temp_line, spelling_changes
    
    words = ARABIC_WORD_PATTERN.findall(temp_line)
    for word in set(words):
        if word in corrections_dict:
            correction = corrections_dict[word]
            if word != correction:
                pattern = r'(?<!\S)' + re.escape(word) + r'(?!\S)'
                temp_line = re.sub(pattern, correction, temp_line)
                spelling_changes[word] = correction

    temp_line = re.sub(r'<[^>]*>', '', temp_line)
    temp_line = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', temp_line)
    
    eastern_to_western = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for east, west in eastern_to_western.items():
        temp_line = temp_line.replace(east, west)
    
    temp_line = re.sub(r'\s+([.,?!:;،؟؛])', r'\1', temp_line)
    temp_line = re.sub(r'([.,?!:;،؟؛])\1+', r'\1', temp_line)
    temp_line = re.sub(r'\bو\s+', 'و', temp_line)
    temp_line = re.sub(r'\s+', ' ', temp_line).strip()
    
    return temp_line, spelling_changes

def check_for_deletion(sentence: str) -> tuple[bool, str]:
    if not sentence.strip():
        return True, "Empty sentence after processing"
    
    if DELETE_NON_ARABIC_SENTENCES and not ARABIC_PATTERN.search(sentence):
        return True, "Contains no Arabic characters (100% non-Arabic)"

    for phrase in ARABIC_JUNK_PHRASES:
        if phrase in sentence:
            return True, f"Contains promotional phrase: '{phrase}'"
    
    if DELETE_SINGLE_WORD_SENTENCES:
        word_count = len(sentence.split())
        if word_count <= 1:
            return True, "Sentence consists of only one word"
    
    return False, None

def run_pipeline():
    print("--- Starting Enhanced Arabic Text Processing Pipeline ---")
    
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"ERROR: The input file '{INPUT_FILE}' was not found.")
        return
    
    print(f"Configuration:")
    print(f"  - Delete single-word sentences: {'YES' if DELETE_SINGLE_WORD_SENTENCES else 'NO'}")
    print(f"  - Delete non-Arabic sentences: {'YES' if DELETE_NON_ARABIC_SENTENCES else 'NO'}")
    
    eda = ArabicTextEDA()
    
    try:
        print(f"Processing '{INPUT_FILE}'. This may take a while for large files...")
        
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
             open(FINAL_CLEAN_FILE, 'w', encoding='utf-8') as f_clean, \
             open(MODIFIED_SENTENCES_REPORT, 'w', encoding='utf-8') as f_report:
            
            f_report.write("--- MODIFICATION & DELETION REPORT ---\n")
            f_report.write("Logs all lines that were: 1) Modified by cleaning/normalization OR 2) Deleted by a filter.\n")
            f_report.write(f"Configuration:\n")
            f_report.write(f"  - Delete single-word sentences: {'YES' if DELETE_SINGLE_WORD_SENTENCES else 'NO'}\n")
            f_report.write(f"  - Delete non-Arabic sentences: {'YES' if DELETE_NON_ARABIC_SENTENCES else 'NO'}\n\n")

            for i, raw_line in enumerate(f_in, 1):
                original_line = raw_line.strip()
                eda.analyze_line(original_line, is_original=True)
                
                if not original_line:
                    continue

                processed_line, spelling_changes = correct_and_normalize(original_line, ARZ_SAFE_CORRECTIONS)
                
                if spelling_changes:
                    eda.record_spelling_changes(spelling_changes)
                
                should_delete, reason = check_for_deletion(processed_line)
                is_modified = (original_line != processed_line)
                
                if should_delete:
                    eda.record_deletion(reason)
                    f_report.write(f"--- Line {i}: DELETED ---\n")
                    f_report.write(f"Reason: {reason}\n")
                    f_report.write(f"Original: {original_line}\n")
                    if processed_line != original_line:
                        f_report.write(f"Processed: {processed_line}\n")
                    f_report.write("\n")
                else:
                    eda.record_kept_line()
                    f_clean.write(processed_line + "\n")
                    
                    if is_modified:
                        eda.record_modification()
                        f_report.write(f"--- Line {i}: MODIFIED ---\n")
                        if spelling_changes:
                            fixes = ', '.join([f'{k} -> {v}' for k, v in spelling_changes.items()])
                            f_report.write(f"Spelling Fixes: {fixes}\n")
                        f_report.write(f"Before: {original_line}\n")
                        f_report.write(f"After:  {processed_line}\n\n")

                if i % 1000000 == 0:
                    print(f" ... Processed {i:,} lines.")

        print("Generating EDA analysis...")
        eda.generate_report(EDA_REPORT)
        
        print("\n--- Pipeline Execution Finished ---")
        print(f"Total Lines Processed: {eda.stats['total_lines']}")
        print(f"Lines Kept: {eda.stats['lines_kept']}")
        print(f"Lines Deleted: {eda.stats['lines_deleted']}")
        print(f"Lines Modified: {eda.stats['lines_modified']}")
        
        if eda.stats['lines_kept'] + eda.stats['lines_deleted'] > 0:
            retention_rate = (eda.stats['lines_kept'] / (eda.stats['lines_kept'] + eda.stats['lines_deleted'])) * 100
            print(f"Retention Rate: {retention_rate:.1f}%")
        
        print(f"Final clean text saved to '{FINAL_CLEAN_FILE}'")
        print(f"Modification/Deletion report saved to '{MODIFIED_SENTENCES_REPORT}'")
        print(f"EDA analysis saved to '{EDA_REPORT}'")

    except UnicodeDecodeError:
        print(f"ERROR: Unable to read '{INPUT_FILE}'. Please ensure it's UTF-8 encoded.")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()
