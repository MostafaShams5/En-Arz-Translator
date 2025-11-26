"""
Author: Shams
Description:
I use this script to clean up messy text files. 
It removes emojis, weird symbols, and lines that are just gibberish.
It also splits the clean text into smaller files so I don't have one massive file.
"""

import os
import re
import random
from collections import defaultdict

class TextFileProcessor:
    def __init__(self, input_files, output_folder="output_texts", max_file_size_kb=700):
        self.input_files = input_files
        self.output_folder = output_folder
        self.max_file_size_bytes = max_file_size_kb * 1024
        self.min_words = 4
        self.max_words = 40
        self.sentences = []
        self.output_files = []
        self.unique_sentences = set()
        self.duplicate_count = 0
        self.filtered_start_count = 0
        
    def normalize_sentence_for_comparison(self, sentence):
        """
        I want to find duplicates even if the punctuation is slightly different.
        So I lowercase everything and strip punctuation for the check.
        """
        normalized = sentence.lower().strip()
        normalized = re.sub(r'[.!?;,]+$', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
        
    def clean_sentence(self, sentence):
        """
        Returns the sentence if it's clean, or None if it's garbage.
        """
        sentence = re.sub(r'\s+', ' ', sentence.strip())
        
        if len(sentence) < 3:
            return None
        
        # If it doesn't start with a letter, I don't want it
        if not self.starts_with_english_letter(sentence):
            self.filtered_start_count += 1
            return None
        
        if self.has_noise(sentence):
            return None
            
        return sentence
    
    def starts_with_english_letter(self, sentence):
        if not sentence:
            return False
        
        first_char = sentence[0]
        if re.match(r'[a-zA-Z]', first_char):
            return True
        
        return False
    
    def is_duplicate(self, sentence):
        normalized = self.normalize_sentence_for_comparison(sentence)
        
        if normalized in self.unique_sentences:
            return True
        else:
            self.unique_sentences.add(normalized)
            return False
    
    def has_noise(self, sentence):
        """
        This checks for emojis, too many symbols, or things that don't look like words.
        """
        # Big regex list of emojis to ban
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+"
        )
        
        if emoji_pattern.search(sentence):
            return True
        
        # If more than 15% of the characters are weird symbols, skip it
        special_chars = re.findall(r'[^\w\s.,!?;:\'"()-]', sentence)
        if len(special_chars) > len(sentence) * 0.15:
            return True
        
        # If it's mostly numbers, skip it
        numbers = re.findall(r'\d+', sentence)
        total_number_chars = sum(len(num) for num in numbers)
        if total_number_chars > len(sentence) * 0.25:
            return True
        
        # If it has too many non-English characters
        non_english = re.findall(r'[^\x00-\x7F]', sentence)
        if len(non_english) > len(sentence) * 0.05:
            return True
        
        # Spam check: if a character repeats 4+ times (liiiiike this)
        if re.search(r'(.)\1{3,}', sentence):
            return True
        
        # Too much punctuation
        punctuation = re.findall(r'[.,!?;:\'"]', sentence)
        if len(punctuation) > len(sentence) * 0.25:
            return True
        
        if self.is_bad_english(sentence):
            return True
        
        return False
    
    def is_bad_english(self, sentence):
        """
        Tries to catch random strings like 'asdfgh' or things with no vowels.
        """
        sentence_lower = sentence.lower()
        words = sentence_lower.split()
        bad_word_count = 0
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) < 2:
                continue
                
            if (
                re.search(r'[bcdfghjklmnpqrstvwxyz]{5,}', clean_word) or # No vowels
                re.search(r'[aeiou]{4,}', clean_word) or # Too many vowels
                re.search(r'[a-z][A-Z][a-z]', word) or # Weird capitalization
                re.search(r'(.)\1{2,}', clean_word) # Repeated letters
            ):
                bad_word_count += 1
        
        if len(words) > 0 and bad_word_count / len(words) > 0.3:
            return True
        
        letter_count = len(re.findall(r'[a-zA-Z]', sentence))
        if letter_count < len(sentence) * 0.5:
            return True
        
        valid_words = re.findall(r'\b[a-zA-Z]{2,}\b', sentence)
        if len(valid_words) < 2:
            return True
        
        return False
    
    def count_words(self, sentence):
        words = sentence.split()
        return len(words)
    
    def load_and_process_files(self):
        print("Reading files...")
        
        total_sentences_processed = 0
        
        for file_path in self.input_files:
            if not os.path.exists(file_path):
                print(f"Can't find {file_path}. Skipping.")
                continue
                
            print(f"Working on: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                
                # Split text by punctuation marks
                sentences = re.split(r'[.!?]+', content)
                
                for sentence in sentences:
                    total_sentences_processed += 1
                    
                    cleaned = self.clean_sentence(sentence)
                    if cleaned:
                        word_count = self.count_words(cleaned)
                        if self.min_words <= word_count <= self.max_words:
                            if not self.is_duplicate(cleaned):
                                self.sentences.append(cleaned)
                            else:
                                self.duplicate_count += 1
                            
            except Exception as e:
                print(f"Error with {file_path}: {str(e)}")
        
        print(f"Read {total_sentences_processed:,} sentences.")
        print(f"Kept {len(self.sentences):,} good ones.")
        print(f"Removed {self.duplicate_count:,} duplicates.")
        print(f"Filtered {self.filtered_start_count:,} for bad starts.")
    
    def shuffle_sentences(self):
        print("Mixing up the sentences...")
        random.shuffle(self.sentences)
    
    def split_into_files(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        print(f"Saving to '{self.output_folder}'...")
        
        current_file_num = 1
        current_content = []
        current_size = 0
        
        for sentence in self.sentences:
            sentence_number = len(current_content) + 1
            formatted_sentence = f"{sentence_number}. {sentence}\n"
            sentence_size = len(formatted_sentence.encode('utf-8'))
            
            # If the file is getting too big, start a new one
            if current_size + sentence_size > self.max_file_size_bytes and current_content:
                self._write_output_file(current_file_num, current_content)
                current_file_num += 1
                current_content = []
                current_size = 0
                sentence_number = 1
                formatted_sentence = f"{sentence_number}. {sentence}\n"
                sentence_size = len(formatted_sentence.encode('utf-8'))
            
            current_content.append(formatted_sentence)
            current_size += sentence_size
        
        if current_content:
            self._write_output_file(current_file_num, current_content)
    
    def _write_output_file(self, file_num, content):
        filename = f"text_{file_num:03d}.txt"
        filepath = os.path.join(self.output_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as file:
            file.writelines(content)
        
        file_size = os.path.getsize(filepath)
        self.output_files.append({
            'filename': filename,
            'filepath': filepath,
            'sentences': len(content),
            'size_bytes': file_size,
            'size_kb': file_size / 1024
        })
        
        print(f"Saved: {filename} ({file_size/1024:.1f} KB)")
    
    def generate_eda(self):
        if not self.output_files:
            print("Nothing to report.")
            return
        
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        
        total_sentences = sum(file_info['sentences'] for file_info in self.output_files)
        total_size_kb = sum(file_info['size_kb'] for file_info in self.output_files)
        
        print(f"Total files created: {len(self.output_files)}")
        print(f"Total sentences: {total_sentences:,}")
        print(f"Duplicates killed: {self.duplicate_count:,}")
        print(f"Bad starts killed: {self.filtered_start_count:,}")
        print(f"Total size: {total_size_kb:.1f} KB")
        
        print(f"\nDetails:")
        for file_info in self.output_files:
            print(f"  {file_info['filename']}: {file_info['sentences']} lines")
    
    def process(self):
        print("Starting...\n")
        
        self.load_and_process_files()
        if not self.sentences:
            print("No sentences found. Stopping.")
            return
        
        self.shuffle_sentences()
        self.split_into_files()
        self.generate_eda()
        
        print(f"\nDone. Check '{self.output_folder}'.")


def main():
    # Put your file names here
    input_files = [
        "eng-3-rap.txt",
        "eng-4-rap.txt",    
        "eng-14-opens.txt",
        'eng-15-opens.txt',
        'eng-60-mick.txt',
        'eng-125-Salt-NLP.txt',
        'huggingfacefw_fineweb_edu_041.txt',
        'squad_006.txt'
    ]
    
    output_folder = "processed_texts"
    max_file_size_kb = 700
    
    processor = TextFileProcessor(
        input_files=input_files,
        output_folder=output_folder,
        max_file_size_kb=max_file_size_kb
    )
    
    processor.process()


if __name__ == "__main__":
    main()
