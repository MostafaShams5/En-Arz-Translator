"""
Author: Shams
Description:
Sometimes the text I scrape is messy. It has HTML junk like "&nbsp;" instead of space,
or "&quot;" instead of a quote mark.
This script is a cleaner. It fixes those HTML entities and normalizes quotes.
It also separates English and Arabic text if they are mixed in a weird way on one line.
"""

import re
import os
from typing import List, Tuple

class BilingualTextProcessor:
    def __init__(self):
        self.english_sentences = []
        self.arabic_sentences = []
        self.cleaning_samples = []
        
    def clean_text(self, text: str) -> str:
        """
        This function does the heavy lifting. It swaps out the HTML codes
        for real characters.
        """
        original_text = text
        
        # Fix the HTML stuff
        text = text.replace('&gt;', '>')
        text = text.replace('&lt;', '<')
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        text = text.replace('&nbsp;', ' ')
        
        # Clean up whitespace
        text = text.strip()
        
        # Sometimes there are weird extra quotes at the start/end of lines
        if text.startswith("'") and text.count("'") > 2:
            text = text[1:]
        if text.endswith("'") and text.count("'") > 2:
            text = text[:-1]
            
        # Turn double spaces into single spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove those weird '>' symbols that people use for quoting in forums
        text = re.sub(r'^>\s*', '', text)
        
        # Fix spacing around punctuation (e.g. "hello ." -> "hello.")
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*$', r'\1', text)
        
        text = text.strip()
        
        # Keep a record if we actually changed anything, so I can check later
        if original_text.strip() != text and len(original_text.strip()) > 0:
            self.cleaning_samples.append((original_text.strip(), text))
            
        return text
    
    def is_arabic_text(self, text: str) -> bool:
        # Checks if the string contains Arabic letters
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        return bool(arabic_pattern.search(text))
    
    def is_english_text(self, text: str) -> bool:
        # Checks if the string is mostly English
        clean_text = re.sub(r'[^\w]', '', text)
        if not clean_text:
            return False
            
        latin_chars = len(re.findall(r'[a-zA-Z]', clean_text))
        total_chars = len(clean_text)
        
        # If more than 70% letters are Latin, I count it as English
        return (latin_chars / total_chars) > 0.7 if total_chars > 0 else False
    
    def process_line(self, line: str) -> Tuple[str, str]:
        """
        Reads a line from my file. It expects format like: "123. English ||| Arabic".
        It splits them up and cleans both sides.
        """
        # Remove the ID number at the start
        line = re.sub(r'^\d+\.\s*', '', line.strip())
        
        # Check if we have the separator
        if '|||' not in line:
            return '', ''
            
        parts = line.split('|||')
        if len(parts) != 2:
            return '', ''
            
        left_part = parts[0].strip()
        right_part = parts[1].strip()
        
        left_clean = self.clean_text(left_part)
        right_clean = self.clean_text(right_part)
        
        # Figure out which side is which language
        english_text = ''
        arabic_text = ''
        
        if self.is_english_text(left_clean) and self.is_arabic_text(right_clean):
            english_text = left_clean
            arabic_text = right_clean
        elif self.is_arabic_text(left_clean) and self.is_english_text(right_clean):
            english_text = right_clean
            arabic_text = left_clean
        
        return english_text, arabic_text
    
    def process_file(self, input_file: str) -> None:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"Reading {len(lines)} lines from {input_file}...")
            
            processed_count = 0
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                    
                english, arabic = self.process_line(line)
                
                if english and arabic:
                    self.english_sentences.append(english)
                    self.arabic_sentences.append(arabic)
                    processed_count += 1
                    
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} lines...")
            
            print(f"Done. Found {processed_count} valid pairs.")
            
        except FileNotFoundError:
            print(f"Error: Could not find {input_file}")
        except Exception as e:
            print(f"Something went wrong: {e}")
    
    def save_results(self, base_filename: str) -> None:
        # Save English to one file
        english_file = f"{base_filename}_english.txt"
        with open(english_file, 'w', encoding='utf-8') as f:
            for sentence in self.english_sentences:
                f.write(sentence + '\n')
        
        # Save Arabic to another
        arabic_file = f"{base_filename}_arabic.txt"
        with open(arabic_file, 'w', encoding='utf-8') as f:
            for sentence in self.arabic_sentences:
                f.write(sentence + '\n')
        
        # Save a log of what got cleaned so I can audit it
        samples_file = f"{base_filename}_cleaning_samples.txt"
        with open(samples_file, 'w', encoding='utf-8') as f:
            f.write("CLEANING LOG\n")
            f.write("============\n\n")
            
            for i, (before, after) in enumerate(self.cleaning_samples[:50], 1):
                f.write(f"#{i}\n")
                f.write(f"RAW:   {before}\n")
                f.write(f"CLEAN: {after}\n")
                f.write("-" * 20 + "\n\n")
        
        print(f"Saved files:")
        print(f"  - {english_file}")
        print(f"  - {arabic_file}")
        print(f"  - {samples_file}")
    
    def generate_eda(self) -> None:
        print("\n--- Quick Stats ---")
        
        print(f"English Lines: {len(self.english_sentences)}")
        print(f"Arabic Lines: {len(self.arabic_sentences)}")
        print(f"Lines Cleaned: {len(self.cleaning_samples)}")
        
        if self.english_sentences:
            # Show me 3 random examples so I verify it looks okay
            print(f"\nExamples (English):")
            for i, sentence in enumerate(self.english_sentences[:3], 1):
                print(f"  {i}. {sentence}")
        
        if self.arabic_sentences:
            print(f"\nExamples (Arabic):")
            for i, sentence in enumerate(self.arabic_sentences[:3], 1):
                print(f"  {i}. {sentence}")

def main():
    input_file = "eng_arz_corpus_P.txt"
    output_base = "Lprocessed"
    
    processor = BilingualTextProcessor()
    
    processor.process_file(input_file)
    processor.save_results(output_base)
    processor.generate_eda()

if __name__ == "__main__":
    main()
