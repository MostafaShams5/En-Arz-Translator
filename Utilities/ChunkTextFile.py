"""
Author: Shams
Description:
This script takes a raw text file (like a book or article) and chops it up into
manageable chunks. I set it to 350 words per chunk.
It then writes these chunks into a new file, numbering them like a list.
"""

def format_text_by_word_count(input_filename, output_filename, words_per_line):
    if words_per_line <= 0:
        print("Error: words_per_line must be a positive number.")
        return

    print(f"Reading from '{input_filename}'...")

    try:
        with open(input_filename, 'r', encoding='utf-8') as file:
            raw_text = file.read()
    except FileNotFoundError:
        print(f"Error: Could not find '{input_filename}'. Make sure it exists.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Split by whitespace to get a list of individual words
    words = raw_text.split()

    if not words:
        print("The input file is empty. Nothing to do.")
        return

    print(f"Grouping text into chunks of {words_per_line} words...")

    try:
        with open(output_filename, 'w', encoding='utf-8') as file:
            line_number = 1

            # Loop through the words list, jumping by 'words_per_line' steps
            for i in range(0, len(words), words_per_line):
                # Slice the list to get the current chunk
                word_chunk = words[i : i + words_per_line]

                # Join them back into a sentence string
                line = " ".join(word_chunk)

                # Write it to the file with a number
                file.write(f"{line_number}. {line}\n\n")

                line_number += 1
    except Exception as e:
        print(f"Error writing file: {e}")
        return

    print("Done.")
    print(f"Check '{output_filename}' for the result.")


if __name__ == "__main__":
    # Files to use
    input_file = "english_raw_text.txt"
    output_file = "en.txt"

    # How many words I want per chunk
    WORDS_PER_LINE = 350

    format_text_by_word_count(input_file, output_file, WORDS_PER_LINE)
