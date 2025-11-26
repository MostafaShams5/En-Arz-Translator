"""
Author: Shams
Description:
This script fixes broken numbering in a text file.
Sometimes my files get messed up numbers (like 1, 2, 5, 100).
This script reads the file and rewrites it with clean, sequential numbers (1, 2, 3...) after a certain missnumbered line.
"""

def renumber_lines_in_file(input_filename, output_filename):
    try:
        with open(input_filename, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find '{input_filename}'.")
        return

    # These variables track where we are in the file
    start_renumbering = False
    current_number = 0
    new_lines = []

    # I can set a specific line text to trigger the renumbering if I want
    trigger_line = "2776. We've hit a dead end."

    for line in lines:
        stripped_line = line.strip()

        if not start_renumbering:
            # If we find the trigger line, we start the new sequence
            if stripped_line == trigger_line:
                start_renumbering = True
                current_number = 2876  # Picking up from this number
                
                # Split the old number from the text
                _, sentence = stripped_line.split('. ', 1)
                
                new_line = f"{current_number}. {sentence}\n"
                new_lines.append(new_line)
                current_number += 1
            else:
                # Just copy the line as-is if we haven't hit the trigger yet
                new_lines.append(line)
        else:
            # We are in renumbering mode
            try:
                # Separate text from old number
                _, sentence = stripped_line.split('. ', 1)
                
                # Write with new number
                new_line = f"{current_number}. {sentence}\n"
                new_lines.append(new_line)
                current_number += 1
            except ValueError:
                # If the line has no number (like a blank line), just keep it
                new_lines.append(line)

    with open(output_filename, 'w', encoding='utf-8') as f_out:
        f_out.writelines(new_lines)

    print(f"Done. Saved fixed file to '{output_filename}'")


if __name__ == "__main__":
    input_file = "eng-01.txt"
    output_file = "output-eng-01.txt"

    renumber_lines_in_file(input_file, output_file)
