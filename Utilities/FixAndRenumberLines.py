"""
Author: Shams
Description:
This is a precise tool to fix a numbering error starting at a specific line.
If I know line 9811 is wrong, I tell this script to find line 9811
and start renumbering from 9810 onwards.
"""

import re

input_filename = "en10.txt"
output_filename = "en1010.txt"

# The line number in the file that is currently wrong
target_line_number = 9811
# The number it SHOULD be (and count up from)
new_start_number = 9810
# --------------

renumbering_active = False
current_number = 0

print(f"Reading '{input_filename}'...")
print(f"Looking for line {target_line_number} to restart count at {new_start_number}.")

# Regex to grab the number and the text separately
# Group 1 is the number, Group 2 is the text
line_pattern = re.compile(r'^(\d+)\.\s*(.*)')

try:
    with open(input_filename, 'r', encoding="utf-8") as infile, \
         open(output_filename, 'w', encoding="utf-8") as outfile:

        for line in infile:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # If we already found the target, just keep counting up
            if renumbering_active:
                current_number += 1
                # Remove old number
                text_content = re.sub(r'^\d+\.\s*', '', stripped_line)
                # Write new number
                new_line = f"{current_number}. {text_content}\n"
                outfile.write(new_line)
                continue

            # If we haven't found the target yet, check this line
            match = line_pattern.match(stripped_line)
            
            if match:
                line_num_found = int(match.group(1))
                text_content = match.group(2)

                if line_num_found == target_line_number:
                    print(f"\nFound target line {target_line_number}. Starting fix...")
                    renumbering_active = True
                    current_number = new_start_number
                    
                    new_line = f"{current_number}. {text_content}\n"
                    outfile.write(new_line)
                else:
                    # Not the target, just copy it
                    outfile.write(f"{stripped_line}\n")
            else:
                # Line has no number, just copy it
                outfile.write(f"{stripped_line}\n")

    if not renumbering_active:
        print(f"\nWarning: I never found line {target_line_number}.")
    else:
        print("\nProcessing complete.")

    print(f"Saved to: '{output_filename}'")

except FileNotFoundError:
    print(f"\nError: File '{input_filename}' not found.")
