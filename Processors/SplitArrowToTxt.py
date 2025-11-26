"""
Author: Shams
Description:
The dataset I got is in a format called Arrow (from Hugging Face). 
It's too big to open in a normal text editor.
So I wrote this to chop it up into 50 smaller text files that are easier to handle.
"""

import os
import math
from datasets import load_from_disk

ARROW_DIR = "etc-mini-(2gb)"

# How many pieces I want
NUM_FILES = 50

# Load the data
ds = load_from_disk(ARROW_DIR)
total = len(ds)
print(f"Found {total} examples.")

# Check the column name
field = "sentence" if "sentence" in ds.features else "text"
print(f"Using column: '{field}'")

# Calculate lines per file
lines_per_file = math.ceil(total / NUM_FILES)
print(f"Lines per file: {lines_per_file}")

OUT_DIR = os.path.join(ARROW_DIR, "ETC-ARZ")
os.makedirs(OUT_DIR, exist_ok=True)

file_idx = 0
line_idx = 0
fout = None

for i, example in enumerate(ds):
    # Start a new file if needed
    if line_idx == 0:
        if fout:
            fout.close()
        file_idx += 1
        filename = os.path.join(OUT_DIR, f"ETC-ARZ-{file_idx:02d}.txt")
        fout = open(filename, "w", encoding="utf-8")
    
    # Flatten the text
    text = example[field].replace("\n", " ").strip()
    fout.write(text + "\n")
    line_idx += 1

    # Switch file if full
    if line_idx >= lines_per_file:
        line_idx = 0

if fout:
    fout.close()

print(f"Done. Made {file_idx} files in '{OUT_DIR}/'")
