"""
Author: Shams
Description:
This script grabs data from the 'faisalq/ETC' dataset on Hugging Face.
Since the dataset is huge, I am using streaming mode to read it piece by piece
instead of downloading the whole thing to my hard drive first.
It cleans up newlines and saves the text to a single file.
"""

from datasets import load_dataset
import os

OUT_FILE = "arz_raw_tweets.txt"
print(f"Starting stream. Saving output to: {OUT_FILE}")

# I am using streaming=True here.
# This is important because otherwise it tries to download everything to RAM/Disk.
try:
    etc_stream = load_dataset(
        "faisalq/ETC",
        split="train",
        streaming=True, 
    )
except Exception as e:
    print(f"Could not load dataset: {e}")
    exit()

empty_tweets_skipped = 0
total_tweets_processed = 0

with open(OUT_FILE, "w", encoding="utf-8") as fout:
    # Loop through the dataset one item at a time (lazy loading)
    for example in etc_stream:
        total_tweets_processed += 1

        # I use .get() just in case the 'tweet' key is missing, so the script doesn't crash
        text = example.get("tweet")

        # Only write it if there is actual text
        if text and text.strip():
            # I need to remove newlines inside the tweet itself so it fits on one line in my text file
            cleaned_text = text.replace("\n", " ").strip()
            fout.write(cleaned_text + "\n")
        else:
            empty_tweets_skipped += 1

print("Done.")
print(f"Processed {total_tweets_processed} items.")
print(f"Saved valid tweets to: {os.path.abspath(OUT_FILE)}")
print(f"Skipped {empty_tweets_skipped} empty items.")a
