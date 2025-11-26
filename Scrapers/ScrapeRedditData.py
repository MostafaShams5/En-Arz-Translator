#!/usr/bin/env python3
"""
Author: Shams
Description:
This is the main Reddit scraper. It runs multiple 'workers' (threads) at the same time
to grab posts from different Egyptian subreddits. 
It uses a local database (SQLite) to remember which posts I already have so I don't 
download duplicates. It also handles saving to text files and creating new files 
when the current one gets too big.
"""

import praw
import prawcore
import os
import re
import time
import random
import sqlite3
import threading

# These correspond to the sections in my praw.ini file
ACCOUNT_NAMES = ["DEFAULT", "worker_2", "worker_3", "worker_4"]     

TARGET_SUBREDDITS = ["MahmoudIsmail","Egy_Memes","AskinEgypt","RedditMasr","Egypt", "AskEgypt", "Cairo", "ExEgypt", "AllHayganeen", "Askmasr","Tanta", "EgySexEducation", "ddirtybagteenagers", "Egypt_Developers","EgyptExTomato", "EgyptFinancePro", "PersonalFinanceEgypt", "AlexandriaEgy", "masr", "Cinema_Egypt", "EGYescapism", "EgyptCS", "EgyptianExpats","EgyReaders", "LGBTEgypt", "EgyGains"]

OUTPUT_DIRECTORY = "scraped_data"
DB_FILE = "processed_posts.db"

# How many posts to grab per request
POSTS_PER_BATCH = 100
# How many batches to run before taking a break
BATCHES_PER_CYCLE = 10
# Limit comments so we don't get stuck on one viral post
COMMENTS_PER_POST = 10
# When a file hits this many lines, start a new file
LINES_PER_FILE = 10000

# Random sleep time to keep Reddit API happy
MIN_DELAY = 0.8
MAX_DELAY = 1.2

# These locks are crucial. They stop two threads from trying to write to the 
# database or text file at the exact same time, which would cause errors.
db_lock = threading.Lock()
output_lock = threading.Lock()


class OutputManager:
    """
    This class handles the text files. It checks what files already exist
    so it can resume writing to 'corpus_0005.txt' instead of overwriting 'corpus_0001.txt'.
    """
    def __init__(self, directory, lines_per_file):
        self.directory = directory
        self.lines_per_file = lines_per_file
        self.current_file_index = 1
        self.current_line_count = 0
        
        os.makedirs(self.directory, exist_ok=True)
        
        # Figure out where we left off last time
        self._initialize_state()

    def _initialize_state(self):
        print("Checking output folder to resume...")
        try:
            # Looking for files named like egyptian_arabic_corpus_0001.txt
            file_pattern = re.compile(r"egyptian_arabic_corpus_(\d{4})\.txt")
            
            files = [f for f in os.listdir(self.directory) if file_pattern.match(f)]

            if not files:
                print("No previous files found. Starting fresh.")
                return

            # Find the highest numbered file
            last_index = 0
            for filename in files:
                match = file_pattern.match(filename)
                if match:
                    file_num = int(match.group(1))
                    if file_num > last_index:
                        last_index = file_num
            
            self.current_file_index = last_index
            last_filename = os.path.join(self.directory, f"egyptian_arabic_corpus_{last_index:04d}.txt")
            
            # Count how many lines are in that last file so we know when to switch
            with open(last_filename, 'r', encoding='utf-8') as f:
                self.current_line_count = sum(1 for _ in f)
            
            print(f"Resuming at file #{self.current_file_index} which has {self.current_line_count} lines.")

        except Exception as e:
            print(f"Error reading old files: {e}. Starting over to be safe.")
            self.current_file_index = 1
            self.current_line_count = 0

    def _get_output_filename(self):
        return os.path.join(self.directory, f"egyptian_arabic_corpus_{self.current_file_index:04d}.txt")

    def write(self, text: str):
        if not text:
            return 0
        
        # Lock this section so only one thread writes at a time
        with output_lock:
            # Check if file is full
            if self.current_line_count >= self.lines_per_file:
                self.current_file_index += 1
                self.current_line_count = 0
                print(f"File full. Switching to file #{self.current_file_index}")
            
            with open(self._get_output_filename(), "a", encoding="utf-8") as f:
                f.write(text + "\n")
            
            self.current_line_count += 1
            return 1

def init_db():
    # check_same_thread=False lets multiple threads use this connection
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        # Table to track individual post IDs
        conn.execute("CREATE TABLE IF NOT EXISTS processed_posts (post_id TEXT PRIMARY KEY, processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        # Table to track progress per subreddit (bookmarks)
        conn.execute("CREATE TABLE IF NOT EXISTS scraping_progress (subreddit_name TEXT PRIMARY KEY, last_processed_id TEXT)")
    print("Database initialized.")

def get_last_post(subreddit_name: str) -> str | None:
    # Reads the bookmark for this subreddit
    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT last_processed_id FROM scraping_progress WHERE subreddit_name = ?", (subreddit_name,))
        result = cursor.fetchone()
        return result[0] if result else None

def update_last_post(subreddit_name: str, post_id: str):
    # Updates the bookmark. Uses lock to be safe.
    with db_lock:
        with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
            conn.execute("INSERT INTO scraping_progress (subreddit_name, last_processed_id) VALUES (?, ?) ON CONFLICT(subreddit_name) DO UPDATE SET last_processed_id = excluded.last_processed_id", (subreddit_name, post_id))

def check_and_claim_post(post_id: str) -> bool:
    # Checks if we already did this post. If not, mark it as done immediately.
    with db_lock:
        with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM processed_posts WHERE post_id = ?', (post_id,))
            if cursor.fetchone():
                return True # We already have it, skip
            else:
                cursor.execute('INSERT INTO processed_posts (post_id) VALUES (?)', (post_id,))
                conn.commit()
                return False # New post, go ahead

def is_egyptian_arabic(text: str) -> bool:
    if not text or len(text.strip()) < 5: return False
    # Count Arabic letters vs total letters
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    total_chars = len(re.sub(r"\s+", "", text))
    if total_chars == 0: return False
    # It needs to be at least 60% Arabic characters to pass
    return (arabic_chars / total_chars) > 0.6

def clean_text(text: str) -> str:
    if not text or text.lower() in ['[deleted]', '[removed]']: return ""
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text.strip())
    # Remove links
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    # Remove mentions and subreddit links
    text = re.sub(r"@[\w_]+|#[\w_]+|/u/[\w_]+|/r/[\w_]+", "", text)
    
    if not is_egyptian_arabic(text): return ""
    
    # Remove English words/numbers
    text = re.sub(r"\b[a-zA-Z0-9_]+\b", "", text)
    # Keep only Arabic letters and specific punctuation
    arabic_pattern = re.compile(r"[\u0600-\u06FF\s،؛؟]+")
    filtered_text = ''.join(arabic_pattern.findall(text)).strip()
    
    return filtered_text if len(filtered_text) > 5 else ""

def process_submission(submission, output_manager: OutputManager):
    lines_written = 0
    # Process title and body text
    for text_part in [submission.title, submission.selftext]:
        lines_written += output_manager.write(clean_text(text_part))
    try:
        # Flatten comments (no nested replies) to save API calls
        submission.comments.replace_more(limit=0)
        for i, comment in enumerate(submission.comments.list()):
            if i >= COMMENTS_PER_POST: break
            lines_written += output_manager.write(clean_text(comment.body))
    except Exception as e:
        print(f"Error processing comments for post {submission.id}: {e}")
    return lines_written

def rate_limit_delay():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

def worker_task(account_name: str, output_manager: OutputManager, worker_id: int):
    try:
        reddit = praw.Reddit(account_name)
        print(f"Worker #{worker_id} connected as {reddit.user.me()}.")
    except Exception as e:
        print(f"Worker #{worker_id} failed to connect: {e}")
        return

    while True:
        for subreddit_name in TARGET_SUBREDDITS:
            print(f"[Worker #{worker_id}] Checking r/{subreddit_name}...")
            
            # Get our bookmark from the DB
            after_post_id = get_last_post(subreddit_name)
            if after_post_id:
                print(f"[Worker #{worker_id}] Resuming from: {after_post_id}")
            else:
                print(f"[Worker #{worker_id}] Starting fresh.")

            for i in range(BATCHES_PER_CYCLE):
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    # 'after' param asks Reddit for posts older than our bookmark
                    params = {'after': f"t3_{after_post_id}"} if after_post_id else {}
                    batch = list(subreddit.new(limit=POSTS_PER_BATCH, params=params))
                    
                    if not batch:
                        print(f"[Worker #{worker_id}] No more posts in r/{subreddit_name}.")
                        break

                    for submission in batch:
                        # Try to claim the post. If return True, another worker got it.
                        if check_and_claim_post(submission.id):
                            continue

                        rate_limit_delay()
                        lines_from_post = process_submission(submission, output_manager)
                        
                        if lines_from_post > 0:
                            print(f"  [Worker #{worker_id}] Saved {lines_from_post} lines from post {submission.id}.")
                    
                    # Update bookmark to the last post in this batch
                    last_post_in_batch_id = batch[-1].id
                    after_post_id = last_post_in_batch_id
                    update_last_post(subreddit_name, after_post_id)
                    print(f"  [Worker #{worker_id}] Batch done. Bookmark updated: {after_post_id}")

                except Exception as e:
                    print(f"  [Worker #{worker_id}] Error in r/{subreddit_name}: {e}")
                    time.sleep(10)
                    break
        
        print(f"Worker #{worker_id} finished cycle. Sleeping 5 minutes.")
        time.sleep(300)

if __name__ == '__main__':
    init_db()
    output_manager = OutputManager(OUTPUT_DIRECTORY, LINES_PER_FILE)
    threads = []
    
    print(f"Starting scraper with {len(ACCOUNT_NAMES)} workers...")

    for i, account_name in enumerate(ACCOUNT_NAMES):
        # Daemon threads die when the main script stops
        thread = threading.Thread(target=worker_task, args=(account_name, output_manager, i + 1), daemon=True)
        threads.append(thread)
        thread.start()
        time.sleep(2) # Stagger starts so they don't hit the DB at once

    try:
        # Keep main thread alive so daemons can run
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\nStopping scraper...")
    finally:
        print("\nScraper stopped.")
