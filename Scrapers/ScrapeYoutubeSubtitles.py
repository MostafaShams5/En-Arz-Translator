"""
Author: Shams
Description:
This script calls the 'yt-dlp' command line tool to download subtitles
from the YouTube playlists I listed. It then cleans the VTT files (removing
timestamps and formatting) and saves the raw Arabic text.
performs some EDA and more.
"""

import os
import re
import subprocess
import tempfile
import json
import time
from pathlib import Path
from collections import Counter
import hashlib

# List of playlists to process
PLAYLIST_URLS = [
    "https://www.youtube.com/playlist?list=PLq9SlAJOwd6XVsxbCyfJ3IFyAxN1YkOcC",
    "https://www.youtube.com/playlist?list=PL7TpI0K9I87JnAH1CFX4MTE3Q1oFYvzdY",
    "https://www.youtube.com/playlist?list=PLUegowww5JEdVHagbJU2GCZQVUs7L0eOg",
    "https://www.youtube.com/playlist?list=PL7V-g0JvWGxht99-wquJmyLgT9SBojd9f",
    "https://www.youtube.com/playlist?list=PL7V-g0JvWGxgkmSy6yNymJtQZY5Gv1FPG",
    "https://www.youtube.com/playlist?list=PL7V-g0JvWGxgvTeT1GZHvBOXTFjm-W0CQ",
    "https://www.youtube.com/playlist?list=PL7V-g0JvWGxh4hSTJsyB5VWeaU0vkUUnt",
    "https://www.youtube.com/playlist?list=PL7V-g0JvWGxjkbFge3hdZUtlT6fp7C6Sb",
    'https://www.youtube.com/playlist?list=PL4_bo90i-4GItp18iodTK2cv-Mzn0k1es',
    'https://www.youtube.com/playlist?list=PL4_bo90i-4GITOqdamSufNVj-UvaJz9eK',
    'https://www.youtube.com/playlist?list=PL4_bo90i-4GJJEU7vmZ6U5eICC2Gusg3g',
    'https://www.youtube.com/playlist?list=PLZxRWV0g4J7-kb5kQofee8kJk0fiT0-Sv',
    'https://www.youtube.com/playlist?list=PLAv3K1xOSL17c7d6ZSFCyXoE1cPpCq5Be',
    'https://www.youtube.com/playlist?list=PLpVBfBQmVoLW2CiDA6NaX7OqRtCtBl_UU',
    'https://www.youtube.com/playlist?list=PLpVBfBQmVoLU13HYgPB6GUbyhMTGUvcnm',
    'https://www.youtube.com/playlist?list=PLpVBfBQmVoLWyIa9xZK8tVcR0u29T5sDe',
    'https://www.youtube.com/playlist?list=PLpVBfBQmVoLXxcLvCre587C7xOtDaoCLl',
    'https://www.youtube.com/playlist?list=PLaICZFp3i9jZLRojkJoa4y548na-V2Aw2',
    'https://www.youtube.com/playlist?list=PLaICZFp3i9jZvEz4GU3rEFDZmAIiY4HL5',
    'https://www.youtube.com/playlist?list=PLaICZFp3i9jabyFJy1mmXfsvC-jgQodlr',
    'https://www.youtube.com/playlist?list=PLettntLzVfXtymrfDlYPC6CgCq19DVdig',
    'https://www.youtube.com/playlist?list=PLupxaP3T6J-gTsPWJJNKPDg0iFLUDWA1j',
    'https://www.youtube.com/playlist?list=PLRCzrSHS5u_FFoGXKpBoG7kJ8yZNgNwLK',
    'https://www.youtube.com/playlist?list=PLHsw1qQLRyqRtPrDGpD3XLXgDqfzJoz2v',
    'https://www.youtube.com/playlist?list=PL7TpI0K9I87K5qQte9awzBO2QkeMl-UAN',
    'https://www.youtube.com/playlist?list=PL3JVNMf4T92chExzUS8Opr2U9lKlXgZzN',
    'https://www.youtube.com/playlist?list=PLsx8vChaZTAtAocagiocoTac4IjtAQjF4',
    'https://www.youtube.com/playlist?list=PL5isa5XjlZ5pH8vT0TJO9qZrRiZcrWuDp',
    'https://www.youtube.com/playlist?list=PLEBpG7xpXmxVxkQ50fIgGI7PsIuikLTEj',
    'https://www.youtube.com/playlist?list=PLEBpG7xpXmxWcIJQB8Pa3y8oCl6-7YLQv',
    'https://www.youtube.com/playlist?list=PLEBpG7xpXmxXRsAE_0fyoZx5iE41q4yMZ',
    'https://www.youtube.com/playlist?list=PL7V-g0JvWGxhA0JJcKujYoqIpvW_05xkc',
    'https://www.youtube.com/playlist?list=PLEcyGBrmEuRLG3VwJ2Xtt648QQeNLnTpZ',
    'https://www.youtube.com/playlist?list=PLEcyGBrmEuRJnPYyvotwzcpRhX6bhQkA2',
    'https://www.youtube.com/playlist?list=PLEcyGBrmEuRJgyuBHPCk4AX1OV7wa4cJC',
    'https://www.youtube.com/playlist?list=PLsxPqi5Cb6Eiu0LLEjOZnvead7unIkogL',
    'https://www.youtube.com/playlist?list=PLsxPqi5Cb6Ej_vz8o9l4s0lRaTR9rAMln'
]
OUTPUT_FILENAME = "ARZ-youtube.txt"
PROCESSED_VIDEOS_FILE = "processed_videos.json"
LANGUAGE_CODE = 'ar'

class CorpusStats:
    """ Keeps track of statistics while we scrape. """
    def __init__(self):
        self.total_videos_processed = 0
        self.total_lines_written = 0
        self.total_duration_seconds = 0
        self.unique_words = set()
        self.all_lines = set()
        self.duplicate_lines = 0
        
    def add_video_stats(self, lines_count, duration_seconds):
        self.total_videos_processed += 1
        self.total_lines_written += lines_count
        self.total_duration_seconds += duration_seconds
        
    def add_line(self, line):
        if line in self.all_lines:
            self.duplicate_lines += 1
        else:
            self.all_lines.add(line)
            words = line.split()
            self.unique_words.update(words)
    
    def get_total_hours(self):
        return self.total_duration_seconds / 3600
    
    def get_total_minutes(self):
        return self.total_duration_seconds / 60
    
    def print_summary(self):
        print(f"\n=== CORPUS STATISTICS ===")
        print(f"Videos: {self.total_videos_processed:,}")
        print(f"Lines: {self.total_lines_written:,}")
        print(f"Unique Lines: {len(self.all_lines):,}")
        print(f"Duplicates: {self.duplicate_lines:,}")
        print(f"Total Duration: {self.get_total_hours():.2f} hours")

stats = CorpusStats()

def load_processed_videos():
    # Reads the JSON file so we don't re-download videos we already have
    if os.path.exists(PROCESSED_VIDEOS_FILE):
        try:
            with open(PROCESSED_VIDEOS_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except (json.JSONDecodeError, TypeError):
            return set()
    return set()

def save_processed_video(video_id):
    # Marks a video as done in the JSON file
    processed_videos = load_processed_videos()
    processed_videos.add(video_id)
    with open(PROCESSED_VIDEOS_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(processed_videos), f, ensure_ascii=False, indent=2)

def estimate_duration_from_subtitles(vtt_content):
    # If metadata is missing, I look at the last timestamp in the subtitle file
    try:
        lines = vtt_content.splitlines()
        last_timestamp = 0
        
        for line in lines:
            if '-->' in line:
                timestamp_match = re.search(r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*$', line)
                if timestamp_match:
                    hours = int(timestamp_match.group(1))
                    minutes = int(timestamp_match.group(2))
                    seconds = int(timestamp_match.group(3))
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                    last_timestamp = max(last_timestamp, total_seconds)
        
        return last_timestamp if last_timestamp > 0 else 0
    except:
        return 0

def get_video_duration(video_url):
    # Uses yt-dlp to fetch JSON metadata for exact duration
    try:
        command = ['yt-dlp', '--dump-json', '--no-download', video_url]
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        
        if result.returncode == 0 and result.stdout.strip():
            try:
                video_info = json.loads(result.stdout)
                duration = None
                if 'duration' in video_info and video_info['duration']:
                    duration = float(video_info['duration'])
                elif 'duration_string' in video_info and video_info['duration_string']:
                    duration_str = video_info['duration_string']
                    if ':' in duration_str:
                        parts = duration_str.split(':')
                        if len(parts) == 2:
                            minutes, seconds = map(int, parts)
                            duration = minutes * 60 + seconds
                        elif len(parts) == 3:
                            hours, minutes, seconds = map(int, parts)
                            duration = hours * 3600 + minutes * 60 + seconds
                
                if duration and duration > 0:
                    return int(duration)
                    
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                print(f"    > Error parsing metadata: {str(e)}")
                pass
    
    except subprocess.CalledProcessError as e:
        print(f"    > Error fetching metadata: {e.stderr.strip() if e.stderr else 'Unknown error'}")
    
    # Fallback logic if JSON fails
    try:
        command_fallback = ['yt-dlp', '--get-duration', video_url]
        result = subprocess.run(command_fallback, capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0 and result.stdout.strip():
            duration_str = result.stdout.strip()
            if ':' in duration_str:
                parts = duration_str.split(':')
                if len(parts) == 2:
                    minutes, seconds = map(int, parts)
                    return minutes * 60 + seconds
                elif len(parts) == 3:
                    hours, minutes, seconds = map(int, parts)
                    return hours * 3600 + minutes * 60 + seconds
    except:
        pass
    
    print(f"    > Could not find duration. Will estimate from text.")
    return 0

def clean_and_parse_vtt(text):
    """
    VTT files have lots of junk like timestamps and HTML tags.
    This strips all that out so we just get the spoken words.
    """
    last_yielded_line = ""
    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        # Skip timestamps and headers
        if not line or '-->' in line or line.upper().startswith(('WEBVTT', 'KIND:', 'LANGUAGE:')):
            continue

        # Remove HTML tags like <c>
        clean_line = re.sub(r'<[^>]+>', '', line)
        clean_line = re.sub(r'\[.*?\]', '', clean_line)
        clean_line = re.sub(r'\(.*?\)', '', clean_line)
        clean_line = clean_line.replace('>>', '').strip()
        clean_line = clean_line.replace('\u200e', '').replace('\u200f', '').strip()

        # Avoid duplicates (subtitles often repeat lines)
        if clean_line and clean_line != last_yielded_line:
            yield clean_line
            last_yielded_line = clean_line

def scrape_with_yt_dlp():
    print("--- Starting YouTube Transcript Scraper ---")
    
    # Check if yt-dlp is installed on the system
    try:
        subprocess.run(['yt-dlp', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nError: 'yt-dlp' not found.")
        print("Please install it: pip install yt-dlp")
        return

    processed_videos = load_processed_videos()
    print(f"Already processed {len(processed_videos)} videos.")

    print("\nStep 1: Getting video list from playlists...")
    all_video_urls = []
    for i, playlist_url in enumerate(PLAYLIST_URLS):
        print(f"Checking playlist {i+1}/{len(PLAYLIST_URLS)}...")
        try:
            # This gets URLs without downloading the video
            command = ['yt-dlp', '--flat-playlist', '--get-url', playlist_url]
            result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
            playlist_videos = [url for url in result.stdout.splitlines() if url]
            print(f"  > Found {len(playlist_videos)} videos.")
            all_video_urls.extend(playlist_videos)
        except subprocess.CalledProcessError as e:
            print(f"  > Warning: Playlist error: {e.stderr.strip()}")

    total_videos = len(all_video_urls)
    
    new_videos = []
    for video_url in all_video_urls:
        video_id_match = re.search(r"v=([^&]+)", video_url)
        if video_id_match:
            video_id = video_id_match.group(1)
            if video_id not in processed_videos:
                new_videos.append(video_url)
    
    print(f"Found {len(new_videos)} new videos to process.")
    
    if len(new_videos) == 0:
        print("Nothing new to download.")
        return

    print("\nStep 2: Downloading and cleaning subtitles...")
    processed_count = 0
    skipped_count = 0
    
    file_mode = 'a' if os.path.exists(OUTPUT_FILENAME) else 'w'
    
    # Use a temp folder so we don't clutter the directory with .vtt files
    with tempfile.TemporaryDirectory() as temp_dir, open(OUTPUT_FILENAME, file_mode, encoding='utf-8') as corpus_file:
        temp_path = Path(temp_dir)
        
        for i, video_url in enumerate(new_videos):
            print(f"\nProcessing video {i+1}/{len(new_videos)}")
            
            video_id_match = re.search(r"v=([^&]+)", video_url)
            if not video_id_match:
                print(f"  > Warning: Can't find ID in URL. Skipping.")
                skipped_count += 1
                continue
                
            video_id = video_id_match.group(1)
            output_template = temp_path / f"{video_id}.%(ext)s"
            
            video_duration = get_video_duration(video_url)
            duration_minutes = video_duration / 60 if video_duration > 0 else 0
            
            if video_duration > 0:
                print(f"  > Duration: {duration_minutes:.1f} mins")
            
            print("  > Checking for human subtitles...")
            # Try manual subs first
            command_manual = [
                'yt-dlp', '--sub-lang', LANGUAGE_CODE, '--write-sub',
                '--skip-download', '--output', str(output_template), video_url
            ]
            subprocess.run(command_manual, capture_output=True, text=True, encoding='utf-8')
            
            subtitle_file = temp_path / f"{video_id}.{LANGUAGE_CODE}.vtt"

            if not subtitle_file.exists():
                print("  > None found. Checking auto-generated...")
                # Try auto subs next
                command_auto = [
                    'yt-dlp', '--sub-lang', LANGUAGE_CODE, '--write-auto-sub',
                    '--skip-download', '--output', str(output_template), video_url
                ]
                subprocess.run(command_auto, capture_output=True, text=True, encoding='utf-8')

            if subtitle_file.exists():
                print(f"  > Got subtitles.")
                raw_vtt_text = subtitle_file.read_text(encoding='utf-8')
                
                if video_duration == 0:
                    video_duration = estimate_duration_from_subtitles(raw_vtt_text)
                    duration_minutes = video_duration / 60
                    if video_duration > 0:
                        print(f"  > Estimated duration: {duration_minutes:.1f} mins")
                    else:
                        print(f"  > Unknown duration. Assuming 5 mins.")
                        video_duration = 300
                
                sentence_count = 0
                for sentence in clean_and_parse_vtt(raw_vtt_text):
                    corpus_file.write(sentence + '\n')
                    stats.add_line(sentence)
                    sentence_count += 1
                
                stats.add_video_stats(sentence_count, video_duration)
                
                print(f"  > Saved {sentence_count} lines.")
                
                save_processed_video(video_id)
                processed_count += 1
                subtitle_file.unlink()
            else:
                print("  > No Arabic subtitles found. Skipping.")
                skipped_count += 1

    print("\n--- Done ---")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Saved to: {OUTPUT_FILENAME}")
    
    stats.print_summary()

def perform_detailed_eda(filename):
    """
    Runs a quick check on the file to see what we actually got.
    """
    print("\n--- Running EDA ---")
    
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print(f"Error: File {filename} missing or empty.")
        return

    print("Analyzing...")
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    all_text = " ".join(lines)
    words = all_text.split()
    
    num_sentences = len(lines)
    num_words = len(words)
    num_chars = len(all_text)
    num_unique_words = len(set(words))
    
    line_counts = Counter(lines)
    unique_lines = len(line_counts)
    duplicate_lines = num_sentences - unique_lines
    
    content_words = [word for word in words if len(word) > 1]
    
    print(f"\nSentences: {num_sentences:,}")
    print(f"Words: {num_words:,}")
    print(f"Unique Words: {num_unique_words:,}")
    print(f"Duplicates: {duplicate_lines:,}")
    
    processed_videos = load_processed_videos()
    print(f"Videos Scraped: {len(processed_videos):,}")
    
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"File Size: {file_size_mb:.2f} MB")
    
    print("\nEDA Complete.")

if __name__ == "__main__":
    scrape_with_yt_dlp()
    perform_detailed_eda(OUTPUT_FILENAME)
