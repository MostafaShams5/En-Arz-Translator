"""
Author: Shams
Description:
This script downloads big English datasets from Hugging Face.
It lets me set a size limit (like 500MB) for each dataset so I don't download the entire thing, which could be terabytes.
It saves the data into chunked text files.
"""

import os
from datasets import load_dataset
import re

# Maximum size for one chunk file (10MB)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 

def sanitize_name(name):
    # Makes the dataset name safe to use as a folder name
    return re.sub(r'[\s\(\)/-]', '_', name).strip('_').lower()

def test_dataset_availability(source_name, subset=None):
    """
    Checks if I can actually access the dataset before trying to download it.
    """
    try:
        print(f"Checking access to: {source_name} (subset: {subset})...")
        if subset:
            # streaming=True connects without downloading
            dataset = load_dataset(source_name, subset, streaming=True, split='train', trust_remote_code=False)
        else:
            dataset = load_dataset(source_name, streaming=True, split='train', trust_remote_code=False)
        
        # Try to grab one item to confirm it works
        first_item = next(iter(dataset))
        print(f"Success: {source_name} is working.")
        print(f"Columns found: {list(first_item.keys())}")
        return True, list(first_item.keys())
    except Exception as e:
        print(f"Failed to access {source_name}: {e}")
        return False, []

def get_text_column_for_source(source_name, available_columns):
    # Datasets use different names for the main text. I check for common ones.
    text_column_candidates = ['text', 'content', 'body', 'article', 'document', 'passage']
    
    for candidate in text_column_candidates:
        if candidate in available_columns:
            return candidate
    
    # Fallback: look for any column containing 'text' or 'body'
    for col in available_columns:
        if any(keyword in col.lower() for keyword in ['text', 'content', 'body', 'article']):
            return col
    
    # If all else fails, take the first column
    return available_columns[0] if available_columns else 'text'

def process_source(source_info, base_output_dir):
    """
    Reads the stream, counts bytes, and writes files until limit is hit.
    """
    source_name, subset, target_mb, text_column = source_info
    
    if target_mb <= 0:
        print(f"Skipping {source_name} (Target size is 0).")
        return

    sanitized_source_name = sanitize_name(source_name)
    output_dir = os.path.join(base_output_dir, sanitized_source_name)
    os.makedirs(output_dir, exist_ok=True)
    
    target_source_bytes = target_mb * 1024 * 1024
    
    print("-" * 60)
    print(f"Processing: {source_name}")
    print(f"Target Limit: {target_mb} MB")
    print(f"Reading Column: {text_column}")
    print(f"Saving to: {output_dir}")
    print("-" * 60)

    try:
        if subset:
            dataset_stream = load_dataset(source_name, subset, streaming=True, split='train', trust_remote_code=False)
        else:
            dataset_stream = load_dataset(source_name, streaming=True, split='train', trust_remote_code=False)
    except Exception as e:
        print(f"Critical Error loading {source_name}: {e}")
        return

    file_counter = 1
    total_bytes_written_for_source = 0
    current_file_content = []
    current_file_size_bytes = 0

    for doc in dataset_stream:
        # Check if we hit our MB limit
        if total_bytes_written_for_source >= target_source_bytes:
            print(f"Limit of {target_mb} MB reached. Stopping.")
            break
            
        text = doc.get(text_column, '')
        if not text or not isinstance(text, str):
            continue
            
        doc_bytes = len(text.encode('utf-8'))

        # If current list of texts is bigger than 10MB, write to file
        if current_file_size_bytes + doc_bytes > MAX_FILE_SIZE_BYTES and current_file_content:
            output_filename = f"{sanitized_source_name}_{file_counter:03d}.txt"
            output_filepath = os.path.join(output_dir, output_filename)
            
            print(f"  -> Writing {output_filename} ({current_file_size_bytes / (1024*1024):.2f} MB)")
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write("\n\n".join(current_file_content))
            
            total_bytes_written_for_source += current_file_size_bytes
            file_counter += 1
            current_file_content = []
            current_file_size_bytes = 0

        current_file_content.append(text)
        current_file_size_bytes += doc_bytes

    # Save whatever is left in memory after the loop finishes
    if current_file_content:
        output_filename = f"{sanitized_source_name}_{file_counter:03d}.txt"
        output_filepath = os.path.join(output_dir, output_filename)
        
        print(f"  -> Writing final file {output_filename}")
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write("\n\n".join(current_file_content))
        
        total_bytes_written_for_source += current_file_size_bytes

    print(f"Finished {source_name}. Total written: {total_bytes_written_for_source / (1024*1024):.2f} MB")

def build_corpus_with_user_params():
    print("=====================================================")
    print("=== ENGLISH CORPUS BUILDER (VERIFIED SOURCES)   ===")
    print("=====================================================\n")
    
    print("Checking dataset availability...")
    
    # List of datasets I might want to use
    candidates = [
        ("HuggingFaceFW/fineweb-edu", None),
        ("HuggingFaceFW/fineweb", None),
        ("allenai/c4", "en"),
        ("monology/pile-uncopyrighted", None),
        ("bookcorpus/bookcorpus", None),
        ("cnn_dailymail", "3.0.0"),
        ("squad", None),
        ("wikitext", "wikitext-103-v1"),
    ]
    
    working_sources = []
    
    for source_name, subset in candidates:
        is_working, columns = test_dataset_availability(source_name, subset)
        if is_working:
            text_column = get_text_column_for_source(source_name, columns)
            working_sources.append((source_name, subset, text_column))
    
    if not working_sources:
        print("No working datasets found. Check internet connection.")
        return
    
    print(f"Found {len(working_sources)} available datasets.")
    
    # Setting how much I want from each source (in MB)
    TARGET_SIZES_MB = {
        "HuggingFaceFW/fineweb-edu": 400,
        "HuggingFaceFW/fineweb": 700,
        "allenai/c4": 500,
        "monology/pile-uncopyrighted": 600,
        "bookcorpus/bookcorpus": 200,
        "cnn_dailymail": 150,
        "squad": 100,
        "wikitext": 250,
    }

    current_working_dir = os.getcwd()
    main_corpus_dir = os.path.join(current_working_dir, "eng-corpus")
    os.makedirs(main_corpus_dir, exist_ok=True)
    
    print(f"Output directory: {main_corpus_dir}\n")
    
    sources_to_process = []
    total_size = 0
    
    for source_name, subset, text_column in working_sources:
        target_mb = TARGET_SIZES_MB.get(source_name, 100)
        total_size += target_mb
        sources_to_process.append((source_name, subset, target_mb, text_column))
        
        print(f"Queueing: {source_name} | Limit: {target_mb} MB")
    
    print(f"Total requested size: {total_size} MB\n")
    
    for source_info in sources_to_process:
        process_source(source_info, main_corpus_dir)

    print("\nProcess Complete.")

if __name__ == "__main__":
    build_corpus_with_user_params()
