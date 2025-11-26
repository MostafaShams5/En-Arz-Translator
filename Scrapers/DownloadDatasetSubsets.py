"""
Author: Shams
Description:
This script downloads specific, smaller datasets (like subtitles or lyrics) to add
variety to the corpus. It has a progress bar and a strict download limit.
"""

import os
from datasets import load_dataset
from tqdm import tqdm

def download_and_process_datasets_with_limits():
    """
    Iterates through the config dict, downloads data up to the limit, and saves text files.
    """
    # Config: Dataset Name -> {Column Name, Limit in MB}
    datasets_to_download = {
        'SetFit/toxic_conversations': {'column': 'text', 'limit_mb': 600},
        'deven367/babylm-100M-open-subtitles': {'column': 'text', 'limit_mb': 600},
        'Cropinky/rap_lyrics_english': {'column': 'text', 'limit_mb': 600},
        'Maxx0/Testing_new_nsfw': {'column': 'message', 'limit_mb': 600},
        'mickume/alt_nsfw': {'column': 'text', 'limit_mb': 600}
    }

    base_dir = 'eng-corpus'
    max_file_size = 10 * 1024 * 1024  # 10MB
    
    os.makedirs(base_dir, exist_ok=True)
    source_statistics = {}

    for dataset_name, config in datasets_to_download.items():
        column_name = config['column']
        limit_mb = config['limit_mb']
        limit_bytes = limit_mb * 1024 * 1024

        print(f"\nProcessing: {dataset_name}")
        print(f"Limit: {limit_mb} MB")

        source_folder_name = dataset_name.replace('/', '_')
        source_dir = os.path.join(base_dir, source_folder_name)
        os.makedirs(source_dir, exist_ok=True)

        try:
            # Streaming mode again to avoid downloading everything first
            dataset = load_dataset(dataset_name, split='train', streaming=True)
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue

        total_data_written = 0
        file_count = 1
        current_file_size = 0

        output_file_path = os.path.join(source_dir, f'eng-{file_count}.txt')
        output_file = open(output_file_path, 'w', encoding='utf-8')

        # tqdm creates the progress bar in the console
        with tqdm(total=limit_bytes, unit='B', unit_scale=True, desc=f"Downloading") as pbar:
            for item in dataset:
                if total_data_written >= limit_bytes:
                    print(f"\nLimit reached for {dataset_name}.")
                    break

                text = item.get(column_name)
                if text and isinstance(text, str):
                    text_bytes = text.encode('utf-8')
                    bytes_to_write = len(text_bytes)

                    # Rotate file if > 10MB
                    if current_file_size + bytes_to_write > max_file_size and current_file_size > 0:
                        output_file.close()
                        file_count += 1
                        output_file_path = os.path.join(source_dir, f'eng-{file_count}.txt')
                        output_file = open(output_file_path, 'w', encoding='utf-8')
                        current_file_size = 0

                    output_file.write(text + '\n')
                    current_file_size += bytes_to_write
                    
                    pbar.update(bytes_to_write)
                    total_data_written += bytes_to_write

        output_file.close()

        # Ensure bar shows 100% if we finished early
        pbar.n = total_data_written
        pbar.refresh()

        source_statistics[dataset_name] = {
            'total_data_written_MB': total_data_written / (1024 * 1024),
            'number_of_files': file_count,
            'subdirectory': source_dir
        }
        print(f"Finished {dataset_name}. Saved to {source_dir}")

    print("\nSummary:")
    if not source_statistics:
        print("No data downloaded.")
    else:
        for source, stats in source_statistics.items():
            print(f"\nSource: {source}")
            print(f"  - Path: {stats['subdirectory']}")
            print(f"  - Size: {stats['total_data_written_MB']:.2f} MB")
            print(f"  - Files: {stats['number_of_files']}")

if __name__ == '__main__':
    download_and_process_datasets_with_limits()
