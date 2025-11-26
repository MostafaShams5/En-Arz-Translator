Here is the complete **README.md** file in Markdown format. You can copy the code block below and paste it directly into your GitHub repository.

```markdown
# English to Egyptian Arabic Translator (En-Arz)

This project contains the full source code for training a Transformer model to translate English into Egyptian Arabic (Masri).

Most translation models output Modern Standard Arabic (MSA). I built this model from scratch to handle the actual spoken dialect used in Egypt, specifically focusing on subtitles, movies, TV shows, and casual conversation.

**Dataset Size:** Approximately 40 Million rows (Scraped & Curated).
**Architecture:** Custom Encoder-Decoder (BART-based) with RMSNorm.

**Links:**
*   **Hugging Face Model:** [Shams03/En-Arz](https://huggingface.co/Shams03/En-Arz)
*   **Try it Live:** [Hugging Face Space](https://huggingface.co/spaces/Shams03/EnglishToArz)

---

## How to Use the Model (Python Wrapper)

Because my tokenizer splits Arabic prefixes (like "Al-" or "Bi-") to help the model learn better, the raw output will have spaces where they shouldn't be. You need to use this wrapper code to glue the text back together.

```python
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Formatting Function
def fix_arabic_output(text):
    if not text: return text
    # Glue Prefixes (connect 'Al-', 'Wa-', etc. to the next word)
    text = re.sub(r'(^|\s)(ال|لل|وال|بال)\s+(?=\S)', r'\1\2', text)
    # Glue Punctuation (connect dots and commas to the previous word)
    text = re.sub(r'\s+([،؟!.,])', r'\1', text)
    return text.strip()

# 2. Load Model
model_id = "Shams03/En-Arz"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Note: If you are loading the raw weights manually using a script, 
# you must apply the RMSNorm patch (explained in the Training section below).
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# 3. Translate
def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    if "token_type_ids" in inputs: del inputs["token_type_ids"]
    
    # Generate
    # Assuming 'model' is loaded
    out = model.generate(**inputs, max_new_tokens=128)
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    
    return fix_arabic_output(raw)

print(translate("I have a bad feeling about this."))
# Output: عندي إحساس وحش أوي بخصوص الموضوع ده
```

---

## Project Structure

I organized the project into five main folders. Here is a detailed explanation of every file in this repository.

### 1. Scrapers
These scripts collect the raw text data. I needed a massive amount of data to train this from scratch, so I built custom scrapers for different platforms.

*   **`ScrapeRedditData.py`**: This connects to the Reddit API (PRAW). It runs multiple threads (workers) to scrape posts and comments from specific Egyptian subreddits (like r/Egypt, r/Cairo). It uses a local SQLite database (`processed_posts.db`) to remember which posts it already scraped so it doesn't download duplicate data.
*   **`ScrapeYoutubeSubtitles.py`**: A wrapper around `yt-dlp`. It takes a list of YouTube playlists, downloads the auto-generated or manual subtitles, and strips out all the timestamps and HTML tags to leave just the raw Egyptian Arabic text. It also calculates how many hours of audio I have processed.
*   **`StreamTwitterData.py`**: There is a huge dataset called 'faisalq/ETC' on Hugging Face. Downloading it all takes too much space, so this script streams it row-by-row over the internet, filters the content, and saves it to a text file on the fly.
*   **`DownloadHuggingFaceDatasets.py`**: This downloads massive English datasets (like C4 or FineWeb) but allows me to set a hard size limit (e.g., "Stop after 500MB"). It chunks the data into smaller text files so I don't crash my RAM loading terabytes of data.
*   **`DownloadDatasetSubsets.py`**: Similar to the script above, but for specific datasets like rap lyrics or toxic conversations. This adds stylistic variety to the model so it understands slang and informal speech.

### 2. Processors
Raw data is messy. These scripts clean it, filter it, and turn it into high-quality training data.

*   **`CleanHtmlEntities.py`**: Fixes web junk like `&nbsp;` or `&quot;`. It also splits bilingual lines. If a line looks like "1. Hello ||| اهلا", it splits them into two separate lists for processing.
*   **`NormalizeSeparators.py`**: My data came from different sources. Some used `->` to separate languages, others used `-`. This script forces everything into one standard format: `ID. English ||| Arabic`.
*   **`FilterTextNoise.py`**: The garbage collector. It looks at every sentence and deletes it if it has too many emojis, weird symbols, mostly numbers, or if it looks like spam (e.g., "hahahahaha").
*   **`AlignAndFilterCorpus.py`**: This is a critical step. It calculates the length ratio between the English sentence and the Arabic sentence. If English is 3 words and Arabic is 50 words, it is a bad translation, so this script deletes it. It also moves extremely long sentences to a separate file so they don't mess up batching during training.
*   **`DeepTextCleaning.py`**: The final polish. Removes specific subtitle artifacts like "(music playing)" or "(laughs)". It also limits duplicates; if a sentence appears 100 times, it keeps only 5 copies to prevent the model from memorizing simple phrases.
*   **`CombineProcessedFiles.py`**: Takes all the cleaned text files, merges them into one giant file, removes exact duplicates, and sorts them alphabetically.
*   **`SplitArrowToTxt.py`**: Converts Hugging Face `.arrow` files into standard `.txt` files so my other scripts can read them line-by-line.
*   **`ApplyRuleBasedPretokenization.py`**: Runs before training. It applies linguistic rules, like splitting `w-` (and) or `al-` (the) from Arabic words so the tokenizer can learn them as separate concepts.

### 3. Analytics
I wrote these to analyze the health of the dataset before training.

*   **`GenerateAdvancedCorpusStats.py`**: Generates a comprehensive report (HTML dashboard, JSON stats). It checks word frequency, sentence length distributions, and vocabulary size. It helps me see if I have too much English or too much Arabic.
*   **`GenerateBasicCorpusStats.py`**: A faster version that just gives me the counts (unique pairs, duplicates, empty lines).

### 4. Training (The Core)
This is where the AI model is built. I did not use a pre-made model; I built the architecture from scratch.

*   **`Step1_PretokenizeRawData.ipynb`**: Handles the logic for splitting Arabic prefixes and English contractions (like "don't" -> "do" "n't") before the BPE training starts.
*   **`Step2_TrainBpeTokenizerFromScratch.ipynb`**: Trains a custom Byte-Pair Encoding (BPE) tokenizer. I set the vocabulary size to 90,000 to handle the mix of English, Standard Arabic, and Egyptian Slang. It adds the specific special tokens BART needs (`<s>`, `</s>`).
*   **`Step3_PretrainBartModelFromScratch.ipynb`**:
    *   **Architecture:** It initializes a BART-style Encoder-Decoder (8 layers, 384 hidden dim).
    *   **RMSNorm Fix:** Standard BART uses `LayerNorm`, which causes training to crash (NaN loss) in FP16 mode. I wrote a script to manually replace every `LayerNorm` layer with `RMSNorm`, which is much more stable.
    *   **Weighted Sampling:** I have different data sources (High Quality vs. Scraped). This script mixes them so the model sees 54% high-quality English and 24% high-quality Arabic, ensuring it learns good grammar while still learning slang.
    *   **Manual Loop:** I wrote a custom training loop to save the exact state of the data loader. If training stops, I can resume exactly where I left off without restarting the dataset from row zero.
*   **`Step4_FinetuneTranslationModel.ipynb`**:
    *   Loads the pre-trained weights.
    *   **The Embedding Fix:** When loading weights, PyTorch sometimes disconnects the input and output embeddings. This script manually ties `embed_tokens` to `lm_head` to ensure the model works correctly.
    *   Fine-tunes specifically on Translation (English to Arabic) using the SacreBLEU metric.
*   **`BpeTrainingLogic.py`**: Helper functions for the tokenizer training.

### 5. Utilities
Small tools for fixing file errors.

*   **`ChunkTextFile.py`**: Splits huge book files into smaller chunks of 350 words.
*   **`SplitFileIntoQuarters.py`**: Breaks a massive 10GB dataset file into 10 smaller files so I can open them.
*   **`RenumberTextLines.py`** & **`FixAndRenumberLines.py`**: Sometimes lines get numbered wrong (1, 2, 5, 100). These scripts find the error and renumber the whole file correctly (1, 2, 3, 4...).
*   **`DebugCorpusParsing.py`**: If a file isn't loading, this script reads it line-by-line and tells me exactly which line is broken.

---

## Model Architecture Details

I chose an **Encoder-Decoder** architecture because it is best for translation. The Encoder understands the English context, and the Decoder generates the Arabic response.

*   **Base:** BART (Bidirectional and Auto-Regressive Transformers).
*   **Modifications:**
    *   **RMSNorm:** Replaced LayerNorm to fix gradient explosions.
    *   **SwiGLU:** Used as the activation function for better convergence.
    *   **Embeddings:** Tied input/output embeddings to reduce parameter count.
*   **Training Strategy:**
    1.  **Pre-training:** I taught the model both languages by masking random words and making it guess them (Masked Language Modeling).
    2.  **Fine-tuning:** I taught it to translate by feeding it parallel English-Arabic pairs.

## Installation

1.  Clone this repository.
2.  Install the requirements:

```bash
pip install transformers datasets tokenizers torch pandas numpy scikit-learn matplotlib seaborn praw yt-dlp evaluate sacrebleu safetensors accelerate bitsandbytes
```

Created by **Mostafa Shams**.
```
