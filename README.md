# End-to-End English ↔ Egyptian Arabic (ARZ) Transformer

This repository contains the complete pipeline for building a bilingual Encoder-Decoder model designed specifically for translating **English to the Egyptian Arabic dialect (Masri)**. Unlike standard models trained on Modern Standard Arabic (MSA), this project engineers the entire stack—from data collection and cleaning to the neural architecture itself—to handle code-switching, slang, and dialectal morphology.

## Model Performance

Despite the low-resource nature of the dialect, the model achieves strong results on held-out test sets.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **BLEU Score** | **28.5** | High result for a dialectal task |
| **Validation Loss** | 2.18 | Stable convergence |
| **Training Loss** | 2.33 | |

**Resources:**
* **Model Weights:** [Hugging Face: Shams03/En-Arz](https://huggingface.co/Shams03/En-Arz)
* **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/Shams03/EnglishToArz)
* **Training Logs:** [Kaggle Notebook](https://www.kaggle.com/code/shams03/arz-en-bart-finetuning)

---

## Key Engineering Achievements

This project required solving significant engineering challenges regarding stability, data scarcity, and linguistic complexity.

### 1. Architectural Stability (RMSNorm Fix)
During the pre-training phase, the standard Transformer LayerNorm caused gradient instability when training with FP16 precision, leading to loss divergence.
* **Solution:** I implemented a custom script to physically replace all `LayerNorm` layers with **RMSNorm** (Root Mean Square Normalization) within the initialized model architecture. This modification successfully stabilized the gradients on NVIDIA T4 hardware.

### 2. ETL Data Pipeline
To address data scarcity, I engineered a pipeline to collect approximately 3GB of high-quality dialectal data.
* **Social Media Mining:** Developed a multi-threaded scraper for the Reddit API to bypass rate limits. This targeted **26 specific Egyptian subreddits** (e.g., `r/Egypt`, `r/Cairo`, `r/Alexandria`), collecting over **250,000 posts and comments** into a local SQLite3 database.
* **Audio Transcription:** Processed over 1,700 hours of YouTube videos to extract VTT subtitles. This allowed me to capture pure spoken dialect, which often differs significantly from written text.
* **Synthetic Alignment:** Leveraged Large Language Models (LLMs) to generate synthetic translations for difficult domains like rap lyrics and slang dictionaries.

### 3. Linguistic Pre-processing
Arabic morphology poses challenges for standard tokenizers due to clitics (prefixes attached to words). I applied rigorous rule-based transformations:

* **Morphological Stripping:** I implemented pre-tokenization rules to separate sticky prefixes (e.g., "wa-", "bi-", "al-") from root words before training.
* **Dialect Standardization:** Created a mapping dictionary of over 2,000 entries to normalize spelling and convert Standard Arabic to Egyptian.

| Category | Source Term | Egyptian Target (ARZ) | Notes |
| :--- | :--- | :--- | :--- |
| **Translation** | لماذا (*Limatha*) | ليه (*Leh*) | "Why" - MSA vs Dialect |
| **Translation** | كيف (*Kayfa*) | إزاي (*Ezzay*) | "How" - MSA vs Dialect |
| **Grammar** | سوف (*Sawfa*) | هـ (*Ha-* prefix) | Future tense marker |
| **Vocabulary** | أريد (*Oreed*) | عايز (*Ayez*) | "I want" |
| **Typo Fix** | عايذ (*Ayeth*) | عايز (*Ayez*) | Correcting common Z/Th spelling errors |

---

## Project Structure

### 1. Scrapers
Scripts responsible for collecting raw data from various platforms.
* `ScrapeRedditData.py`: Connects to the Reddit API using multi-threading to scrape **posts and comments** from targeted Egyptian subreddits. Uses SQLite for deduplication.
* `ScrapeYoutubeSubtitles.py`: A wrapper around `yt-dlp` that downloads and cleans subtitles from YouTube playlists, stripping timestamps and HTML to extract raw VTT text.
* `StreamTwitterData.py`: Streams massive datasets row-by-row to filter and save content without loading the entire file into memory.
* `DownloadHuggingFaceDatasets.py`: Downloads large English datasets (C4, FineWeb) with strict file size limits to prevent memory interactions.
* `DownloadDatasetSubsets.py`: Target specific domains (e.g., lyrics, toxic comments) to improve model stylistic coverage.

### 2. Processors
The cleaning and transformation pipeline.
* `ProcessArabicTextPipeline.py`: Main linguistic processor. Handles spelling correction, number normalization, and junk phrase filtering.
* `FilterHighJunkRatio.py`: In-place filter that removes lines containing a high percentage of non-text symbols or corrupt characters.
* `CleanHtmlEntities.py`: Decodes HTML entities and splits bilingual text lines into separate lists.
* `NormalizeSeparators.py`: Standardizes various dataset delimiters into a unified format.
* `FilterTextNoise.py`: Removes sentences with excessive emojis, spam repetition, or mostly numeric content.
* `AlignAndFilterCorpus.py`: Calculates length ratios between source and target sentences to remove bad alignments (e.g., removing pairs where English length > 3x Arabic length).
* `DeepTextCleaning.py`: Removes subtitle artifacts (e.g., "[music]") and downsamples frequent duplicates to prevent overfitting.
* `CombineProcessedFiles.py`: Merges cleaned files, removes exact duplicates, and sorts the final corpus.
* `SplitArrowToTxt.py`: Converts binary Arrow/Parquet files into standard text files.
* `ApplyRuleBasedPretokenization.py`: Applies linguistic splitting rules (e.g., splitting "وال" to "و ال") before tokenizer training.

### 3. Analytics
Tools for analyzing dataset health.
* `GenerateAdvancedCorpusStats.py`: Generates comprehensive reports on vocabulary size, sentence length distribution, and quality metrics.
* `GenerateBasicCorpusStats.py`: Provides quick counts of unique pairs, duplicates, and empty lines.

### 4. Training
The core machine learning workflows.
* `Step1_PretokenizeRawData.ipynb`: Pre-processes text by splitting Arabic prefixes and English contractions.
* `Step2_TrainBpeTokenizerFromScratch.ipynb`: Trains a custom BPE tokenizer with a 90,000 token vocabulary.
* `Step3_PretrainBartModelFromScratch.ipynb`: Initializes the BART architecture from scratch, applies the **RMSNorm fix**, and executes pre-training with weighted sampling.
* `Step4_FinetuneTranslationModel.ipynb`: Fine-tunes the model specifically on the **English to Arabic** translation task, including manual fixes for embedding weight tying.
* `BpeTrainingLogic.py`: Helper functions for the tokenizer training process.

### 5. Utilities
Maintenance scripts for file handling.
* `ChunkTextFile.py`: Splits large text streams into chunks of 350 words.
* `SplitFileIntoQuarters.py`: Shards multi-gigabyte files into smaller segments.
* `RenumberTextLines.py` & `FixAndRenumberLines.py`: Corrects malformed line numbering sequences.
* `DebugCorpusParsing.py`: Inspects files to identify lines causing parser failures.

---

## Model Architecture

The model uses a **BART-style Encoder-Decoder** architecture built from scratch.

* **Parameters:** ~98.4 Million
* **Layers:** 8 Encoder / 8 Decoder
* **Hidden Dimension:** 384
* **Feed-Forward Dimension:** 1152
* **Attention Heads:** 12
* **Tokenizer:** Custom BPE (90k vocabulary)
* **Data Strategy:** Weighted sampling to prioritize high-quality transcripts over web-scraped data.

---

## Usage

To use the model in Python, you must handle two things:
1. **Architecture Patching:** You must replace `LayerNorm` with `RMSNorm` after loading the model structure, as the model was trained with this custom modification.
2. **Post-Processing:** You must re-attach Arabic prefixes (clitics) that were split during tokenization.

```python
import torch
import torch.nn as nn
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- 1. Architecture Patch (RMSNorm) ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale

def replace_layernorm_with_rmsnorm(module: nn.Module):
    """
    Recursively replaces all LayerNorm layers with RMSNorm.
    Essential for matching the trained model's architecture.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            dim = child.normalized_shape[0] if isinstance(child.normalized_shape, (tuple, list)) else child.normalized_shape
            rms = RMSNorm(dim=dim, eps=1e-6)
            # Transfer weights from the loaded LayerNorm to the new RMSNorm
            if hasattr(child, 'weight'): 
                rms.scale = child.weight
            setattr(module, name, rms)
        else:
            replace_layernorm_with_rmsnorm(child)

# --- 2. Post-Processing Function ---
def fix_arabic_output(text):
    if not text: return text
    # Glue Prefixes: Re-attaches 'al', 'wa', 'bi' etc.
    text = re.sub(r'(^|\s)(ال|لل|وال|بال)\s+(?=\S)', r'\1\2', text)
    # Glue Punctuation
    text = re.sub(r'\s+([،؟!.,])', r'\1', text)
    return text.strip()

# --- 3. Load & Patch Model ---
model_id = "Shams03/En-Arz"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Apply the RMSNorm fix
replace_layernorm_with_rmsnorm(model)

# --- 4. Translate Function ---
def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    if "token_type_ids" in inputs: del inputs["token_type_ids"]
    
    # Generate with Beam Search
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128,
        num_beams=5,
        repetition_penalty=1.5
    )
    
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return fix_arabic_output(raw_output)

# Example Usage
print(translate("I am really happy to see you."))
# Output: "أنا مبسوط أوي إني شفتك"
