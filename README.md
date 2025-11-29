# English ↔ Egyptian Arabic Transformer

This project develops a translation system for the Egyptian dialect (Masri), built entirely from scratch. Rather than fine-tuning an existing model, it implements a custom Encoder-Decoder architecture with stability improvements—like replacing LayerNorm with RMSNorm—and a BPE tokenizer designed to handle irregular spelling and slang. The system produces a robust model that reaches a BLEU score of 28.5, showing that reliable translation is achievable even without huge standard datasets.

## Data Collection

### Egyptian Arabic Sources
*  **Reddit Scraping:** Data was collected from 30+ Egyptian subreddits (e.g., r/Egypt, r/Cairo, r/AlexandriaEgy) using a multi-threaded script to handle Reddit API limits.  
 To avoid duplication, a SQLite database was used so that no two workers would scrape the same post or comment.  This resulted in approximately 250,000 posts and comments.
*  **YouTube Transcription:** Over 1,700 hours of Egyptian videos were processed, extracting VTT subtitles to capture the real spoken dialect rather than just formal text.
*  **Generated Parallel Data:** Gemini was used with tailored prompt engineering to create parallel sentences for tricky slang and dialect-specific words, boosting the total count to 700k pairs.


### English Sources
Subsets of standard English datasets (C4, FineWeb, OpenSubtitles) were selected to cover a wide range of topics and writing styles.

## Preprocessing & Cleaning

### Data Sanitation & Filtering
This pipeline employs rigorous cleaning steps to ensure high-quality input:
*   **HTML & Entity Decoding:** Decodes HTML entities (e.g., `&quot;` → `"`) and splits bilingual text lines into separate lists.
*   **Delimiter Standardization:** Standardizes various dataset delimiters (e.g., pipes `|`, tabs `\t`) into a unified format for consistent parsing.
*   **Noise Removal:** Removes sentences with excessive emojis, spam repetition, or mostly numeric content (e.g., 😂😂😂, 12345).
*   **Alignment Quality:** Calculates strict length ratios between source and target sentences to remove bad alignments (e.g., rejecting pairs where English length > 3x Arabic length).
*   **Artifact Removal:** Removes subtitle artifacts (e.g., `[music]`) and promotional phrases (e.g., اشترك في القناة).
*   **Deduplication:** Merges cleaned files, removes exact duplicates, and downsamples frequent duplicates to prevent overfitting.

### Linguistic Engineering
*   **Normalization:** Converted Eastern Arabic numerals (٠-٩) to Western (0-9).
*   **Dialect Mapping:** Created a dictionary of 2,000+ key pairs to correct common typos and map Modern Standard Arabic (MSA) to Egyptian.
    *   Example: لماذا → ليه
    *   Example: كيف → إزاي
    *   Example: أريد → عايز
    *   Example (Typo): مصطفي → مصطفى

## Model Architecture

The model uses a custom Encoder-Decoder (BART configuration) built from scratch with the following specifications:
![Model Architecture](model_architecture.png)

### Core Specs:
*   **Architecture:** Transformer Encoder-Decoder (BART)
*   **Parameters:** ~98.4 Million
*   **Embedding Dimension ($d_{model}$):** 384
*   **Layers:** 8 Encoder / 8 Decoder
*   **Attention Heads:** 12 (Encoder) / 12 (Decoder)
*   **Feed-Forward Dimension:** 1152
*   **Activation Function:** GELU

### Positional & Token Embeddings:
*   **Max Position Embeddings:** 1024 (Learned)
*   **Scale Embedding:** True
*   **Vocab Size:** 90,000 (Custom BPE)

### Stability Patches:
*   **Normalization:** Root Mean Square Normalization (RMSNorm) replaces standard LayerNorm.
*   **Embedding Tying:** Encoder, Decoder, and LM Head share weights to prevent "lobotomy" during fine-tuning.

## Resources

*   **Model Link:** [Hugging Face: Shams03/En-Arz](https://huggingface.co/Shams03/EgyLated)  
*   **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/Shams03/EgyLated)

## Key Engineering Challenges & Solutions

This project addresses critical failure points common in training dialectal models:

**1. The "RMSNorm" Stability Fix**
*   **Problem:** During pre-training with mixed-precision (FP16), the standard Transformer LayerNorm caused gradient explosions, leading to NaN losses.
*   **Solution:** A dynamic architecture patch was implemented to recursively replace all LayerNorm layers with RMSNorm (Root Mean Square Normalization) at runtime. This removes the mean-centering operation, stabilizing the gradients on T4 GPUs without sacrificing performance.

**2. Morphological Pre-Tokenization**
*   **Problem:** Arabic attaches prepositions (e.g., وال, لل, ال) to words. Standard BPE tokenizers treat alkitab (the book) and kitab (book) as totally different tokens, bloating the vocabulary and increasing sparsity.
*   **Solution:** A pre-tokenization rule was applied to separate these sticky prefixes before BPE training.
    *   Input: الكتاب
    *   Pre-tokenized: ال كتاب
    *   Result: The model learns the root word effectively.

**3. The "Lobotomy" Embedding Fix**
*   **Problem:** During fine-tuning, reloading weights via safetensors can sometimes untie the input/output embeddings, causing the model to lose language association ("lobotomy").
*   **Solution:** The fine-tuning script explicitly forces weight sharing between the Encoder, Decoder, and LM Head (`model.shared.weight`) during the loading state, ensuring semantic alignment.

**4. Data Imbalance (Weighted Sampling)**
*   **Problem:** High-quality transcript data is scarce compared to noisier web-scraped data.
*   **Solution:** The dataset was split into quality tiers (S-tier transcripts, A-tier social media) and weighted sampling was applied during training to prioritize high-quality conversational data while maintaining stylistic diversity.

## Performance

| Metric | Score | Note |
| :--- | :--- | :--- |
| **BLEU Score** | 28.5 | [View Kaggle Logs](https://www.kaggle.com/code/shams03/arz-en-bart-finetuning) |
| **Validation Loss** | 2.18 | Stable convergence |
| **Training Loss** | 2.33 | |

## Translation Examples

| Type | English Input | Model Output (Masri) | Notes |
| :--- | :--- | :--- | :--- |
| **Good** | "Get in the car, we have to go now!" | "ادخل العربية، لازم نمشي دلوقتي!" | Captures urgency and dialect terms. |
| **Good** | "I have a very bad feeling about this." | "عندي إحساس وحش أوي بخصوص الموضوع ده." | Natural phrasing. |
| **Good** | "Why are you doing this?" | "انت بتعمل كده ليه؟" | Correct question structure. |
| **Bad** | "The mitochondria is the powerhouse of the cell." | "الأرياريا هو كتلة الخلايا الجذعية"| Limitation: Struggles with scientific terms. |
| **Bad** | "Complex philosophical prose with archaic terms." | "الأفكار الفلسفية بالمصطلحات القديمة" | Limitation: Acceptable but not optimized for complex phrasing. |


## Usage

This model requires a specific patching procedure to load the custom RMSNorm architecture and ensure the correct special tokens are used.

### Requirements
`torch>=2.0.0`, `transformers>=4.30.0`

### Inference Code

```python
import torch
import torch.nn as nn
import re
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# 1. Define Architecture Components
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale

def load_patched_model(repo_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Config and fix Token IDs (overriding config.json errors)
    config = AutoConfig.from_pretrained(repo_id)

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSeq2SeqLM.from_config(config)
    
    # Patch LayerNorm -> RMSNorm
    for name, child in list(model.named_children()):
        def patch_recursive(m):
            for n, c in list(m.named_children()):
                if isinstance(c, nn.LayerNorm):
                    dim = c.normalized_shape[0] if isinstance(c.normalized_shape, (tuple, list)) else c.normalized_shape
                    setattr(m, n, RMSNorm(dim))
                else:
                    patch_recursive(c)
        patch_recursive(model)

    # Load Weights
    try:
        f = hf_hub_download(repo_id, "model.safetensors")
        model.load_state_dict(load_file(f), strict=False)
    except:
        f = hf_hub_download(repo_id, "pytorch_model.bin")
        model.load_state_dict(torch.load(f, map_location="cpu"), strict=False)
        
    return model.to(device).eval(), tokenizer

def fix_arabic(text):
    if not text: return text
    # Re-connect prefixes and fix punctuation
    text = re.sub(r'(^|\s)(ال|لل|وال|بال)\s+(?=\S)', r'\1\2', text)
    text = re.sub(r'\s+([،؟!.,])', r'\1', text)
    return text.strip()

# 2. Run Inference
REPO_NAME = "Shams03/EgyLated" 
model, tokenizer = load_patched_model(REPO_NAME)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    if "token_type_ids" in inputs: del inputs["token_type_ids"]
    
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=128, 
            num_beams=5, 
            early_stopping=True,
        )
    
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    return fix_arabic(raw)

print(translate("I am really happy because the model works."))
# Output: "أنا مبسوط جدا عشان الموديل شغال"
```

## Project Structure

### 1. Scrapers
Scripts responsible for collecting raw data from various platforms.
*   **ScrapeRedditData.py:** Connects to the Reddit API using multi-threading to scrape posts and comments from targeted Egyptian subreddits. Uses SQLite for deduplication.
*   **ScrapeYoutubeSubtitles.py:** A wrapper around yt-dlp that downloads and cleans subtitles from YouTube playlists, stripping timestamps and HTML to extract raw VTT text.
*   **DownloadHuggingFaceDatasets.py:** Downloads large English datasets (C4, FineWeb) with strict file size limits to prevent memory interactions.
*   **DownloadDatasetSubsets.py:** Target specific domains (e.g., lyrics, toxic comments) to improve model stylistic coverage.

### 2. Processors
The cleaning and transformation pipeline.
*   **ProcessArabicTextPipeline.py:** Main linguistic processor. Handles spelling correction, number normalization, and junk phrase filtering.
*   **FilterHighJunkRatio.py:** In-place filter that removes lines containing a high percentage of non-text symbols or corrupt characters.
*   **CleanHtmlEntities.py:** Decodes HTML entities and splits bilingual text lines into separate lists.
*   **NormalizeSeparators.py:** Standardizes various dataset delimiters into a unified format.
*   **FilterTextNoise.py:** Removes sentences with excessive emojis, spam repetition, or mostly numeric content.
*   **AlignAndFilterCorpus.py:** Calculates length ratios between source and target sentences to remove bad alignments (e.g., removing pairs where English length > 3x Arabic length).
*   **DeepTextCleaning.py:** Removes subtitle artifacts (e.g., "[music]") and downsamples frequent duplicates to prevent overfitting.
*   **CombineProcessedFiles.py:** Merges cleaned files, removes exact duplicates, and sorts the final corpus.
*   **SplitArrowToTxt.py:** Converts binary Arrow/Parquet files into standard text files.
*   **ApplyRuleBasedPretokenization.py:** Applies linguistic splitting rules (e.g., splitting "وال" to "و ال") before tokenizer training.

### 3. Analytics
Tools for analyzing dataset health.
*   **GenerateAdvancedCorpusStats.py:** Generates comprehensive reports on vocabulary size, sentence length distribution, and quality metrics.
*   **GenerateBasicCorpusStats.py:** Provides quick counts of unique pairs, duplicates, and empty lines.

### 4. Training
The core machine learning workflows.
*   **PretokenizeRawData.ipynb:** Pre-processes text by splitting Arabic prefixes and English contractions.
*   **TrainBpeTokenizerFromScratch.ipynb:** Trains a custom BPE tokenizer with a 90,000 token vocabulary.
*   **PretrainBartModelFromScratch.ipynb:** Initializes the BART architecture from scratch, applies the RMSNorm fix, and executes pre-training with weighted sampling.
*   **FinetuneTranslationModel.ipynb:** Fine-tunes the model specifically on the English to Arabic translation task, including manual fixes for embedding weight tying.
*   **BpeTrainingLogic.py:** Helper functions for the tokenizer training process.

### 5. Utilities
Maintenance scripts for file handling.
*   **ChunkTextFile.py:** Splits large text streams into chunks of 350 words.
*   **SplitFileIntoQuarters.py:** Shards multi-gigabyte files into smaller segments.
*   **RenumberTextLines.py & FixAndRenumberLines.py:** Corrects malformed line numbering sequences.
*   **DebugCorpusParsing.py:** Inspects files to identify lines causing parser failures.

## Limitations & Warnings

* **Scientific & Complex Text:** The model struggles with scientific terminology and complex phrasing, often producing literal or inaccurate translations.  
* **Names:** Personal names may be mistranslated or inconsistently handled.  
* **Content Warning:** Due to the unfiltered nature of the training data (social media), the model may produce offensive or inappropriate language.  
* **Planned Improvements (V2):** The next version of the model is planned to be trained on a dataset of over 1M parallel sentences. This will specifically target the current model's weaknesses, improving performance on complex text, handling of names, and overall translation quality.


## License

This project is licensed under the MIT License.
