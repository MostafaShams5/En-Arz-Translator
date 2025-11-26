"""
Author: Shams
Description:
This is the big analysis script. 
It does a deep dive into the corpus, making histograms, word clouds, 
and checking detailed quality metrics like weird characters or mixed scripts.
It generates a full report in a folder so I can look at it later.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from wordcloud import WordCloud
import warnings
import re
from datetime import datetime
import json
from pathlib import Path
import unicodedata
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try importing plotly, but don't crash if I don't have it
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly missing - skipping interactive stuff.")

# --- Config ---
ARABIC_FILE = 'corpus.ar'
ENGLISH_FILE = 'corpus.en'
REPORT_DIR = 'enhanced_eda_report'
DETAILED_LOG_DIR = os.path.join(REPORT_DIR, 'detailed_logs')
VISUALIZATIONS_DIR = os.path.join(REPORT_DIR, 'visualizations')
STATISTICS_DIR = os.path.join(REPORT_DIR, 'statistics')

# Regex for Arabic specific things
ARABIC_DIACRITICS = re.compile(r'[\u064B-\u0652\u0670\u0640]')
ARABIC_PUNCTUATION = re.compile(r'[،؛؟]')
ARABIC_NUMBERS = re.compile(r'[٠-٩]')

# Length Rules
LENGTH_THRESHOLDS = {
    'very_short': 3,
    'short': 5,
    'medium': 15,
    'long': 50,
    'very_long': 100,
    'extreme': 200
}

# Ratio Rules
RATIO_THRESHOLDS = {
    'perfect': (0.9, 1.1),
    'good': (0.7, 1.4),
    'acceptable': (0.5, 2.0),
    'concerning': (0.3, 3.0),
    'extreme': (0.1, 10.0)
}

# Quality Rules
QUALITY_METRICS = {
    'min_words_per_sentence': 2,
    'max_words_per_sentence': 150,
    'min_chars_per_word': 1,
    'max_chars_per_word': 50,
    'suspicious_repetition_threshold': 10
}

# --- Functions ---

def setup_directories():
    """Make the folders if they don't exist."""
    directories = [REPORT_DIR, DETAILED_LOG_DIR, VISUALIZATIONS_DIR, STATISTICS_DIR]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"Created directories in '{REPORT_DIR}/'")

def setup_arabic_fonts():
    """Find a font that can actually print Arabic."""
    arabic_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # cause i am using Ubuntu (Linux)
    ]
    
    for font_path in arabic_fonts:
        if os.path.exists(font_path):
            return font_path
    
    try:
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']
    except:
        pass
    
    return None

def clean_arabic_text(text):
    """Normalize Arabic text (remove tatweel, fix alefs, etc)."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = unicodedata.normalize('NFKC', text)
    
    text_no_diacritics = ARABIC_DIACRITICS.sub('', text)
    
    for arabic_digit, western_digit in zip('٠١٢٣٤٥٦٧٨٩', '0123456789'):
        text = text.replace(arabic_digit, western_digit)
    
    # Normalize common letter variations
    text = text.replace('ي', 'ي').replace('ى', 'ي')
    text = text.replace('ة', 'ه').replace('ه', 'ه')
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    
    return text.strip()

def analyze_text_quality(text, language='ar'):
    """Check if the sentence looks like garbage or not."""
    if pd.isna(text) or not isinstance(text, str):
        return {'valid': False, 'issues': ['empty_or_invalid']}
    
    issues = []
    text = text.strip()
    
    if not text:
        return {'valid': False, 'issues': ['empty']}
    
    words = text.split()
    
    # Check lengths
    if len(words) < QUALITY_METRICS['min_words_per_sentence']:
        issues.append('too_short')
    if len(words) > QUALITY_METRICS['max_words_per_sentence']:
        issues.append('too_long')
    
    # Check individual words
    for word in words:
        if len(word) < QUALITY_METRICS['min_chars_per_word']:
            issues.append('very_short_word')
        if len(word) > QUALITY_METRICS['max_chars_per_word']:
            issues.append('very_long_word')
    
    # Check for repeated words (spam)
    word_counts = Counter(words)
    max_repetition = max(word_counts.values()) if word_counts else 0
    if max_repetition > QUALITY_METRICS['suspicious_repetition_threshold']:
        issues.append('excessive_repetition')
    
    # Language checks
    if language == 'ar':
        if not re.search(r'[\u0600-\u06FF]', text):
            issues.append('no_arabic_chars')
        
        # Check if it has Latin chars in Arabic text
        if re.search(r'[a-zA-Z]', text) and re.search(r'[\u0600-\u06FF]', text):
            latin_ratio = len(re.findall(r'[a-zA-Z]', text)) / len(text)
            if latin_ratio > 0.3:
                issues.append('mixed_script')
    
    elif language == 'en':
        if not re.search(r'[a-zA-Z]', text):
            issues.append('no_english_chars')
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'word_count': len(words),
        'char_count': len(text),
        'unique_words': len(set(words)),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0
    }

def categorize_by_length(word_count):
    """Sort lengths into buckets."""
    if word_count <= LENGTH_THRESHOLDS['very_short']:
        return 'very_short'
    elif word_count <= LENGTH_THRESHOLDS['short']:
        return 'short'
    elif word_count <= LENGTH_THRESHOLDS['medium']:
        return 'medium'
    elif word_count <= LENGTH_THRESHOLDS['long']:
        return 'long'
    elif word_count <= LENGTH_THRESHOLDS['very_long']:
        return 'very_long'
    else:
        return 'extreme'

def categorize_ratio(ratio):
    """Sort ratios into buckets."""
    for category, (min_val, max_val) in RATIO_THRESHOLDS.items():
        if min_val <= ratio <= max_val:
            return category
    return 'extreme'

def advanced_frequency_analysis(tokens_series, language='ar', top_n=50):
    """Count words and find patterns."""
    all_tokens = [token for sublist in tokens_series for token in sublist if token.strip()]
    
    if not all_tokens:
        return {}
    
    token_freq = Counter(all_tokens)
    
    by_length = defaultdict(list)
    for token in all_tokens:
        by_length[len(token)].append(token)
    
    # Linguistic pattern checks
    if language == 'ar':
        patterns = {
            'starts_with_al': len([t for t in all_tokens if t.startswith('ال')]),
            'ends_with_ha': len([t for t in all_tokens if t.endswith('ها')]),
            'contains_numbers': len([t for t in all_tokens if re.search(r'\d', t)]),
            'single_char': len([t for t in all_tokens if len(t) == 1])
        }
    else:
        patterns = {
            'capitalized': len([t for t in all_tokens if t[0].isupper()]),
            'all_caps': len([t for t in all_tokens if t.isupper()]),
            'contains_numbers': len([t for t in all_tokens if re.search(r'\d', t)]),
            'single_char': len([t for t in all_tokens if len(t) == 1])
        }
    
    return {
        'frequency': dict(token_freq.most_common(top_n)),
        'total_tokens': len(all_tokens),
        'unique_tokens': len(token_freq),
        'avg_token_length': np.mean([len(t) for t in all_tokens]),
        'length_distribution': {k: len(v) for k, v in by_length.items()},
        'patterns': patterns,
        'hapax_legomena': len([t for t, c in token_freq.items() if c == 1]) 
    }

def create_enhanced_visualizations(df, font_path=None):
    """Make the charts."""
    
    print("Making plots...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    if font_path:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    colors = {
        'arabic': '#2E86C1',
        'english': '#E74C3C',
        'neutral': '#7D3C98',
        'accent': '#F39C12'
    }
    
    # 1. Big Distribution Grid
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Length Analysis', fontsize=24, fontweight='bold')
    
    sns.histplot(df['ar_word_count'], color=colors['arabic'], alpha=0.7, kde=True, ax=axes[0,0])
    axes[0,0].set_title('Arabic Words')
    axes[0,0].set_xlabel('Words per Sentence')
    
    sns.histplot(df['en_word_count'], color=colors['english'], alpha=0.7, kde=True, ax=axes[0,1])
    axes[0,1].set_title('English Words')
    axes[0,1].set_xlabel('Words per Sentence')
    
    sns.histplot(df['ar_word_count'], color=colors['arabic'], alpha=0.5, label='Arabic', ax=axes[0,2])
    sns.histplot(df['en_word_count'], color=colors['english'], alpha=0.5, label='English', ax=axes[0,2])
    axes[0,2].set_title('Comparison')
    axes[0,2].legend()
    
    sns.histplot(df['ar_char_count'], color=colors['arabic'], alpha=0.7, kde=True, ax=axes[1,0])
    axes[1,0].set_title('Arabic Characters')
    axes[1,0].set_xlabel('Chars per Sentence')
    
    sns.histplot(df['en_char_count'], color=colors['english'], alpha=0.7, kde=True, ax=axes[1,1])
    axes[1,1].set_title('English Characters')
    axes[1,1].set_xlabel('Chars per Sentence')
    
    sns.histplot(df['len_ratio'], color=colors['neutral'], alpha=0.7, kde=True, ax=axes[1,2])
    axes[1,2].axvline(1.0, color='red', linestyle='--', alpha=0.8, label='1:1 Ratio')
    axes[1,2].set_title('Ratio Distribution')
    axes[1,2].set_xlabel('Ratio (EN/AR)')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, '01_distributions.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. Heatmap
    quality_matrix = []
    categories = ['very_short', 'short', 'medium', 'long', 'very_long', 'extreme']
    
    for ar_cat in categories:
        row = []
        for en_cat in categories:
            count = len(df[(df['ar_length_category'] == ar_cat) & (df['en_length_category'] == en_cat)])
            row.append(count)
        quality_matrix.append(row)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(quality_matrix, 
                xticklabels=[f'EN {cat}' for cat in categories],
                yticklabels=[f'AR {cat}' for cat in categories],
                annot=True, fmt='d', cmap='YlOrRd',
                cbar_kws={'label': 'Pairs'})
    plt.title('Length Cross-Analysis', fontsize=18, pad=20)
    plt.xlabel('English Categories')
    plt.ylabel('Arabic Categories')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, '02_heatmap.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 3. Ratio Analysis
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Ratio Analysis', fontsize=20, fontweight='bold')
    
    # Scatter
    scatter = axes[0,0].scatter(df['ar_word_count'], df['en_word_count'], 
                              c=df['len_ratio'], cmap='viridis', alpha=0.6, s=20)
    axes[0,0].plot([0, df['ar_word_count'].max()], [0, df['ar_word_count'].max()], 
                   'r--', alpha=0.8, label='1:1 Line')
    axes[0,0].set_xlabel('Arabic Words')
    axes[0,0].set_ylabel('English Words')
    axes[0,0].set_title('Word Count Correlation')
    axes[0,0].legend()
    plt.colorbar(scatter, ax=axes[0,0], label='Ratio')
    
    # Ratio bar chart
    ratio_by_length = df.groupby('ar_length_category')['len_ratio'].agg(['mean', 'std', 'median'])
    x_pos = range(len(ratio_by_length))
    axes[0,1].bar(x_pos, ratio_by_length['mean'], yerr=ratio_by_length['std'], 
                  capsize=5, alpha=0.7, color=colors['neutral'])
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(ratio_by_length.index, rotation=45)
    axes[0,1].set_ylabel('Avg Ratio')
    axes[0,1].set_title('Ratio by Length')
    axes[0,1].axhline(1.0, color='red', linestyle='--', alpha=0.8)
    
    # Pie chart
    ratio_counts = df['ratio_category'].value_counts()
    axes[1,0].pie(ratio_counts.values, labels=ratio_counts.index, autopct='%1.1f%%',
                  colors=plt.cm.Set3(np.linspace(0, 1, len(ratio_counts))))
    axes[1,0].set_title('Ratio Quality')
    
    # Cumulative plot
    sorted_ratios = np.sort(df['len_ratio'])
    y_vals = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
    axes[1,1].plot(sorted_ratios, y_vals, color=colors['neutral'], linewidth=2)
    axes[1,1].axvline(1.0, color='red', linestyle='--', alpha=0.8, label='Perfect')
    axes[1,1].set_xlabel('Ratio')
    axes[1,1].set_ylabel('Cumulative Prob')
    axes[1,1].set_title('Ratio Cumulative Dist')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, '03_ratio_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Visualizations done.")

def generate_frequency_plots(df, font_path=None):
    """Generate the word clouds."""
    
    print("Making frequency plots...")
    
    ar_analysis = advanced_frequency_analysis(df['ar_words'], 'ar')
    en_analysis = advanced_frequency_analysis(df['en_words'], 'en')
    
    if ar_analysis['frequency']:
        # Arabic WordCloud
        ar_text = ' '.join([word for word, freq in ar_analysis['frequency'].items() for _ in range(min(freq, 100))])
        ar_wordcloud = WordCloud(
            width=1600, height=800,
            background_color='white',
            font_path=font_path,
            colormap='Blues',
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(ar_text)
        
        plt.figure(figsize=(20, 10))
        plt.imshow(ar_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Arabic Words', fontsize=24, pad=20)
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, '04_arabic_wordcloud.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    if en_analysis['frequency']:
        # English WordCloud
        en_text = ' '.join([word for word, freq in en_analysis['frequency'].items() for _ in range(min(freq, 100))])
        en_wordcloud = WordCloud(
            width=1600, height=800,
            background_color='white',
            colormap='Reds',
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(en_text)
        
        plt.figure(figsize=(20, 10))
        plt.imshow(en_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('English Words', fontsize=24, pad=20)
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, '05_english_wordcloud.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Top words bars
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    if ar_analysis['frequency']:
        ar_words = list(ar_analysis['frequency'].keys())[:25]
        ar_counts = list(ar_analysis['frequency'].values())[:25]
        
        axes[0].barh(range(len(ar_words)), ar_counts, color='#2E86C1', alpha=0.8)
        axes[0].set_yticks(range(len(ar_words)))
        axes[0].set_yticklabels(ar_words[::-1]) 
        axes[0].set_xlabel('Frequency')
        axes[0].set_title('Top 25 Arabic', fontsize=16)
        axes[0].grid(axis='x', alpha=0.3)
    
    if en_analysis['frequency']:
        en_words = list(en_analysis['frequency'].keys())[:25]
        en_counts = list(en_analysis['frequency'].values())[:25]
        
        axes[1].barh(range(len(en_words)), en_counts, color='#E74C3C', alpha=0.8)
        axes[1].set_yticks(range(len(en_words)))
        axes[1].set_yticklabels(en_words[::-1]) 
        axes[1].set_xlabel('Frequency')
        axes[1].set_title('Top 25 English', fontsize=16)
        axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, '06_top_words.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Frequency plots done.")
    
    return ar_analysis, en_analysis

def convert_to_serializable(obj):
    """Make things work with JSON."""
    if hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def generate_statistical_report(df, ar_analysis, en_analysis):
    """Write all the numbers to a JSON file."""
    
    print("Writing statistics...")
    
    stats_report = {
        'dataset_overview': {
            'total_sentence_pairs': int(len(df)),
            'unique_pairs': int(len(df.drop_duplicates(['arabic', 'english']))),
            'duplicate_pairs': int(len(df) - len(df.drop_duplicates(['arabic', 'english']))),
            'empty_arabic': int((df['arabic'].str.strip() == '').sum()),
            'empty_english': int((df['english'].str.strip() == '').sum()),
            'generation_timestamp': datetime.now().isoformat()
        },
        
        'length_statistics': {
            'arabic_words': {
                'mean': float(df['ar_word_count'].mean()),
                'median': float(df['ar_word_count'].median()),
                'std': float(df['ar_word_count'].std()),
                'min': int(df['ar_word_count'].min()),
                'max': int(df['ar_word_count'].max()),
                'percentiles': {
                    '25th': float(df['ar_word_count'].quantile(0.25)),
                    '75th': float(df['ar_word_count'].quantile(0.75)),
                    '90th': float(df['ar_word_count'].quantile(0.90)),
                    '95th': float(df['ar_word_count'].quantile(0.95)),
                    '99th': float(df['ar_word_count'].quantile(0.99))
                }
            },
            'english_words': {
                'mean': float(df['en_word_count'].mean()),
                'median': float(df['en_word_count'].median()),
                'std': float(df['en_word_count'].std()),
                'min': int(df['en_word_count'].min()),
                'max': int(df['en_word_count'].max()),
                'percentiles': {
                    '25th': float(df['en_word_count'].quantile(0.25)),
                    '75th': float(df['en_word_count'].quantile(0.75)),
                    '90th': float(df['en_word_count'].quantile(0.90)),
                    '95th': float(df['en_word_count'].quantile(0.95)),
                    '99th': float(df['en_word_count'].quantile(0.99))
                }
            },
            'length_ratio': {
                'mean': float(df['len_ratio'].mean()),
                'median': float(df['len_ratio'].median()),
                'std': float(df['len_ratio'].std()),
                'min': float(df['len_ratio'].min()),
                'max': float(df['len_ratio'].max())
            }
        },
        
        'quality_assessment': {
            'length_categories': {k: int(v) for k, v in df['ar_length_category'].value_counts().to_dict().items()},
            'ratio_categories': {k: int(v) for k, v in df['ratio_category'].value_counts().to_dict().items()},
            'quality_issues': {
                'total_with_issues': int(df['has_quality_issues'].sum()),
                'percentage_with_issues': float(df['has_quality_issues'].mean() * 100),
                'issue_breakdown': {}
            }
        },
        
        'vocabulary_analysis': {
            'arabic': convert_to_serializable(ar_analysis),
            'english': convert_to_serializable(en_analysis)
        },
        
        'correlation_analysis': {
            'word_count_correlation': float(df['ar_word_count'].corr(df['en_word_count'])),
            'char_count_correlation': float(df['ar_char_count'].corr(df['en_char_count'])),
            'length_ratio_vs_ar_length': float(df['len_ratio'].corr(df['ar_word_count'])),
            'length_ratio_vs_en_length': float(df['len_ratio'].corr(df['en_word_count']))
        }
    }
    
    all_issues = []
    for issues_list in df['quality_issues']:
        all_issues.extend(issues_list)
    
    issue_counts = Counter(all_issues)
    stats_report['quality_assessment']['quality_issues']['issue_breakdown'] = {k: int(v) for k, v in dict(issue_counts).items()}
    
    with open(os.path.join(STATISTICS_DIR, 'comprehensive_statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats_report, f, indent=2, ensure_ascii=False)
    
    # Simple markdown report
    report_text = f"""
# Analysis Report
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- Pairs: {stats_report['dataset_overview']['total_sentence_pairs']:,}
- Unique: {stats_report['dataset_overview']['unique_pairs']:,}
- Issues: {stats_report['quality_assessment']['quality_issues']['total_with_issues']:,}

## Lengths
- Arabic Mean: {stats_report['length_statistics']['arabic_words']['mean']:.1f} words
- English Mean: {stats_report['length_statistics']['english_words']['mean']:.1f} words
- Ratio Mean: {stats_report['length_statistics']['length_ratio']['mean']:.2f}

## Vocab
- Arabic Tokens: {ar_analysis['total_tokens']:,}
- English Tokens: {en_analysis['total_tokens']:,}

## Correlation
- Word Count: {stats_report['correlation_analysis']['word_count_correlation']:.3f}
"""
    
    with open(os.path.join(STATISTICS_DIR, 'analysis_report.md'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("Reports done.")
    return stats_report

def generate_detailed_logs(df):
    """Write text files with lists of problematic sentences."""
    
    print("Writing logs...")
    
    # Log quality issues
    quality_issues_path = os.path.join(DETAILED_LOG_DIR, 'quality_issues.txt')
    with open(quality_issues_path, 'w', encoding='utf-8') as f:
        f.write("=== QUALITY ISSUES ===\n\n")
        
        issue_df = df[df['has_quality_issues']].copy()
        if len(issue_df) > 0:
            f.write(f"Found {len(issue_df):,} bad pairs:\n\n")
            
            for idx, row in issue_df.iterrows():
                f.write(f"#{idx} Issues: {', '.join(row['quality_issues'])}\n")
                f.write(f"AR: {row['arabic']}\n")
                f.write(f"EN: {row['english']}\n\n")
    
    # Log weird ratios
    extreme_ratios_path = os.path.join(DETAILED_LOG_DIR, 'extreme_ratios.txt')
    extreme_df = df[df['ratio_category'] == 'extreme'].copy()
    
    with open(extreme_ratios_path, 'w', encoding='utf-8') as f:
        f.write("=== EXTREME RATIOS ===\n\n")
        
        extreme_df['abs_ratio_diff'] = np.abs(extreme_df['len_ratio'] - 1.0)
        extreme_df = extreme_df.sort_values('abs_ratio_diff', ascending=False)
        
        for idx, row in extreme_df.head(500).iterrows(): 
            f.write(f"#{idx} Ratio: {row['len_ratio']:.2f}\n")
            f.write(f"AR: {row['arabic']}\n")
            f.write(f"EN: {row['english']}\n\n")
    
    # Log length outliers
    length_outliers_path = os.path.join(DETAILED_LOG_DIR, 'length_outliers.txt')
    
    def get_outliers(series, factor=1.5):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return series[(series < Q1 - factor * IQR) | (series > Q3 + factor * IQR)]
    
    ar_outliers = get_outliers(df['ar_word_count'])
    en_outliers = get_outliers(df['en_word_count'])
    
    with open(length_outliers_path, 'w', encoding='utf-8') as f:
        f.write("=== LENGTH OUTLIERS ===\n\n")
        
        if len(ar_outliers) > 0:
            f.write(f"Arabic Outliers ({len(ar_outliers):,}):\n")
            for idx in ar_outliers.index[:100]:
                row = df.loc[idx]
                f.write(f"#{idx} ({row['ar_word_count']}w): {row['arabic']}\n")
            f.write("\n")
        
        if len(en_outliers) > 0:
            f.write(f"English Outliers ({len(en_outliers):,}):\n")
            for idx in en_outliers.index[:100]:
                row = df.loc[idx]
                f.write(f"#{idx} ({row['en_word_count']}w): {row['english']}\n")
    
    print("Logs written.")

def create_interactive_dashboard(df):
    """Make an HTML file if Plotly is installed."""
    
    if not PLOTLY_AVAILABLE:
        print("Skipping dashboard.")
        return
    
    print("Creating HTML dashboard...")
    
    try:
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Word Count', 'Ratio', 'Correlation', 'Length Cats', 'Quality', 'Vocab')
        )
        
        fig.add_trace(go.Histogram(x=df['ar_word_count'], name='Arabic', marker_color='#2E86C1'), row=1, col=1)
        fig.add_trace(go.Histogram(x=df['en_word_count'], name='English', marker_color='#E74C3C'), row=1, col=1)
        fig.add_trace(go.Histogram(x=df['len_ratio'], name='Ratio', marker_color='#7D3C98'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['ar_word_count'], y=df['en_word_count'], mode='markers', name='Corr'), row=2, col=1)
        
        length_counts = df['ar_length_category'].value_counts()
        fig.add_trace(go.Pie(labels=length_counts.index, values=length_counts.values, name="Lengths"), row=2, col=2)
        
        quality_counts = df['has_quality_issues'].value_counts()
        fig.add_trace(go.Pie(labels=['No Issues', 'Issues'], values=quality_counts.values, name="Quality"), row=3, col=1)
        
        fig.update_layout(height=1200, title_text="Corpus Dashboard")
        
        dashboard_path = os.path.join(VISUALIZATIONS_DIR, 'interactive_dashboard.html')
        fig.write_html(dashboard_path)
        
        print("Dashboard HTML saved.")
        
    except Exception as e:
        print(f"Failed to create dashboard: {e}")

# --- Main ---
def main():
    
    print("=" * 80)
    print("Starting Advanced EDA")
    print("=" * 80)
    
    setup_directories()
    font_path = setup_arabic_fonts()
    
    if font_path:
        print(f"Using font: {font_path}")
    else:
        print("Using system default font.")
    
    # Phase 1: Load
    print("\nPhase 1: Loading Data")
    
    try:
        print("Reading files...")
        with open(ARABIC_FILE, 'r', encoding='utf-8') as f_ar:
            ar_lines = [line.strip() for line in f_ar.readlines()]
        with open(ENGLISH_FILE, 'r', encoding='utf-8') as f_en:
            en_lines = [line.strip() for line in f_en.readlines()]
        
        if len(ar_lines) != len(en_lines):
            raise ValueError(f"Mismatch! AR: {len(ar_lines)}, EN: {len(en_lines)}")
        
        print(f"Loaded {len(ar_lines):,} pairs.")
        
        df = pd.DataFrame({
            'arabic': ar_lines,
            'english': en_lines
        })
        
        # Clean
        initial_count = len(df)
        df = df[
            (df['arabic'].str.strip() != '') & 
            (df['english'].str.strip() != '')
        ].drop_duplicates().reset_index(drop=True)
        
        print(f"Removed {initial_count - len(df):,} duplicates/empties.")
        print(f"Working with {len(df):,} pairs.")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Phase 2: Features
    print("\nPhase 2: Features")
    print("Processing text...")
    
    df['arabic_clean'] = df['arabic'].apply(clean_arabic_text)
    df['english_clean'] = df['english'].str.strip().str.lower()
    
    df['ar_words'] = df['arabic_clean'].str.split()
    df['en_words'] = df['english_clean'].str.split()
    
    df['ar_word_count'] = df['ar_words'].str.len()
    df['en_word_count'] = df['en_words'].str.len()
    df['ar_char_count'] = df['arabic_clean'].str.len()
    df['en_char_count'] = df['english_clean'].str.len()
    
    df['len_ratio'] = df['en_word_count'] / (df['ar_word_count'] + 1e-6)
    
    df['ar_length_category'] = df['ar_word_count'].apply(categorize_by_length)
    df['en_length_category'] = df['en_word_count'].apply(categorize_by_length)
    df['ratio_category'] = df['len_ratio'].apply(categorize_ratio)
    
    print("Checking quality...")
    ar_quality = df['arabic'].apply(lambda x: analyze_text_quality(x, 'ar'))
    en_quality = df['english'].apply(lambda x: analyze_text_quality(x, 'en'))
    
    df['ar_quality_issues'] = [q['issues'] for q in ar_quality]
    df['en_quality_issues'] = [q['issues'] for q in en_quality]
    
    df['quality_issues'] = df.apply(
        lambda row: list(set(row['ar_quality_issues'] + row['en_quality_issues'])), 
        axis=1
    )
    df['has_quality_issues'] = df['quality_issues'].apply(lambda x: len(x) > 0)
    
    print("Features ready.")
    
    # Phase 3: Stats
    print("\nPhase 3: Stats")
    
    print("\nOverview:")
    print(f"  Total pairs: {len(df):,}")
    print(f"  Issues found: {df['has_quality_issues'].sum():,}")
    print(f"  Perfect ratios: {(df['ratio_category'] == 'perfect').sum():,}")
    
    print("\nLengths:")
    print(f"  Arabic Avg: {df['ar_word_count'].mean():.1f}")
    print(f"  English Avg: {df['en_word_count'].mean():.1f}")
    
    print("\nRatios:")
    print(f"  Mean Ratio: {df['len_ratio'].mean():.2f}")
    print(f"  Correlation: {df['ar_word_count'].corr(df['en_word_count']):.3f}")
    
    # Phase 4: Charts
    print("\nPhase 4: Visuals")
    
    create_enhanced_visualizations(df, font_path)
    ar_analysis, en_analysis = generate_frequency_plots(df, font_path)
    
    # Phase 5: Dashboard
    print("\nPhase 5: Dashboard")
    create_interactive_dashboard(df)
    
    # Phase 6: Reports
    print("\nPhase 6: Reporting")
    stats_report = generate_statistical_report(df, ar_analysis, en_analysis)
    generate_detailed_logs(df)
    
    print("\n" + "="*80)
    print("Done.")
    print("="*80)
    print(f"Outputs are in: '{REPORT_DIR}/'")

if __name__ == "__main__":
    main()
