#!/usr/bin/env python3
"""
setup.py - Tải dữ liệu NLTK và khởi tạo
"""

import nltk
import sys

def download_nltk_data():
    """Tải các dữ liệu NLTK cần thiết"""
    print("📥 Downloading NLTK data...")
    
    try:
        # Punkt tokenizer
        nltk.download('punkt', quiet=False)
        
        # Stopwords
        nltk.download('stopwords', quiet=False)
        
        # WordNet (cho stemming/synonyms)
        nltk.download('wordnet', quiet=False)
        
        # Averaged perceptron tagger
        nltk.download('averaged_perceptron_tagger', quiet=False)
        
        print("✅ NLTK data downloaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

if __name__ == "__main__":
    if download_nltk_data():
        sys.exit(0)
    else:
        sys.exit(1)