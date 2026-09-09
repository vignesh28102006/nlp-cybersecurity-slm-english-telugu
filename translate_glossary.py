#!/usr/bin/env python3
"""
Cybersecurity Glossary Translation from English to Telugu Using a Small Language Model (SLM)

Pipeline:
1. Extract cybersecurity terms and definitions from English_Cybersecurity_Glossary.pdf using PyMuPDF.
2. Load the compact MarianMT Neural Machine Translation model (Helsinki-NLP/opus-mt-en-dra).
3. Translate terms and definitions using Beam Search (num_beams=4) for enhanced quality.
4. Post-process to remove UI/PO localization metadata artifacts (Name, Comment, (z), etc.).
5. Export clean results to CSV (cybersecurity_glossary_english_telugu.csv) and TXT (translated_glossary.txt).
6. Print clean formatted tabular results with execution timing.
"""

import os
import re
import sys
import time
import pandas as pd
import torch

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import pymupdf
from transformers import MarianMTModel, MarianTokenizer


MODEL_NAME = "Helsinki-NLP/opus-mt-en-dra"
DEFAULT_PDF_PATH = "English_Cybersecurity_Glossary.pdf"
CSV_OUTPUT_PATH = "cybersecurity_glossary_english_telugu.csv"
TXT_OUTPUT_PATH = "translated_glossary.txt"


def extract_glossary_from_pdf(pdf_path=DEFAULT_PDF_PATH):
    """
    Extracts numbered cybersecurity terms and definitions from the PDF document.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Glossary PDF not found at path: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"

    lines = [line.strip() for line in full_text.splitlines() if line.strip()]

    header_patterns = [
        r"^English Cybersecurity Glossary",
        r"^Input glossary for",
        r"^Context: Cybersecurity",
        r"^Purpose:",
        r"^Reference basis:",
        r"^=== PAGE",
        r"^--- PAGE"
    ]

    filtered_lines = [l for l in lines if not any(re.match(p, l, re.IGNORECASE) for p in header_patterns)]

    items = []
    current_num = None
    current_term = None
    current_def = []

    number_pattern = re.compile(r"^(\d+)\.\s+(.+)$")

    for line in filtered_lines:
        m = number_pattern.match(line)
        if m:
            if current_term:
                clean_def = " ".join(current_def).strip()
                if "Reference basis" in clean_def:
                    clean_def = clean_def.split("Reference basis")[0].strip()
                items.append({
                    "id": current_num,
                    "term": current_term,
                    "definition": clean_def
                })
            current_num = int(m.group(1))
            current_term = m.group(2).strip()
            current_def = []
        else:
            if current_term is not None:
                if not line.lower().startswith("reference basis") and not "nist cybersecurity terminology" in line.lower():
                    current_def.append(line)

    if current_term:
        clean_def = " ".join(current_def).strip()
        if "Reference basis" in clean_def:
            clean_def = clean_def.split("Reference basis")[0].strip()
        items.append({
            "id": current_num,
            "term": current_term,
            "definition": clean_def
        })

    return items


def clean_telugu_output(text):
    """
    Cleans UI localization and SentencePiece metadata artifacts (e.g., Comment, Name, (z))
    that originate from the software localization training corpora in OPUS-MT.
    """
    if not text:
        return ""
    cleaned = text
    # 1. Remove specific localization / PO file metadata tokens
    artifact_patterns = [
        r'Comment', r'Name', r'GenericName', r'color', r'Query', 
        r'@\s*item:\s*inlistbox\s*Sort', r'\(z\)', r'\(Z\)', r'\(syncation\)',
        r'progress', r'wulder-', r'wilder-', r'Zero-\s*Day'
    ]
    for p in artifact_patterns:
        cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE)
    
    # 2. Clean remaining bracketed shortcuts and UI symbols
    cleaned = re.sub(r'[\(\)\[\]\{\}\@\:\-_/]+', ' ', cleaned)
    # 3. Remove leftover standalone ASCII words (metadata remnants)
    cleaned = re.sub(r'\b[a-zA-Z]+\b', '', cleaned)
    # 4. Normalize spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def load_translation_model(model_name=MODEL_NAME):
    """
    Loads the compact MarianMT translation model and tokenizer.
    Prints model name, architecture, and exact parameter count.
    """
    print("=" * 70)
    print("LOADING TRANSLATION MODEL")
    print("=" * 70)
    print(f"Model Identifier : {model_name}")

    t0 = time.time()
    cache_snapshot = os.path.expanduser(r"~\.cache\huggingface\hub\models--Helsinki-NLP--opus-mt-en-dra\snapshots\5ecfd223f954f860b67f1b4c42e6f4c70d347581")
    
    if os.path.exists(cache_snapshot):
        target_source = cache_snapshot
        local_flag = True
    else:
        target_source = model_name
        local_flag = False

    try:
        tokenizer = MarianTokenizer.from_pretrained(target_source, local_files_only=local_flag)
        model = MarianMTModel.from_pretrained(target_source, local_files_only=local_flag)
    except Exception:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

    model.eval()
    t1 = time.time()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model Architecture : {model.__class__.__name__} ({model.config.model_type})")
    print(f"Total Parameters   : {total_params:,} ({total_params / 1e6:.2f} Million)")
    print(f"Trainable Params   : {trainable_params:,}")
    print(f"Precision          : FP32 (CPU Native)")
    print(f"Estimated RAM Size : {total_params * 4 / (1024 * 1024):.1f} MB")
    print(f"Model Load Time    : {t1 - t0:.2f} seconds")
    print("=" * 70)

    return tokenizer, model


def translate_text(text, tokenizer, model, target_prefix=">>tel<<", max_length=128, num_beams=4):
    """
    Translates English text to Telugu using Beam Search (num_beams=4) and cleans metadata artifacts.
    """
    if not text or not str(text).strip():
        return "", ""

    input_text = f"{target_prefix} {str(text).strip()}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=2
        )

    raw_translated = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
    cleaned_translated = clean_telugu_output(raw_translated)
    return raw_translated, cleaned_translated


def run_translation_pipeline(pdf_path=DEFAULT_PDF_PATH):
    """
    Executes the end-to-end cybersecurity glossary translation pipeline.
    """
    total_start_time = time.time()

    # Step 1: Extract PDF
    print(f"\n[Step 1/5] Extracting cybersecurity terms from: {pdf_path}")
    extract_start = time.time()
    glossary_items = extract_glossary_from_pdf(pdf_path)
    extract_time = time.time() - extract_start
    print(f"-> Successfully extracted {len(glossary_items)} terms in {extract_time:.3f} seconds.")

    # Step 2: Load Model
    tokenizer, model = load_translation_model()

    # Step 3: Sanity Verification
    print("\n[Step 2/5] Running single-sentence sanity test:")
    sample_sentence = "Firewall protects a network from unauthorized access."
    raw_sample, clean_sample = translate_text(sample_sentence, tokenizer, model)
    print(f"   English Source : \"{sample_sentence}\"")
    print(f"   Raw Model Out  : \"{raw_sample}\"")
    print(f"   Cleaned Telugu : \"{clean_sample}\"")

    # Step 4: Translate Glossary
    print(f"\n[Step 3/5] Translating {len(glossary_items)} cybersecurity terms and definitions using Beam Search (k=4)...")
    trans_start = time.time()

    results = []
    for i, item in enumerate(glossary_items, 1):
        term_en = item["term"]
        def_en = item["definition"]

        # Translate term
        term_raw, term_clean = translate_text(term_en, tokenizer, model, max_length=32, num_beams=4)

        # Translate definition
        def_raw, def_clean = translate_text(def_en, tokenizer, model, max_length=128, num_beams=4)

        results.append({
            "ID": item["id"],
            "English_Term": term_en,
            "Telugu_Term_Raw": term_raw,
            "Telugu_Term_Clean": term_clean,
            "English_Definition": def_en,
            "Telugu_Definition_Raw": def_raw,
            "Telugu_Definition_Clean": def_clean
        })

        if i % 10 == 0 or i == len(glossary_items):
            print(f"   Progress: {i:02d}/{len(glossary_items)} terms translated...")

    trans_time = time.time() - trans_start
    avg_per_item = (trans_time / len(results)) if results else 0
    print(f"-> Translation completed in {trans_time:.2f} seconds ({avg_per_item*1000:.1f} ms per term+definition pair).")

    # Step 5: Save Outputs
    print("\n[Step 4/5] Saving results to disk...")
    df = pd.DataFrame(results)
    df.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"-> Saved CSV output to : {CSV_OUTPUT_PATH}")

    with open(TXT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CYBERSECURITY GLOSSARY — ENGLISH TO TELUGU TRANSLATION\n")
        f.write(f"Model: {MODEL_NAME} (~76.89M parameters MarianMT SLM)\n")
        f.write("=" * 80 + "\n\n")
        for row in results:
            f.write(f"[{row['ID']:02d}] {row['English_Term']} -> {row['Telugu_Term_Clean']}\n")
            f.write(f"    Raw Model Output: {row['Telugu_Term_Raw']}\n")
            f.write(f"    English Def     : {row['English_Definition']}\n")
            f.write(f"    Telugu Def      : {row['Telugu_Definition_Clean']}\n\n")
    print(f"-> Saved TXT output to : {TXT_OUTPUT_PATH}")

    # Step 6: Display Clean Formatted Table (for PPT Screenshot)
    print("\n[Step 5/5] Formatted Output for Presentation:")
    print("=" * 85)
    print(f"{'ENGLISH TERM':<32} | {'RAW MODEL OUTPUT':<25} | {'CLEANED TELUGU'}")
    print("=" * 85)
    for row in results[:20]:
        print(f"{row['English_Term']:<32} | {row['Telugu_Term_Raw']:<25} | {row['Telugu_Term_Clean']}")
    print("... [Remaining terms formatted in CSV/TXT]")
    print("=" * 85)

    total_time = time.time() - total_start_time
    print(f"\n[Done] Complete pipeline finished in {total_time:.2f} seconds.")
    return df


if __name__ == "__main__":
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH
    run_translation_pipeline(pdf_file)
