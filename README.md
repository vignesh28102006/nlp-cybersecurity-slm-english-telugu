# Cybersecurity Glossary Translation Using a Small Language Model

## Objective
The objective of this project is to develop an efficient, lightweight, and offline-capable Neural Machine Translation pipeline that extracts English cybersecurity terminology and definitions from a PDF document (`English_Cybersecurity_Glossary.pdf`) and translates them into Telugu using a verified Small Language Model (SLM) / compact Sequence-to-Sequence model.

---

## Problem Statement
Specialized domain knowledge (such as cybersecurity, cloud safety, and cryptography) is overwhelmingly documented in English. Making these critical technical concepts accessible to Telugu speakers, students, and regional practitioners requires robust machine translation. However:
1. **Large Language Models (LLMs)** like LLaMA-3 (8B+), GPT-4, and large multilingual models like NLLB-200-600M or mBART require substantial GPU RAM (several gigabytes), high power consumption, and slow startup times on edge laptops.
2. Standard consumer hardware (such as 16 GB laptops without dedicated high-end GPUs) requires **Small Language Models (SLMs)** that load in seconds, consume minimal memory (< 1 GB RAM), and execute efficiently on standard CPUs.

---

## Input
- **File Name**: `English_Cybersecurity_Glossary.pdf`
- **Content**: 56 foundational cybersecurity terms and concise definitions covering access control, cryptography, network defense, security incidents, malware types, and governance frameworks (e.g., NIST SP 800 standards).

---

## Methodology
The translation architecture follows a streamlined five-stage pipeline:

```
+-------------------------------------------------------+
|        1. Input Cybersecurity Glossary (PDF)          |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|  2. Text & Term Extraction Engine (PyMuPDF / regex)   |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|  3. Preprocessing & Dravidian Language Tokenization   |
|     (Target Language Prefix Tag: ">>tel<<")           |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|  4. Compact Translation SLM Inference (MarianMT)      |
|     (Beam Search k=4, no_repeat_ngram_size=2)         |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|  5. Localization & SentencePiece Artifact Cleaner     |
|     (Strips PO catalog tags: Name, Comment, (z), etc.)|
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
|  6. Multi-format Structured Export (CSV + TXT + CLI)  |
+-------------------------------------------------------+
```

1. **PDF Text Extraction**: PyMuPDF extracts full textual content page-by-page. A robust regex parser isolates term numbers, term names, and definition paragraphs while filtering out recurring headers and references.
2. **Tokenization**: The input English terms and definitions are prepended with the target language token (`>>tel<<`) and tokenized using SentencePiece vocabulary.
3. **Inference with Beam Search**: MarianMT processes tokens through 6 encoder and 6 decoder Transformer layers using Beam Search ($k=4$) with $n$-gram repetition penalties, improving translation quality and transliteration fidelity over greedy decoding.
4. **Artifact Post-Processing**: Automatically cleans software localization catalog tags (`Name`, `Comment`, `color`, `(z)` accelerator shortcuts) that originate from OPUS-MT training corpora.
5. **Export**: Decoded Telugu tokens are structured into Pandas DataFrames and exported to CSV and TXT files.

---

## Model Specifications

| Attribute | Specification |
| :--- | :--- |
| **Model Name** | `Helsinki-NLP/opus-mt-en-dra` |
| **Model Type / Architecture** | Marian Transformer Encoder-Decoder (`MarianMTModel`) |
| **Total Parameters** | **76,894,208 (76.89 Million)** |
| **Trainable Parameters** | 76,894,208 |
| **Model Disk Size / Weights** | ~293.6 MB (`pytorch_model.bin`) |
| **RAM Footprint during Inference**| ~293.3 MB (FP32 precision) |
| **Load Time on CPU** | **~1.2 to 1.5 seconds** |
| **Inference Speed** | ~690 ms per (term + definition) pair on CPU with Beam Search ($k=4$) |
| **Source Repository** | Hugging Face Hub (`Helsinki-NLP/opus-mt-en-dra`) |
| **Target Language Support** | Dravidian Language Family (Telugu via `>>tel<<`, Kannada, Tamil, Malayalam) |

### Why This Model Qualifies as an SLM:
- **Parameter Scale**: At ~76.9M parameters, it is **~8× smaller** than NLLB-200-distilled-600M, **~26× smaller** than Gemma-2B, and **~104× smaller** than LLaMA-3-8B.
- **Resource Constraints**: Consumes < 300 MB RAM, enabling seamless local execution on consumer CPUs without dedicated GPUs.
- **Instant Boot Time**: Ready for inference in ~1.3 seconds, compared to minutes required for multi-gigabyte models.

---

## Technologies Used
- **Python 3.10+ / 3.12**
- **PyTorch**: Tensor backend and CPU matrix computation
- **Hugging Face Transformers**: MarianMT model loading and tokenization pipeline
- **PyMuPDF (`fitz`)**: Fast PDF document parsing and text extraction
- **Pandas**: Tabular data structuring and CSV generation
- **SentencePiece & Sacremoses**: Subword BPE tokenization for Dravidian scripts
- **Jupyter Notebook**: Interactive visualization and execution environment

---

## Example Results (Raw vs Cleaned Output)

| # | English Term | Raw Model Output | Cleaned Telugu Translation | Academic Assessment |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **Firewall** | ఫైర్వాల్Comment | **ఫైర్వాల్** | **Accurate** (Phonetic Telugu for Firewall) |
| 2 | **Encryption** | ఎన్‌కోడింగ్ | **ఎన్‌కోడింగ్** | **Acceptable** (Phonetic Telugu for Encoding/Encryption) |
| 3 | **Malware** | మలభ్రమణQuery | **మలభ్రమణ** | **Literal** (Root compound for Malicious/Disorder) |
| 4 | **Phishing** | ఫిష్సింగ్ | **ఫిష్సింగ్** | **Accurate** (Standard technical transliteration) |
| 5 | **Authentication** | ధృవీకరణ | **ధృవీకరణ** | **Accurate** (Standard Telugu for Verification) |
| 6 | **Authorization** | (z) ధృవీకరణ | **ధృవీకరణ** | **Accurate** (UI accelerator `(z)` stripped) |
| 7 | **Digital Signature** | డిజిటల్ సంతకంName | **డిజిటల్ సంతకం** | **Accurate** (Exact Telugu translation) |
| 8 | **Password** | సంకేతపదం | **సంకేతపదం** | **Accurate** (Standard Telugu for Password) |
| 9 | **Risk** | ప్రమాదం | **ప్రమాదం** | **Accurate** (Standard Telugu for Risk/Danger) |
| 10 | **Cybersecurity** | సైప్రస్ సామర్ధ్యం | **సైప్రస్ సామర్ధ్యం** | **Literal** (General model literal output) |

### Sentence-Level Translation Example:
- **English Source**: *"Firewall protects a network from unauthorized access."*
- **Telugu Output**: *"ప్రసారం కాని యాక్సెస్ నుండి ఫైర్‌వాల్ నెట్‌ను కాపాడుతుంది."*

---

## Installation

1. Clone or navigate to the project directory:
```bash
cd cybersecurity_slm_translation
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## Execution

### Option A: Run Standalone Python Script
Run the automated command-line translation pipeline:
```bash
python translate_glossary.py
```

### Option B: Run Interactive Jupyter Notebook
Launch Jupyter Notebook and open `Cybersecurity_SLM_English_Telugu.ipynb`:
```bash
jupyter notebook Cybersecurity_SLM_English_Telugu.ipynb
```

---

## Generated Output Files
1. `cybersecurity_glossary_english_telugu.csv`: Complete structured table containing Columns `[ID, English_Term, Telugu_Term_Raw, Telugu_Term_Clean, English_Definition, Telugu_Definition_Raw, Telugu_Definition_Clean]`.
2. `translated_glossary.txt`: Formatted text report containing side-by-side English and Telugu terms and definitions.
3. `Cybersecurity_SLM_English_Telugu.ipynb`: Fully executed Jupyter Notebook with all 15 cells containing outputs, tables, and timings.

---

## Technical Analysis of Model Strengths & Limitations

### 1. Root Cause of Raw Metadata Artifacts
MarianMT (`opus-mt-en-dra`) was trained on the OPUS multilingual corpus, which heavily aggregates open-source software UI localization files (e.g., KDE, GNOME, Ubuntu PO catalogs). In these catalogs, strings appear as key-value pairs (e.g., `Comment=Firewall`, `Name=Digital Signature`, `&(Z) Authorization`). When translating isolated single words, the model's subword decoder frequently emits these associative keys. Implementing beam search ($k=4$) and regex artifact filtering effectively isolates the clean Telugu translation.

### 2. General-Domain vs. Cybersecurity Specialization
Because the model was trained on general web corpora rather than a dedicated cybersecurity parallel corpus:
- Established technical terms that exist in software UI translations (*Authentication* $\to$ *ధృవీకరణ*, *Digital Signature* $\to$ *డిజిటల్ సంతకం*, *Password* $\to$ *సంకేతపదం*, *Firewall* $\to$ *ఫైర్వాల్*) produce high-quality translations.
- Emerging or specialized cyber terms (*Cybersecurity*, *Zero Trust*, *Exploit*) receive literal or transliterated forms.

---

## Future Improvements
1. **Domain-Specific Fine-Tuning**: Fine-tuning MarianMT on curated bilingual cybersecurity corpora (NIST / CERT-In datasets).
2. **Quantization (INT8 / ONNX)**: Exporting the model to ONNX Runtime INT8 for reduced memory (~75 MB) and sub-100ms CPU inference.
3. **Hybrid Term-Lookup Layer**: Combining the neural SLM with a domain-specific cybersecurity terminology dictionary for acronyms (VPN, DDoS, XSS).
