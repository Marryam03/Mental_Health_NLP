# Mental Health NLP — Agent Guide

> **Context**: This is a university course project (CIT656 NLP, 2026 Spring) for a Master's in ITCS program. The repository is hosted at `https://github.com/Marryam03/Mental_Health_NLP.git`.

---

## Project Overview

This project builds an NLP pipeline to classify mental health states from short text messages. The core task is to **convert binary labels (`0` = mentally well, `1` = at risk) into ordinal severity scores (`1–10`)** using an ensemble of Large Language Models (LLMs), then fine-tune smaller transformer models on the re-labeled data.

The dataset contains **27,977 rows** of user-generated text related to anxiety, depression, and other mental health issues.

### High-Level Pipeline

1. **Phase 1 — LLM Re-Labeling**: Use multiple LLMs (Mistral, Qwen3, Gemini, Kimi) with few-shot prompting to score each text from 1–10.
2. **Phase 1 Evaluation — Inter-Rater Agreement**: Compare Qwen, Mistral, and Gemini outputs against a Kimi reference baseline using Quadratic Weighted Kappa (QWK), MAE, and Sum of Absolute Differences (SAD). Qwen was selected as the optimal labeler based on lowest deviation.
3. **Phase 2 — Fine-Tuning**: Stratified split into Train (70%), Validation (15%), Test (15%). Train transformer models (`roberta-base`, `microsoft/deberta-v3-large`, `mental-roberta-base`) on the Qwen-generated labels with early stopping.
4. **Phase 3 — Prompt Engineering Testing**: Evaluate additional LLM prompts (Gemma, GPT-OSS) on the held-out test set against the Qwen reference scores.
5. **Evaluation & Paper**: Report accuracy, precision, recall, F1-score, MAE, MSE, QWK, and confusion matrices.

---

## Technology Stack

- **Language**: Python 3.12+
- **Environment**: Jupyter Notebooks (designed for Google Colab, Kaggle, and local execution)
- **Key Libraries** (installed via `!pip install` inside notebooks):
  - `pandas`, `numpy` — data manipulation
  - `openai` — OpenAI-compatible client for NVIDIA API inference
  - `google.generativeai`, `tiktoken` — Gemini API interaction
  - `transformers`, `datasets`, `accelerate` — Hugging Face model training
  - `torch` / `PyTorch` — deep learning backend
  - `scikit-learn` — stratified splitting and metrics
  - `matplotlib`, `seaborn` — visualization
  - `tqdm`, `requests`, `sentencepiece`, `optuna`, `openpyxl`
- **No formal dependency lock file** exists (no `requirements.txt`, `pyproject.toml`, or `setup.py`).

### LLM Inference Endpoints

| Phase | Model | Provider / Base URL |
|-------|-------|---------------------|
| Phase 1 | `mistralai/mistral-small-4-119b-2603` | NVIDIA API (`https://integrate.api.nvidia.com/v1`) |
| Phase 1 | `qwen/qwen3-next-80b-a3b-thinking` | NVIDIA API |
| Phase 1 | `gemini-1.5-flash` / `gemini-3.1-flash-lite` | Google Generative AI (`https://generativelanguage.googleapis.com/v1beta/openai/`) |
| Phase 1 | Kimi (Moonshot) | OpenAI-compatible client; output CSV present in repo |
| Phase 3 | `google/gemma-3n-e4b-it` | NVIDIA API |
| Phase 3 | `openai/gpt-oss-20b` | NVIDIA API |

---

## Project Structure

```
Mental_Health_NLP/
├── Dataset/
│   └── mental_health.csv                          # Source data: (text, label) — 27,977 rows
├── Phase1_Prompt Engineering/
│   ├── Mistral_Prompt_Labeling.ipynb              # Mistral Small 4 119B labeling
│   ├── Qwen3_Prompt_Labeling.ipynb                # Qwen3 Next 80B labeling
│   ├── Gemini_Prompt_Labeling.ipynb               # Gemini 1.5 Flash labeling
│   ├── gemini-prompt-labeling.ipynb               # Variant Gemini notebook (root + here)
│   ├── Phase1_Evaluation.ipynb                    # Inter-rater evaluation vs. Kimi baseline
│   └── Output_Lablels/
│       ├── gemini_labels_output.csv               # Gemini batch output
│       ├── mental_health_classification_results_kimi.csv  # Kimi output
│       └── Results (Qwen & Mistral)               # Google Sheets links
├── Phase2_Fine Tuning/
│   ├── Dataset for Fine-Tuning/
│   │   └── NLP_qwen_labels.xlsx                   # Qwen labels used as training ground truth
│   ├── Fine_Tuning_RoBERTa_base.ipynb             # RoBERTa-base 10-class classifier
│   ├── Fine_Tuning_mental-roberta-base.ipynb      # Mental-RoBERTa variant
│   └── NLP_Fine_Tuning_DeBERTa_Large.ipynb        # DeBERTa-v3-large 10-class classifier
├── Phase3_Prompt Engineering Testing/
│   ├── Phase3_Prompt_Labeling_Gemma&GPT_Labeling.ipynb
│   ├── Test Dataset/
│   │   └── phase3_test.csv                        # 2,799-row test set with qwen_score reference
│   ├── gemma_final_scores.csv                     # Gemma predictions + metrics
│   └── gpt_final_scores.csv                       # GPT predictions + metrics
├── generate_batch_notebooks.py                    # Script generating Mistral & Qwen3 notebooks
├── generate_gemini_notebook.py                    # Script generating Gemini notebook
├── gemini-prompt-labeling.ipynb                   # Root-level variant notebook
└── AGENTS.md                                      # This file
```

### Directory Purposes

- **`Dataset/`**: Holds the raw CSV. Columns: `text` (string), `label` (int: `0` or `1`).
- **`Phase1_Prompt Engineering/`**: Contains the primary labeling notebooks and evaluation. Each notebook implements a batched few-shot prompt and labeling logic for a different LLM. Outputs are saved as CSVs.
- **`Phase2_Fine Tuning/`**: Hugging Face `Trainer`-based training loops inside notebooks. Uses Qwen-generated labels as the ground-truth target.
- **`Phase3_Prompt Engineering Testing/`**: Evaluation notebooks that test prompt variations (Gemma, GPT) on the held-out test set.

---

## Code Organization & Module Divisions

There is **no formal package structure** (no `setup.py`, `pyproject.toml`, or `src/` layout). All logic lives inside Jupyter notebook cells or standalone generator scripts.

### Shared Pattern in Phase 1 Notebooks

Each labeling notebook follows this exact structure:

1. **Imports** — `pandas`, `openai`, `tqdm`, standard library modules.
2. **Configuration** — Hardcoded API keys, model names, base URLs, input/output paths, and rate-limiting constants.
3. **Prompt Template** — A large few-shot prompt string (`SUMMARY_PROMPT`) with 10 scored examples (1–10). The prompt instructs the model to act as a clinical NLP expert and return strict CSV.
4. **Helper Functions**:
   - `build_prompt(csv_text)` — formats the prompt template with a CSV chunk.
   - `extract_csv(raw, expected_rows)` — strips `<think>` blocks and markdown fences, then parses the CSV response.
   - `rate_limited_wait()` — enforces a 28-requests-per-minute cap with a 60-second sliding window and 35-second rest intervals.
   - `label_chunk(chunk_df)` — calls the API with retries (max 5), exponential backoff on 429 errors.
   - `label_dataframe(df)` — iterates the dataframe in chunks, collects outputs, retries failed chunks once, and writes results to CSV.
5. **Execution Block** — `if __name__ == "__main__":` cell that reads the CSV, runs labeling, and prints a preview.

### Notebook Generator Scripts

- **`generate_batch_notebooks.py`**: Programmatically generates `Mistral_Prompt_Labeling.ipynb` and `Qwen3_Prompt_Labeling.ipynb` by embedding the prompt template and core logic as JSON cells.
- **`generate_gemini_notebook.py`**: Programmatically generates `Gemini_Prompt_Labeling.ipynb` with the same pattern but using the Gemini API endpoint.

> **Important**: When modifying the prompt template, update it in **both** generator scripts and then regenerate the notebooks to keep them consistent.

### Rate-Limiting Constants

```python
RPM_LIMIT  = 28   # requests per minute
WINDOW_SEC = 60   # sliding window duration
REST_SEC   = 35   # base sleep between retries
CHUNK_SIZE = 20   # rows per API call (batch mode)
```

### Expected CSV Output Schema (Batch Mode)

```csv
row_id,reasoning,dominant_signals,score,rating
```

- `row_id` — preserved from input to match rows back to original data.
- `reasoning` — 2-3 sentences explaining the score.
- `dominant_signals` — comma-separated list of signal names.
- `score` — integer from 1 to 10.
- `rating` — one of `BEST`, `VERY_GOOD`, `GOOD`, `MODERATE_GOOD`, `NEUTRAL`, `MODERATE_BAD`, `BAD`, `VERY_BAD`, `VERY_VERY_BAD`, `WORST`.

### Labeling Heuristic

- `label == 0` → expected LLM score ≤ 5
- `label == 1` → expected LLM score > 5

---

## Build and Test Commands

There is **no build system** and **no formal test suite** currently.

### How to Run Phase 1 Labeling

1. Open `Phase1_Prompt Engineering/Mistral_Prompt_Labeling.ipynb` (or `Qwen3_Prompt_Labeling.ipynb`, `Gemini_Prompt_Labeling.ipynb`) in Jupyter or Google Colab.
2. Update the `API_KEY`, `INPUT_CSV`, and `OUTPUT_CSV` configuration cells.
3. Run all cells sequentially.
4. The notebook will write a CSV with new columns (e.g., `mistral_reasoning`, `mistral_score`, `mistral_rating`, `mistral_dominant_signals`).

### How to Run Phase 2 Fine-Tuning

1. Ensure `NLP_qwen_labels.xlsx` is available in the notebook environment.
2. Open `Phase2_Fine Tuning/Fine_Tuning_RoBERTa_base.ipynb` (or DeBERTa / Mental-RoBERTa variant).
3. Run the dependency installation cell (`!pip install -q transformers>=4.45.0 datasets scikit-learn>=1.5.0 accelerate openpyxl`).
4. Run all cells sequentially. The notebook handles stratified splitting, tokenization, training, and evaluation.
5. The best model checkpoint is saved to `./roberta_mental_severity/best_model` (or equivalent per notebook).

### How to Run Phase 3 Testing

1. Open `Phase3_Prompt Engineering Testing/Phase3_Prompt_Labeling_Gemma&GPT_Labeling.ipynb`.
2. Update `DATA_PATH` and API key list (`API_KEYS`) in the configuration cells.
3. Run all cells sequentially. The notebook uses multi-key parallel workers with per-key rate limiting.
4. Evaluation cells compute metrics and save `gemma_final_scores.csv` and `gpt_final_scores.csv`.

---

## Code Style Guidelines

- **Notebook-first development**: All production logic is in `.ipynb` files.
- **Prompt consistency**: The few-shot prompt template must remain identical across all labeler notebooks to ensure comparability for inter-rater agreement.
- **Hardcoded configuration**: API keys and paths are set in dedicated cells at the top of each notebook.
- **Sequential execution**: Labeling is intentionally single-threaded per API key to respect rate limits.
- **Error handling**: Failed chunks are collected and retried once in a second pass; permanently failed rows are left as `None`/`NaN`.
- **Checkpointing**: Progress is saved to CSV every 10 chunks during Phase 1 labeling.

---

## Testing Instructions

- **Current state**: No automated tests exist.
- **Manual validation**: Verify labeled outputs by inspecting the generated CSVs and comparing scores against the original binary label heuristic (`label 0` → `score ≤ 5`, `label 1` → `score > 5`).
- **Phase 1 validation**: Compute **Quadratic Weighted Kappa (QWK)**, **MAE**, and **SAD** across LLM labelers using Kimi as the reference baseline. Target QWK ≥ 0.70.
- **Phase 2 validation**: Use stratified train/val/test splits to ensure class balance. Evaluate with accuracy, macro/weighted F1, MAE, and confusion matrices.
- **Phase 3 validation**: Compare Gemma and GPT predictions against the `qwen_score` reference on the held-out test set using the same metrics.

---

## Security Considerations

> ⚠️ **API keys are hardcoded in notebook source cells and generator scripts.**
>
> The Phase 1 notebooks and generator scripts (`generate_batch_notebooks.py`, `generate_gemini_notebook.py`) contain plaintext API keys (`nvapi-...`, `AIzaSy...`).
> - **Do not commit notebooks or generators with live keys** to public repositories.
> - Rotate any exposed keys immediately.
> - Use environment variables (e.g., `os.environ.get("NVIDIA_API_KEY")`) or Colab secrets instead of hardcoded strings.

---

## Dataset Details

- **File**: `Dataset/mental_health.csv`
- **Size**: 27,977 rows
- **Columns**:
  - `text`: Raw user message (lowercased, informal, may contain profanity or distressing content).
  - `label`: Binary indicator — `0` (no mental health issue) or `1` (mental health issue).
- **Fine-Tuning Ground Truth**: `Phase2_Fine Tuning/Dataset for Fine-Tuning/NLP_qwen_labels.xlsx` — contains Qwen-generated `qwen_score`, `qwen_rating`, `qwen_reasoning`, and `qwen_dominant_signals`.
- **Test Set**: `Phase3_Prompt Engineering Testing/Test Dataset/phase3_test.csv` — 2,799 rows including `qwen_score` and `class_label` (`qwen_score - 1`) used as the reference for Phase 3 evaluation.

---

## Development Notes for Agents

- When editing notebooks, preserve the exact prompt template across all labeler variants so inter-rater agreement remains valid. If you change the prompt, regenerate notebooks from `generate_batch_notebooks.py` and `generate_gemini_notebook.py`.
- Output columns are model-prefixed (e.g., `mistral_score`, `qwen_score`, `gemini_score`).
- The fine-tuning notebooks use **10-class classification** where the target class is `class_label = score - 1` (scores 1–10 map to classes 0–9).
- The fine-tuning notebooks implement **sliding-window tokenization** for long texts, **class-weighted cross-entropy**, **label smoothing**, and **layer-wise learning rate decay (LLRD)**.
- Phase 3 testing uses **multiple API keys in parallel workers** with per-worker rate limiting (`MIN_INTERVAL = 60.0 / RPM`).
- The project is actively being developed; older variant notebooks (e.g., `gemini-prompt-labeling.ipynb` at root and in `Phase1_Prompt Engineering/`) may contain legacy logic and should be treated as reference only. Use the generator scripts or the main labeled notebooks (`Gemini_Prompt_Labeling.ipynb`, `Mistral_Prompt_Labeling.ipynb`, `Qwen3_Prompt_Labeling.ipynb`) as the canonical versions.
