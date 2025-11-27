# FlowDevMiner

FlowDevMiner is a powerful tool designed to extract chemical reaction parameters (e.g., reaction type, reactants, products, conditions, metrics) from scientific literature (PDFs) using Large Language Models (LLMs). It supports both local deployment (privacy-focused) and online API usage (high performance).

## 🚀 Features

- **PDF to Structured Data**: Converts raw PDF papers into structured JSON data.
- **Three Operation Modes**:
  1.  **Local Unfinetuned**: Uses standard open-source models (e.g., Llama 2/3) with a robust heuristic 5-step pipeline.
  2.  **Local Finetuned**: Optimized for speed and accuracy using custom fine-tuned models with specialized prompts.
  3.  **Online API**: Leverages powerful cloud models (Qwen, Gemini) for direct, high-quality extraction.
- **Smart Processing**:
  - **Embedding Selection**: Automatically filters and selects the most relevant paragraphs using semantic search.
  - **Smart Packing**: Intelligently merges adjacent text segments to capture context (e.g., conditions appearing before results).
  - **Unit Boosting**: Prioritizes text containing experimental units (°C, min, mL/min, etc.).
  - **Impact Analysis**: Optionally extracts cause-effect relationships (Factor -> Metric -> Direction).

## 📂 Project Structure

- **`main.py`**: The main Command Line Interface (CLI) entry point. Use this to run the extraction pipeline on folders of PDFs.
- **`FlowDevMiner_Debug.ipynb`**: An interactive Jupyter Notebook for development, debugging, and testing new features. It allows step-by-step execution and visualization of intermediate results.
- **`config.yaml`**: Configuration file for model paths, API keys (env vars), and pipeline settings.
- **`core/`**: Contains the core logic:
  - `local_pipeline.py`: Implements the 5-step local extraction logic (Filter -> Abstract -> Summarize -> Overall -> Impact).
  - `processor.py`: Handles the online API logic (UnifiedTextProcessor).
  - `embedding.py`: Handles text embedding and semantic selection.
  - `text_utils.py`: PDF text extraction and cleaning.
- **`models/`**: Directory to store local GGUF model files.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/wzjeh/FlowDevMiner.git
    cd FlowDevMiner
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirement.txt
    ```

3.  **Download Local Models** (for Local mode):
    Place your `.gguf` model files (e.g., `My_Finetuned_Model.gguf`, `nous-hermes-llama2-13b.Q4_0.gguf`) in the `models/` directory.

4.  **Set up API Keys** (for Online mode):
    Export your API keys as environment variables:
    ```bash
    export QWEN_API_KEY="your_key_here"
    export GOOGLE_API_KEY="your_key_here"
    ```

## 📖 Usage

### 1. Using CLI (`main.py`)

Run the extractor on a directory of papers:

```bash
# Run with Local LLM (Default)
python main.py --input_dir data/papers --output_dir data/results --engine local

# Run with Local Finetuned Model (Force activation)
python main.py --input_dir data/papers --output_dir data/results --engine local --force-finetuned

# Run with Qwen API
python main.py --input_dir data/papers --output_dir data/results --engine qwen

# Run with Gemini API
python main.py --input_dir data/papers --output_dir data/results --engine gemini
```

### 2. Using Jupyter Notebook (`FlowDevMiner.ipynb`)

Start Jupyter Lab or Notebook:
```bash
jupyter lab FlowDevMiner_Debug.ipynb
```
This notebook is ideal for experimenting with different extraction parameters (`FCPD_TOP_N`, `FCPD_SUM_PACK`, etc.) and visualizing the output of each pipeline stage.

## ⚙️ Configuration

You can adjust pipeline behavior via `config.yaml` or environment variables:

-   `engine`: `local`, `qwen`, or `gemini`
-   `FCPD_TOP_N`: Number of paragraphs to select via embedding (Default: 15).
-   `FCPD_RUN_IMPACT`: Enable/Disable impact analysis (Default: 0/Off).
-   `FCPD_FORCE_FINETUNED`: Force the use of finetuned model logic in local mode.

## 📄 License

MIT License. See `LICENSE` for details.

