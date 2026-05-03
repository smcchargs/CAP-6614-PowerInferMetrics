# CAP-6614-PowerInferMetrics
Metrics around the PowerInfer Repo

🛠 Prerequisites
Hardware
GPU: NVIDIA RTX GPU (Tested on 8GB VRAM).

You may use a variety of hardware with PowerInfer, but NVIDIA GPUs seem to be the most prominent

CPU: Multi-core processor (AVX2/AVX512 support recommended).

RAM: 16GB+ recommended for 13B+ model testing.

Dependencies
PowerInfer Binary: A compiled main.exe from the PowerInfer repository.
Windows 11 environment
Python 3.8+
PowerShell 7+

Models: ReluLLaMA or ReluFalcon models in .gguf format (PowerInfer-optimized). https://github.com/Tiiny-AI/PowerInfer

📁 Project Structure
Plaintext
├── main.exe                      # PowerInfer executable, this was built following PowerInfer's instructions: 
├── models/                       # Directory containing .gguf models, These can be downloaded from hugging face. Examples:
    ├── ReluLLaMA-7B\llama-7b-relu.powerinfer.gguf
    ├── ReluLLaMA-13B\llama-13b-relu.q4.powerinfer.gguf
    ├── ReluFalcon-40B\falcon-40b-relu.q4.powerinfer.gguf
├── benchmark_baseline.py         # This script will execute all models a single time with a 0 VRAM budget and capture TPM logs for a single run.
├── bendhmark_powerinfer.py       # This script will execute all models a single time with a 8 GB VRAM budget and capture TPM logs for a single run.
├── benchmark_both.py             # This script will execute all models 6 times total, 3 times with a VRAM budget, 3 times with 0 VRAM
├── large_prompt_benchmark.py     # This script will execute a single model on a large set of prompts both with and without a VRAM budget
├── benchmark_prompt_disparity.py # This script will prompts a single model on 15 prompts accross 3 different categories.
├── extract_hot_neuron_metrics.py # Execute on a single model and use pytorch to determine hot neuron distribution of a given model
├── prompts_all.csv               # Large list of various prompts
├── prompts.csv                   # subset of large list of prompts
├── prompts.disparity.csv         # Input: 15 prompts across 3 categories

🚀 Setup & Installation
Clone & Compile: Follow the PowerInfer GitHub instructions to compile the binary for your OS (Windows/Linux).

Environment: Ensure your NVIDIA drivers are up to date to support Unified Memory if VRAM budgets are exceeded.

Data Preparation: Place your prompts.csv in the root folder. Ensure it follows the format:
"Prompt Text","Category"

📊 Reproducing Results
To reproduce the simple findings from the Prompts experiment:

Configure the Script: Open benchmark_both.py and update the EXE_PATH and MODELS_DIR variables to match your local file system.

Execution: Run the script via terminal:

Windows
python3 benchmark_both.py
The script is pre-configured with:

VRAM Safety: Uses a 7.5GB budget to prevent OS crashes.

Context Scaling: Uses -c 512 to maximize FFN offloading.

Cooldown: Includes a 2-second sleep between runs to prevent VRAM fragmentation.

📈 Understanding the Output
The script generates .csv files To analyze:

Open the file in Excel or Google Sheets.

⚠️ Troubleshooting1
https://github.com/Tiiny-AI/PowerInfer