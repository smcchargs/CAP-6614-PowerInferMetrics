import subprocess
import os
import re
import csv

# --- CONFIGURATION ---
EXE_PATH = r"C:\Users\smcch\OneDrive\Desktop\PowerInfer\build\bin\Release\main.exe"
MODELS_DIR = r"C:\Users\smcch\OneDrive\Desktop\Models"
OUTPUT_CSV = "powerinfer_experiment_results.csv"
BENCHMARK_LOG = "benchmark.log"
SUMMARY_LOG = "results_summary.log"

# Models to test
MODELS = [
    r"ReluLLaMA-7B\llama-7b-relu.powerinfer.gguf", 
    r"ReluLLaMA-13B\llama-13b-relu.q4.powerinfer.gguf",
    r"ReluFalcon-40B\falcon-40b-relu.q4.powerinfer.gguf"
]

# Experimental Variables
VRAM_BUDGETS = [0, 8]  # 0 for Baseline, 8 for Offloading
TRIALS = 3             # Number of repetitions per configuration
PROMPT = "Write a short poem about lightning."

# --- INITIALIZE LOGS ---
with open(BENCHMARK_LOG, "w", encoding="utf-8") as f:
    f.write("--- RAW POWERINFER OUTPUT LOG ---\n")

with open(SUMMARY_LOG, "w", encoding="utf-8") as s:
    s.write("PowerInfer Scaling & Sparsity Analysis Summary\n")
    s.write("="*60 + "\n")

def parse_output(output):
    """
    Extracts metrics using advanced regex to distinguish between 
    Prompt Evaluation and Generation (Eval) speeds.
    """
    data = {}
    
    # 1. Generation Speed (Eval Time): Use negative lookahead to ensure we don't grab 'prompt eval'
    # This looks for the line that has 'eval time' but NOT the word 'prompt'
    gen_match = re.search(r"^(?!.*prompt).*eval time\s*=\s*.*?([\d.]+\s*tokens per second)", 
                          output, re.MULTILINE | re.IGNORECASE)
    
    # 2. Prompt Processing Speed
    prompt_match = re.search(r"prompt eval time\s*=\s*.*?([\d.]+\s*tokens per second)", 
                             output, re.IGNORECASE)
    
    # 3. Hardware Metrics
    ffn_match = re.search(r"offloaded\s+([\d.]+ MiB)\s+of FFN", output, re.IGNORECASE)
    vram_match = re.search(r"total VRAM used:\s+([\d.]+\s*(?:MB|MiB))", output, re.IGNORECASE)
    threshold_match = re.search(r"sparse_pred_threshold\s*=\s*([\d.]+)", output)

    data['gen_speed'] = gen_match.group(1) if gen_match else "N/A"
    data['prompt_speed'] = prompt_match.group(1) if prompt_match else "N/A"
    data['ffn_offload'] = ffn_match.group(1) if ffn_match else "0.0 MiB"
    data['vram_total'] = vram_match.group(1) if vram_match else "N/A"
    data['threshold'] = threshold_match.group(1) if threshold_match else "N/A"
    
    return data

results_for_csv = []

# --- EXPERIMENT LOOP ---
for model_rel_path in MODELS:
    full_model_path = os.path.join(MODELS_DIR, model_rel_path)
    model_name = os.path.basename(model_rel_path)
    
    for budget in VRAM_BUDGETS:
        for trial in range(1, TRIALS + 1):
            print(f"Testing {model_name} | Budget: {budget}GB | Trial: {trial}/{TRIALS}")
            
            command = [
                EXE_PATH,
                "-m", full_model_path,
                "-p", PROMPT,
                "-n", "128",
                "--vram-budget", str(budget),
                "--temp", "0.0"
            ]
            
            try:
                # Run and capture output
                process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore")
                combined_output = process.stdout + "\n" + process.stderr
                
                # Update Raw Log
                with open(BENCHMARK_LOG, "a", encoding="utf-8") as raw:
                    raw.write(f"\n[ENTRY] Model: {model_name} | Budget: {budget} | Trial: {trial}\n")
                    raw.write(combined_output)
                    raw.write("\n" + "="*80 + "\n")
                
                # Parse data
                metrics = parse_output(combined_output)
                
                # Extract clean answer
                answer = process.stdout.split(PROMPT)[-1].strip() if PROMPT in process.stdout else "No text generated"

                # Update Summary Log
                with open(SUMMARY_LOG, "a", encoding="utf-8") as s:
                    s.write(f"\n[MODEL]: {model_name} | BUDGET: {budget}GB | TRIAL: {trial}\n")
                    s.write(f"  > Gen Speed:    {metrics['gen_speed']}\n")
                    s.write(f"  > Prompt Speed: {metrics['prompt_speed']}\n")
                    s.write(f"  > FFN Offload:  {metrics['ffn_offload']}\n")
                    s.write(f"  > VRAM Used:    {metrics['vram_total']}\n")
                    s.write(f"  > Answer:       {answer[:60]}...\n")
                
                # Prepare data for CSV
                results_for_csv.append({
                    "model": model_name,
                    "vram_budget": budget,
                    "trial": trial,
                    "gen_speed": metrics['gen_speed'],
                    "prompt_speed": metrics['prompt_speed'],
                    "ffn_offload": metrics['ffn_offload'],
                    "vram_used": metrics['vram_total']
                })

            except Exception as e:
                print(f"Error during {model_name} run: {e}")

# --- WRITE FINAL CSV ---
if results_for_csv:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_for_csv[0].keys())
        writer.writeheader()
        writer.writerows(results_for_csv)

print(f"\nExperiment Complete.")
print(f"Check '{SUMMARY_LOG}' for a quick look and '{OUTPUT_CSV}' for data analysis.")