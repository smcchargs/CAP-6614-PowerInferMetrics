import subprocess
import os
import re
import csv

# --- CONFIGURATION ---
EXE_PATH = r"C:\Users\smcch\OneDrive\Desktop\PowerInfer\build\bin\Release\main.exe"
MODELS_DIR = r"C:\Users\smcch\OneDrive\Desktop\Models"
PROMPTS_CSV = "prompts.csv"
OUTPUT_CSV = "speed_comparison_results.csv"
BENCHMARK_LOG = "benchmark.log"

# Focused on 7B model for this iteration
MODEL_PATH = r"ReluLLaMA-7B\llama-7b-relu.powerinfer.gguf"

def parse_gen_speed(output):
    """
    Extracts ONLY the generation speed (eval time) tokens per second.
    Uses negative lookahead to ignore the 'prompt eval' line.
    """
    # Matches the line with 'eval time' but WITHOUT the word 'prompt'
    match = re.search(r"^(?!.*prompt).*eval time\s*=\s*.*?([\d.]+)\s*tokens per second", 
                      output, re.MULTILINE | re.IGNORECASE)
    return float(match.group(1)) if match else None

# --- MAIN ENGINE ---
if not os.path.exists(PROMPTS_CSV):
    print(f"Error: {PROMPTS_CSV} not found!")
    exit()

full_model_path = os.path.join(MODELS_DIR, MODEL_PATH)
final_data = []

print(f"Starting experiment for ReluLLaMA-7B...")

with open(PROMPTS_CSV, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row: continue
        prompt_text = row[1]
        
        # We store both speeds for the same prompt in one dictionary (one row in CSV)
        prompt_results = {"Prompt_Snippet": prompt_text[:30]} 

        for budget in [0, 8]:
            mode_label = "Baseline_0GB" if budget == 0 else "PowerInfer_8GB"
            print(f"  > Testing {mode_label}: {prompt_text[:40]}...")
            
            command = [
                EXE_PATH, "-m", full_model_path, "-p", prompt_text,
                "-n", "8", "--vram-budget", str(budget), "--temp", "0.0"
            ]
            
            try:
                process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore")
                combined_output = process.stdout + "\n" + process.stderr
                
                # Extract the numeric speed
                speed = parse_gen_speed(combined_output)
                prompt_results[mode_label] = speed
                
                # Debugging log
                with open(BENCHMARK_LOG, "a", encoding="utf-8") as log:
                    log.write(f"\nModel: 7B | Budget: {budget} | Speed: {speed}\n")
                    
            except Exception as e:
                print(f"    Error: {e}")
                prompt_results[mode_label] = None

        final_data.append(prompt_results)

# --- CSV EXPORT ---
# Output format: Prompt_Snippet, Baseline_0GB, PowerInfer_8GB
if final_data:
    headers = ["Prompt_Snippet", "Baseline_0GB", "PowerInfer_8GB"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(final_data)
    
    print(f"\nSuccessfully generated {OUTPUT_CSV}")
    print("You can now open this in Excel and average the columns.")