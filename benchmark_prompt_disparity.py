import subprocess
import os
import re
import csv
import time

# --- CONFIGURATION ---
EXE_PATH = r"C:\Users\smcch\OneDrive\Desktop\PowerInfer\build\bin\Release\main.exe"
MODELS_DIR = r"C:\Users\smcch\OneDrive\Desktop\Models"
PROMPTS_CSV = "prompts_disparity.csv"
OUTPUT_CSV = "disparity_analysis_7B.csv"
BENCHMARK_LOG = "benchmark_disparity.log"

# Model selection
MODEL_PATH = r"ReluLLaMA-7B\llama-7b-relu.powerinfer.gguf"
MODEL_NAME_HEADER = "ReluLLaMA_7B_Speed_TPS"

# SAFETY SETTINGS
VRAM_BUDGET = "7.5"  # Reduced slightly to prevent OS-level OOM crashes
TIMEOUT_SEC = 120    # Stop waiting after 2 minutes if the model hangs

def parse_gen_speed(output):
    """Robust regex to extract generation speed."""
    if not output:
        return None
    match = re.search(r"^(?!.*prompt).*eval time\s*=\s*.*?([\d.]+)\s*tokens per second", 
                      output, re.MULTILINE | re.IGNORECASE)
    return float(match.group(1)) if match else None

# --- INITIALIZATION ---
if not os.path.exists(PROMPTS_CSV):
    print(f"Error: {PROMPTS_CSV} not found!")
    exit()

full_model_path = os.path.join(MODELS_DIR, MODEL_PATH)
results_list = []

print(f"Starting PowerInfer Disparity Test (VRAM Safety Mode: {VRAM_BUDGET}GB)")
print("-" * 60)

with open(PROMPTS_CSV, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row: continue
        
        prompt_text = row[0]
        category = row[1] if len(row) > 1 else "General"
        
        print(f"[{category}] Processing: {prompt_text[:40]}...")

        command = [
            EXE_PATH, "-m", full_model_path, "-p", prompt_text,
            "-n", "64", "-c", "512", 
            "--vram-budget", VRAM_BUDGET, 
            "--temp", "0.0", "--simple-io" # simple-io helps subprocess stability
        ]

        try:
            # Capture both stdout and stderr with a timeout
            process = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                encoding="utf-8", 
                errors="ignore", 
                timeout=TIMEOUT_SEC
            )
            
            combined_output = (process.stdout or "") + "\n" + (process.stderr or "")
            speed = parse_gen_speed(combined_output)

            # Verification: If speed is None, log what actually happened
            if speed is None:
                print(f"    Warning: No speed data found for this prompt. Check logs.")
                with open(BENCHMARK_LOG, "a", encoding="utf-8") as log:
                    log.write(f"\n!!! DATA MISSING FOR PROMPT: {prompt_text[:50]}\n")
                    log.write(f"STDOUT: {process.stdout[:200]}\nSTDERR: {process.stderr[:200]}\n")

            results_list.append({
                "Category": category,
                "Prompt_Snippet": prompt_text[:50].replace("\n", " ") + "...",
                MODEL_NAME_HEADER: speed if speed else 0.0
            })

            # Append full raw trace
            with open(BENCHMARK_LOG, "a", encoding="utf-8") as log:
                log.write(f"\nCATEGORY: {category} | SPEED: {speed}\n{combined_output}\n")
                log.write("-" * 40 + "\n")

        except subprocess.TimeoutExpired:
            print(f"    Error: Process timed out after {TIMEOUT_SEC}s.")
            results_list.append({"Category": category, "Prompt_Snippet": "TIMEOUT", MODEL_NAME_HEADER: 0.0})
        except Exception as e:
            print(f"    Critical Error: {e}")
            results_list.append({"Category": category, "Prompt_Snippet": "ERROR", MODEL_NAME_HEADER: 0.0})

        # COOLDOWN: Short sleep to let Windows/GPU settle
        time.sleep(2)

# --- CSV OUTPUT ---
if results_list:
    headers = ["Category", "Prompt_Snippet", MODEL_NAME_HEADER]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results_list)
    print(f"\nDone! Analysis saved to {OUTPUT_CSV}")