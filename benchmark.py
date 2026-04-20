import subprocess
import os
import re

exe_path = r"C:\Users\smcch\OneDrive\Desktop\PowerInfer\build\bin\Release\main.exe"
models_dir = r"C:\Users\smcch\OneDrive\Desktop\Models"

models = [
    r"ReluLLaMA-7B\llama-7b-relu.powerinfer.gguf", 
    r"ReluLLaMA-13B\llama-13b-relu.q4.powerinfer.gguf",
    r"ReluFalcon-40B\falcon-40b-relu.q4.powerinfer.gguf"
]

# Create or clear the log files
with open("benchmark.log", "w", encoding="utf-8") as f, open("results_summary.log", "w", encoding="utf-8") as s:
    s.write("PowerInfer Scaling & Sparsity Analysis\n")
    s.write("="*50 + "\n")

for model in models:
    full_model_path = os.path.join(models_dir, model)
    prompt_text = "Wrtie a poem"
    
    command = [
        exe_path,
        "-m", full_model_path,
        "-p", prompt_text,
        "-n", "128",
        "--vram-budget", "8",
        "--temp", "0.0"
    ]
    
    print(f"--- Benchmarking: {model} ---")
    
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    output_combined = result.stdout + result.stderr

    # --- DATA EXTRACTION ---
    # Extract the AI's answer (text between the prompt and the timing stats)
    # This logic assumes the answer starts after the prompt text
    answer = result.stdout.split(prompt_text)[-1].strip() if prompt_text in result.stdout else "Not found"

    # Regex for performance and sparsity metrics
    metrics = {
        "model": model,
        "gen_speed": re.search(r"eval time =.*?([\d.]+ tokens per second)", output_combined),
        "prompt_speed": re.search(r"prompt eval time =.*?([\d.]+ tokens per second)", output_combined),
        "ffn_offload": re.search(r"llm_load_gpu_split: offloaded ([\d.]+ MiB) of FFN", output_combined),
        "vram_total": re.search(r"total VRAM used: ([\d.]+ MB|[\d.]+ MiB)", output_combined),
        "sparse_threshold": re.search(r"sparse_pred_threshold = ([\d.]+)", output_combined)
    }

    # --- LOGGING TO SUMMARY ---
    with open("results_summary.log", "a", encoding="utf-8") as s:
        s.write(f"\n[MODEL]: {model}\n")
        s.write(f"  > Output: {answer}\n\n")
        s.write(f"  PERFORMANCE:\n")
        s.write(f"    - Generation Speed: {metrics['gen_speed'].group(1) if metrics['gen_speed'] else 'N/A'}\n")
        s.write(f"    - Prompt Processing: {metrics['prompt_speed'].group(1) if metrics['prompt_speed'] else 'N/A'}\n")
        s.write(f"  SPARSITY & VRAM (Hardware: RTX 5060 Ti):\n")
        s.write(f"    - FFN Neurons on GPU: {metrics['ffn_offload'].group(1) if metrics['ffn_offload'] else 'N/A'}\n")
        s.write(f"    - Total VRAM Consumption: {metrics['vram_total'].group(1) if metrics['vram_total'] else 'N/A'}\n")
        s.write(f"    - Activation Threshold: {metrics['sparse_threshold'].group(1) if metrics['sparse_threshold'] else 'N/A'}\n")
        s.write("-" * 50 + "\n")

    # Save raw data for backup
    with open("benchmark.log", "a", encoding="utf-8") as raw:
        raw.write(f"\n--- START {model} ---\n{output_combined}\n--- END ---\n")

    print(f"Done tracking {model}.")

print("\nAll models processed. Review 'results_summary.log'")