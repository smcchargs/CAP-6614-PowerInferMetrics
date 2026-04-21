import torch
import os
import csv

def analyze_sparsity_gradient(activation_dir, output_csv, num_layers=32, thresholds=[0.8, 0.5, 0.2, 0.1, 0.05, 0.01]):
    """
    Iterates through all layers, counts the number of hot neurons at various thresholds,
    prints the formatted table to the console, and exports the data to a CSV file.
    """
    print(f"Analyzing General Profile: {activation_dir}")
    print(f"Saving results to: {output_csv}\n")
    
    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # 1. Prepare and print the header row
        console_header = f"{'Layer':<7} |"
        csv_header = ['Layer'] # First column for the CSV
        
        for t in thresholds:
            console_header += f" >{t:<4} |"
            csv_header.append(f">{t}") # Add each threshold as a column header
            
        print(console_header)
        print("-" * len(console_header))
        writer.writerow(csv_header) # Write headers to CSV
        
        # 2. Process each layer
        for layer_id in range(num_layers):
            file_path = os.path.join(activation_dir, f"activation_{layer_id}.pt")
            
            if not os.path.exists(file_path):
                print(f"Layer {layer_id:<2} | File not found")
                writer.writerow([layer_id] + ["File not found"] * len(thresholds))
                continue
                
            # Load and normalize the tensor
            layer_tensor = torch.load(file_path, map_location="cpu", weights_only=False)
            if not isinstance(layer_tensor, torch.Tensor):
                continue
                
            frequencies = layer_tensor.squeeze()
            max_val = frequencies.max().item()
            
            if max_val > 1.0:
                frequencies = frequencies / max_val
                
            # Calculate hot neurons for each threshold
            console_row = f"Layer {layer_id:<2} |"
            csv_row = [layer_id] # Start the CSV row with the layer ID
            
            for t in thresholds:
                hot_count = (frequencies > t).sum().item()
                console_row += f" {hot_count:<5} |"
                csv_row.append(hot_count) # Add the count to the CSV row
                
            print(console_row)
            writer.writerow(csv_row) # Write the finished row to the CSV file

    print(f"\nAnalysis complete! Data successfully exported to:\n{os.path.abspath(output_csv)}")

# ==========================================
# CONFIGURATION & EXECUTION
# ==========================================

# Point to your existing generic profile directory
activation_folder = r'C:\Users\smcch\OneDrive\Desktop\Models\ReluLLaMA-7B\activation'

# Define exactly where you want the CSV file to be saved
output_filename = r'C:\Users\smcch\OneDrive\Desktop\Models\ReluLLaMA-7B\sparsity_metrics.csv'

# Define the gradient to test
gradient_thresholds = [0.8, 0.5, 0.2, 0.1, 0.05, 0.01]

analyze_sparsity_gradient(
    activation_dir=activation_folder, 
    output_csv=output_filename, 
    num_layers=32, 
    thresholds=gradient_thresholds
)