import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_csv(filename):
    # Read the CSV file
    df = pd.read_csv(filename)

    # Sample 10 random rows from the DataFrame if there are enough rows
    if len(df) > 60:
        df = df.sample(60, random_state=np.random.randint(0, 10000))

    # Prepare the plot
    plt.figure(figsize=(10, 6))
    plt.title(f"Scatter Plot of {os.path.basename(filename)} - Sampled")

    # Color mapping for the groups
    colors = {1: 'blue', 2: 'red'}
    
    # Process each row
    for index, row in df.iterrows():
        group = row.iloc[0]
        values = row.iloc[1:]

        # Plot each value against its column index (1-based index for x-axis)
        plt.scatter(range(1, len(values) + 1), values, color=colors[group], 
                    label=f'Group {group}' if f'Group {group}' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Set labels and adjust legend
    plt.xlabel("Pixel location")
    plt.ylabel("Value")
    plt.legend(title="Identification")
    plt.grid(True)

    # Save the plot to a file, safely remove ".csv" from filename and ensure directory
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_directory = './pyplot'
    os.makedirs(output_directory, exist_ok=True)
    plot_filename = os.path.join(output_directory, f"{base_name}_sampled.png")
    plt.savefig(plot_filename)
    plt.close()  # Close the figure to free up memory

print("Current Working Directory:", os.getcwd())
# Filenames
filenames = ["PCA.csv", "Standardized_training.csv"]

# Plot each file
for file in filenames:
    plot_csv(file)
