import json
import matplotlib.pyplot as plt
import numpy as np

# Load data from 'kernels.json'
file_name = 'kernels.json'
with open(file_name, 'r') as file:
    data = json.load(file)

# Extract data
models = list(data.keys())
test_loss = [data[model]["Test Loss"] for model in models]
test_mae = [data[model]["Test MAE"] for model in models]
test_accuracy = [data[model]["Test Accuracy"] for model in models]

# Prepare bar chart parameters
x = np.arange(len(models))  # X-axis positions for models
bar_width = 0.25  # Width of each bar

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each metric
ax.bar(x - bar_width, test_loss, bar_width, label='Test Loss', color='skyblue')
ax.bar(x, test_mae, bar_width, label='Test MAE', color='orange')
ax.bar(x + bar_width, test_accuracy, bar_width, label='Test Accuracy', color='green')

# Customize the chart
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Values', fontsize=12)
ax.set_title('Comparison of Test Metrics Across Models', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend()

# Add a grid for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()