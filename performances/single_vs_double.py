import json
import matplotlib.pyplot as plt

# Load the JSON data from a file
with open('single_vs_double.json', 'r') as file:
    data = json.load(file)

# Prepare the data for plotting
model_names = list(data.keys())
test_loss = [model["Test Loss"] for model in data.values()]
test_mae = [model["Test MAE"] for model in data.values()]

# Plot the Test Loss and Test MAE
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Test Loss
ax.plot(model_names, test_loss, label='Test Loss', marker='o', linestyle='-', color='b')
# Plot Test MAE
ax.plot(model_names, test_mae, label='Test MAE', marker='o', linestyle='-', color='r')

# Customize the plot
ax.set_xlabel('Model')
ax.set_ylabel('Performance Metric')
ax.set_title('Model Performance Comparison')
ax.legend()

# Display the plot
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()