import numpy as np
import matplotlib.pyplot as plt

# Example data
labels = ['EM', 'F1']
direct = [46.85, 35.27]
rewrt = [49.12, 37.74]
extract = [48.39, 37.29]
rewrt_extract_rag = [52.39, 40.33]

x = np.arange(len(labels))  # Label locations
width = 0.2  # Width of the bars

# Create the plot
fig, ax = plt.subplots()

# Plotting each set of data with color gradients of green
rects1 = ax.bar(x - width*1.5, direct, width, label='direct', color='#66c2a5')
rects2 = ax.bar(x - width/2, rewrt, width, label='rewrt', color='#8fdcb6')
rects3 = ax.bar(x + width/2, extract, width, label='extract', color='#b3e6c7')
rects4 = ax.bar(x + width*1.5, rewrt_extract_rag, width, label='rew_et_rag', color='#d9f0d8')

# Add labels, title, and custom x-axis tick labels
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()
