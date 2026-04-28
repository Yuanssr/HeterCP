import matplotlib.pyplot as plt
import numpy as np
import os

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Data
methods = ['BackAlign', 'MPDA', 'CodeFilling', 'STAMP', 'GenComm', 'Baseline', 'Foreground-aware']
ap50 = [0.961, 0.859, 0.918, 0.926, 0.950, 0.937, 0.946]
ap70 = [0.861, 0.745, 0.774, 0.766, 0.826, 0.805, 0.820]

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

# Colors matching plot_performance.py
color_ap50 = '#2B7BB9'
color_ap70 = '#87CEEB'

# Removing black edge colors and diagonal hatch
rects1 = ax.bar(x - width/2, ap50, width, label='AP@50', color=color_ap50)
rects2 = ax.bar(x + width/2, ap70, width, label='AP@70', color=color_ap70)


# Aesthetics
ax.set_ylim(0.65, 1.0)
ax.set_ylabel('Average Precision', fontsize=10)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.xaxis.grid(False)

# X-axis settings
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10, rotation=0)
for label in ax.get_xticklabels():
    if label.get_text() in ('Baseline', 'Foreground-aware'):
        label.set_fontweight('bold')

# Title (reduced padding slightly to rely on bbox anchor for legend)
ax.set_title('Performance Comparison of Different Methods', fontsize=12, fontweight='bold', pad=10)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# Shared legend: Place it high enough so it doesn't overlap the title
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout(pad=2.0)
output_path = 'performance_comparison.png' # Keeping the original output format
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved successfully to {os.path.abspath(output_path)}")
