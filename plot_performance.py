import matplotlib.pyplot as plt
import numpy as np

# Data definition
methods = ['BackAlign', 'MPDA', 'CodeFilling', 'STAMP', 'GenComm']

# AP@70 data
hetero_ap70 = [0.8631, 0.8180, 0.8495, 0.8028, 0.8775]
cross_ap70 = [0.8607, 0.7451, 0.7744, 0.7661, 0.8258]
decline_ap70 = [0.0024, 0.0729, 0.0751, 0.0366, 0.0517]

x = np.arange(len(methods))
width = 0.35

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

# Create figure and axes
fig, ax = plt.subplots(figsize=(7, 5.5))

# Colors
color_hetero = '#2B7BB9'
color_cross = '#87CEEB'

def plot_bar_chart(ax, data_hetero, data_cross, declines, y_limits, title):
    # Highlight MPDA
    mpda_idx = methods.index('MPDA')
    ax.axvspan(mpda_idx - 0.5, mpda_idx + 0.5, color='lightyellow', alpha=0.15)
    
    # Draw bars
    rects1 = ax.bar(x - width/2, data_hetero, width, color=color_hetero, label='Heterogeneous Setting')
    rects2 = ax.bar(x + width/2, data_cross, width, color=color_cross, hatch='////', edgecolor='white', label='Cross-Domain Heterogeneous Setting')
    
    # Y-axis settings
    ax.set_ylim(y_limits)
    ax.set_ylabel('Average Precision', fontsize=10)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.grid(False)
    
    # X-axis settings
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, fontsize=10)
    
    # Title
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Draw arrows and text
    for i in range(len(methods)):
        # 1. 从左侧深色柱顶部拉出一条高雅的灰色细水平虚线
        ax.plot([x[i] - width/2, x[i] + width/2], 
                [data_hetero[i], data_hetero[i]], 
                color='#7f8c8d', linestyle=':', linewidth=1.2)
        
        # 2. 垂直下降的箭头
        ax.annotate('', 
                    xy=(x[i] + width/2, data_cross[i]), 
                    xytext=(x[i] + width/2, data_hetero[i]),
                    arrowprops=dict(arrowstyle='-|>', color='#7f8c8d', lw=1.0, shrinkA=0, shrinkB=0))
        
        # 3. 优雅的下降比例文本
        dec_val = declines[i]
        mid_y = (data_hetero[i] + data_cross[i]) / 2.0
        
        if methods[i] == 'BackAlign' and dec_val < 0.005:
            # 几乎没下降的项，文本悬浮在虚线上方
            ax.text(x[i] + width/2, data_hetero[i] + 0.002, 
                    f'-{dec_val*100:.2f}%', ha='center', va='bottom', 
                    color='#7f8c8d', fontsize=8, style='italic')
        else:
            # 其他下降明显的，将用更为沉稳的砖红色放置在侧边
            ax.text(x[i] + width/2 + 0.06, mid_y, 
                    f'-{dec_val*100:.1f}%', ha='left', va='center', 
                    color='#b03a2e', fontsize=9, fontweight='600')

# Plot AP@70
plot_bar_chart(ax, hetero_ap70, cross_ap70, decline_ap70, [0.70, 0.92], 'Performance Comparison (AP@70)')

# Shared legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)

plt.tight_layout(pad=2.0)
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
print("Figure saved successfully to performance_comparison.png")