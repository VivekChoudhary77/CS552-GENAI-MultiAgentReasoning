import matplotlib.pyplot as plt
import numpy as np

topics = ['Machine\nLearning', 'Neural\nNetworks', 'Transformers']
baseline_cosine = [0.342, 0.392, 1.000]
multiagent_cosine = [0.277, 1.000, 1.000]
baseline_bert = [0.867, 0.880, 1.000]
multiagent_bert = [0.860, 1.000, 1.000]

x = np.arange(len(topics))
width = 0.18

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(x - 1.5*width, baseline_cosine, width, label='Baseline - Cosine Sim.', color='#999999', edgecolor='#111111', linewidth=0.8)
ax.bar(x - 0.5*width, multiagent_cosine, width, label='Multi-Agent - Cosine Sim.', color='#111111', edgecolor='#111111', linewidth=0.8)
ax.bar(x + 0.5*width, baseline_bert, width, label='Baseline - BERTScore F1', color='#cccccc', edgecolor='#111111', linewidth=0.8)
ax.bar(x + 1.5*width, multiagent_bert, width, label='Multi-Agent - BERTScore F1', color='#444444', edgecolor='#111111', linewidth=0.8)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Distractor Quality: Baseline vs. Multi-Agent', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(topics, fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(loc='upper left', fontsize=9, frameon=True, edgecolor='#cccccc')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/vickinez_077/WPI FALL 2025/GenAI/CS552-GENAI-MultiAgentReasoning/docs/images/results.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.show()