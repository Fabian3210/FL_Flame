import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

path = os.path.join(os.getcwd(), "experiment_setting_results.csv")

df = pd.read_csv(path)
df = df[df.server_acc.notna()]
df.server_acc = [eval(x) if isinstance(x, str) else np.nan for x in df.server_acc]
df.number = [int(x) for x in df.number]
df = df.set_index("number")

labels = ['1-FL-IID-zeroAttackers', '2-FL-IID-MP', '3-FL-IID-RL', '4-FL-IID-BD', '5-FL-IID-All', '6-Flame-IID-zeroAttackers', '7-Flame-IID-MP', '8-Flame-IID-RL', '9-Flame-IID-BD', '10-Flame-IID-All', '11-FL-nonIID-zeroAttackers', '12-FL-nonIID-MP', '13-FL-nonIID-RL',  '14-FL-nonIID-BD', '15-FL-nonIID-All', '16-Flame-NonIID-zeroAttacker', '17-Flame-NonIID-MP', '18-Flame-NonIID-RL', '19-Flame-NonIID-BD', '20-Flame-NonIID-All']
colors = [c[1] for c in mcolors.TABLEAU_COLORS.items()]
fig, ax = plt.subplots(figsize=(9, 6))
for i, acc, c in zip(df.index, df.server_acc, colors+colors):
    if i > 10:
        ax.plot(acc, color=c, linestyle="dashed", label=labels[i-1])
    else:
        ax.plot(acc, color=c, label=labels[i-1])
ax.set_ylabel("Accuracy")
ax.set_xlabel("Round")
ax.set_ylim(0, 1.05)
ax.set_yticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
ax.set_xticks([-1, 4, 9, 14, 19, 24, 29])
ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30])
ax.set_title("Comparison of all testing settings")
ax.grid()
ax.legend()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
plt.show()

# IID, FL+FLAME
fig, ax = plt.subplots(figsize=(9, 6))
for i, acc, c in zip(df.index[:10], df.server_acc[:10], colors[:5]+colors[:5]):
    if i <= 5:
        ax.plot(acc, color=c, linestyle="dashed", label=labels[i-1])
    else:
        ax.plot(acc, color=c, label=labels[i-1])
ax.set_ylabel("Accuracy")
ax.set_xlabel("Round")
ax.set_ylim(0, 1.05)
ax.set_yticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
ax.set_xticks([-1, 4, 9, 14, 19, 24, 29])
ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30])
ax.set_title("Comparison of FL and FLAME for IID data")
ax.grid()
ax.legend()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
plt.show()

# nonIID, FL+FLAME
fig, ax = plt.subplots(figsize=(9, 6))
for i, acc, c in zip(df.index[10:], df.server_acc[10:], colors[:5]+colors[:5]):
    if i <= 15:
        ax.plot(acc, color=c, linestyle="dashed", label=labels[i-1])
    else:
        ax.plot(acc, color=c, label=labels[i-1])
ax.set_ylabel("Accuracy")
ax.set_xlabel("Round")
ax.set_ylim(0, 1.05)
ax.set_yticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
ax.set_xticks([-1, 4, 9, 14, 19, 24, 29])
ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30])
ax.set_title("Comparison of FL and FLAME for nonIID data")
ax.grid()
ax.legend()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
plt.show()

# FL, IID + nonIID
fig, ax = plt.subplots(figsize=(9, 6))
for i, acc, c in zip(df.index[:5].to_list()+df.index[10:15].to_list(), df.server_acc[:5].to_list()+df.server_acc[10:15].to_list(), colors[:5]+colors[:5]):
    if i > 10:
        ax.plot(acc, color=c, linestyle="dashed", label=labels[i-1])
    else:
        ax.plot(acc, color=c, label=labels[i-1])
ax.set_ylabel("Accuracy")
ax.set_xlabel("Round")
ax.set_ylim(0, 1.05)
ax.set_yticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
ax.set_xticks([-1, 4, 9, 14, 19, 24, 29])
ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30])
ax.set_title("Comparison of IID and nonIID data for FL")
ax.grid()
ax.legend()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
plt.show()

# FLAME, IID + nonIID
fig, ax = plt.subplots(figsize=(9, 6))
for i, acc, c in zip(df.index[5:10].to_list()+df.index[15:].to_list(), df.server_acc[5:10].to_list()+df.server_acc[15:].to_list(), colors[:5]+colors[:5]):
    if i > 15:
        ax.plot(acc, color=c, linestyle="dashed", label=labels[i-1])
    else:
        ax.plot(acc, color=c, label=labels[i-1])
ax.set_ylabel("Accuracy")
ax.set_xlabel("Round")
ax.set_ylim(0, 1.05)
ax.set_yticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
ax.set_xticks([-1, 4, 9, 14, 19, 24, 29])
ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30])
ax.set_title("Comparison of IID and nonIID data for FLAME")
ax.grid()
ax.legend()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
plt.show()

