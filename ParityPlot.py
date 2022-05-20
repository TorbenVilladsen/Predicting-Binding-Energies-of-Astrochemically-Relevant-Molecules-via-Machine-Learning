import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager

# Do you want plot for 'mono' or 'multi'?
LayerType = 'mono'

# tfont = {'fontname':'Times New Roman'}
tfont = {'fontname':'Times'}
font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=20)

######################################################################

if LayerType == 'mono':
    data = np.load(r'GPRmono.npy', allow_pickle=True)

    labels = [e[0] for e in data]

    predictions = [e[1] for e in data]

    fig, ax1 = plt.subplots(figsize=(8, 7))
    ax1.scatter(labels, predictions, marker="o", label='Prediction', edgecolors='white', linewidth=0.5, s=65)

    mn, mx = ax1.set_ylim(200, 36500)
    # mn, mx = ax1.set_ylim(200, 17500)
    mn, mx = ax1.set_xlim(200, 36500)
    # mn, mx = ax1.set_xlim(200, 17100)
    ax1.set_ylabel('Predicted BE (K)', **tfont, fontsize=22)

    ax2 = ax1.twinx()
    ax3 = ax1.twiny()
    ax3.set_xlabel('Actual BE (eV)', **tfont, fontsize=22)
    ax1.set_xlabel('Actual BE (K)', **tfont, fontsize=22)
    ax3.set_xlim(mn * 8.62 * 10 ** (-5), mx * 8.62 * 10 ** (-5))
    ax2.set_ylim(mn * 8.62 * 10 ** (-5), mx * 8.62 * 10 ** (-5))
    ax2.set_ylabel('Predicted BE (eV)', **tfont, fontsize=22)

    label_min = np.amin(labels)
    label_max = np.amax(labels)

    Xs = [label_min * 0.9, label_max * 1.01]

    ax1.plot(Xs, Xs, color="black", label='Exact')

    ax1.lines[0].set_linestyle("--")

    ax1.tick_params(axis='x', labelsize=18, rotation=30)
    ax1.tick_params(axis='y', labelsize=18)
    ax3.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)

    ax1.legend(loc="upper left", prop=font)
    plt.text(1.7, 7100, "RMSE = 879 (K)",   **tfont, weight='bold', fontsize=24, color='0')
    # plt.text(0.8, 3600, "RMSE = 957 (K)",   **tfont, weight='bold', fontsize=24, color='0')
    plt.text(2.2, 4500, "= 0.076 (eV)",      **tfont, weight='bold', fontsize=24, color='0')
    # plt.text(1.03, 2300, "= 0.082 (eV)",      **tfont, weight='bold', fontsize=24, color='0')
    plt.text(1.7, 1900, "R$^2$",              **tfont, weight='bold', fontsize=24, color='0')
    # plt.text(0.8, 1000, "R$^2$",              **tfont, weight='bold', fontsize=24, color='0')
    plt.text(2.2, 1900, "= 0.992",           **tfont, weight='bold', fontsize=24, color='0')
    # plt.text(1.03, 1000, "= 0.946",           **tfont, weight='bold', fontsize=24, color='0')
    # plt.title("rs = 29, k = 5, n = 1", **tfont, weight='bold', fontsize=26)
    # plt.title("rs = 7, k = 5, n = 1", **tfont, weight='bold', fontsize=26)
    plt.show()
    # plt.savefig('mono_BE.pdf', bbox_inches='tight')

else:
    data = np.load(r'GPRmulti.npy', allow_pickle=True)

    labels = [e[0] for e in data]

    predictions = [e[1] for e in data]

    fig, ax1 = plt.subplots(figsize=(8, 7))
    ax1.scatter(labels, predictions, marker="o", label='Prediction', edgecolors='white', linewidth=0.5, s=65)

    ax1.set_xlabel('\n Actual BE (K)', **tfont, fontsize=22)
    mn, mx = ax1.set_ylim(600, 8100)
    mn, mx = ax1.set_xlim(600, 8100)
    ax1.set_ylabel('Predicted BE (K)', **tfont, fontsize=22)

    ax2 = ax1.twinx()
    ax3 = ax1.twiny()
    ax3.set_xlabel('Actual BE (eV)', **tfont, fontsize=22)
    ax3.set_xlim(mn * 8.62 * 10 ** (-5), mx * 8.62 * 10 ** (-5))
    ax2.set_ylim(mn * 8.62 * 10 ** (-5), mx * 8.62 * 10 ** (-5))
    ax2.set_ylabel('Predicted BE (eV)', **tfont, fontsize=22)

    label_min = np.amin(labels)
    label_max = np.amax(labels)

    Xs = [label_min * 0.9, label_max * 1]

    ax1.plot(Xs, Xs, color="black", label='Exact')

    ax1.lines[0].set_linestyle("--")

    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax3.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)

    ax1.legend(loc="best", prop=font)
    plt.text(0.398, 2000, "RMSE = 705 (K)", **tfont, weight='bold', fontsize=24)
    plt.text(0.502, 1450, "= 0.061 (eV)", **tfont, weight='bold', fontsize=24)
    plt.text(0.398, 900, "R$^2$", **tfont, weight='bold', fontsize=24)
    plt.text(0.502, 900, "= 0.778", **tfont, weight='bold', fontsize=24)
    # plt.title("rs = 21, k = 5, n = 1", **tfont, weight='bold', fontsize=26)
    plt.show()
    # plt.savefig('multi_BE.pdf', bbox_inches='tight')
