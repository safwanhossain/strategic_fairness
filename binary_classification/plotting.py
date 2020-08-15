import matplotlib.pyplot as plt

OFFSET = 0.1
MARGIN = 0.05
COLORS = ['r', 'b']
MARKER_SIZE = 7

def plot_vertical(erm_group_confusion_matrix, erm_gain_flip, erm_loss_flip, \
                            eo_group_confusion_matrix, eo_gain_flip, eo_loss_flip, \
                            dp_group_confusion_matrix, dp_gain_flip, dp_loss_flip, \
                            plt_name, _title):
    groups = erm_group_confusion_matrix.keys()
    increment = (1 - 2*OFFSET) / (4-1)
    x_vals = [OFFSET+i*increment for i in range(4)]

    _, ax = plt.subplots()

    for index, i in enumerate(x_vals):
        # Plot the TP rates
        if index == 0:
            ax.axvline(x=i-MARGIN, ymin=0, ymax=1, label="ERM", color="k")
            ax.axvline(x=i, ymin=0, ymax=1, linestyle="--", label="EO", color="k")
            ax.axvline(x=i+MARGIN, ymin=0, ymax=1, linestyle="-.", label="DP", color="k")

            for _color, group in zip(COLORS[:len(groups)], groups):
                tp = erm_group_confusion_matrix[group][0]
                plt.plot(i-MARGIN, tp, marker='o', color=_color, label=group, markersize=MARKER_SIZE)
            
            for _color, group in zip(COLORS[:len(groups)], groups):
                tp = eo_group_confusion_matrix[group][0]
                plt.plot(i, tp, marker='o', color=_color, markersize=MARKER_SIZE)
            
            for _color, group in zip(COLORS[:len(groups)], groups):
                tp = dp_group_confusion_matrix[group][0]
                plt.plot(i+MARGIN, tp, marker='o', color=_color, markersize=MARKER_SIZE)
            continue

        # Plot the TP rates
        if index == 1:
            ax.axvline(x=i-MARGIN, ymin=0, ymax=1, color="k")
            ax.axvline(x=i, ymin=0, ymax=1, linestyle="--", color="k")
            ax.axvline(x=i+MARGIN, ymin=0, ymax=1, linestyle="-.", color="k")

            for _color, group in zip(COLORS[:len(groups)], groups):
                tn = erm_group_confusion_matrix[group][1]
                plt.plot(i-MARGIN, tn, marker='o', color=_color, markersize=MARKER_SIZE)
            
            for _color, group in zip(COLORS[:len(groups)], groups):
                tn = eo_group_confusion_matrix[group][1]
                plt.plot(i, tn, marker='o', color=_color, markersize=MARKER_SIZE)
            
            for _color, group in zip(COLORS[:len(groups)], groups):
                tn = dp_group_confusion_matrix[group][1]
                plt.plot(i+MARGIN, tn, marker='o', color=_color, markersize=MARKER_SIZE)
            continue

        # Plot the Gain rates
        if index == 2:
            ax.axvline(x=i-MARGIN, ymin=0, ymax=1, color="k")
            ax.axvline(x=i, ymin=0, ymax=1, linestyle="--", color="k")
            ax.axvline(x=i+MARGIN, ymin=0, ymax=1, linestyle="-.", color="k")
            for _color, group in zip(COLORS[:len(groups)], groups):
                gain = erm_gain_flip[group]
                plt.plot(i-MARGIN, gain, marker='o', color=_color, markersize=MARKER_SIZE)
            
            for _color, group in zip(COLORS[:len(groups)], groups):
                gain = eo_gain_flip[group]
                plt.plot(i, gain, marker='o', color=_color, markersize=MARKER_SIZE)
            
            for _color, group in zip(COLORS[:len(groups)], groups):
                gain = dp_gain_flip[group]
                plt.plot(i+MARGIN, gain, marker='o', color=_color, markersize=MARKER_SIZE)
            continue

        # Plot the Loss rates
        if index == 3:
            ax.axvline(x=i-MARGIN, ymin=0, ymax=1, color="k")
            ax.axvline(x=i, ymin=0, ymax=1, linestyle="--", color="k")
            ax.axvline(x=i+MARGIN, ymin=0, ymax=1, linestyle="-.", color="k")

            for _color, group in zip(COLORS[:len(groups)], groups):
                loss = erm_loss_flip[group]
                plt.plot(i-MARGIN, loss, marker='o', color=_color, markersize=MARKER_SIZE)
            
            for _color, group in zip(COLORS[:len(groups)], groups):
                loss = eo_loss_flip[group]
                plt.plot(i, loss, marker='o', color=_color,markersize=MARKER_SIZE)
            
            for _color, group in zip(COLORS[:len(groups)], groups):
                loss = dp_loss_flip[group]
                plt.plot(i+MARGIN, loss, marker='o', color=_color, markersize=MARKER_SIZE)
            continue

    plt.xticks(x_vals, ["TP Rate", "TN Rate", "Gain Flip", "Loss Flip"])
    ax.legend()
    plt.legend()
    plt.title(_title)
    plt.savefig(plt_name)

