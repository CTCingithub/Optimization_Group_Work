"""
Author: CTC_322 2310227@tongji.edu.cn
Date: 2023-11-29 08:48:11
LastEditors: CTC_322 2310227@tongji.edu.cn
LastEditTime: 2023-11-29 13:40:32
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""
"""
Author: CTC 2801320287@qq.com
Date: 2023-11-24 16:14:58
LastEditors: CTC 2801320287@qq.com
LastEditTime: 2023-11-25 14:40:16
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False


def SpearmanHeatMap(
    DATA,
    TICK,
    TITLE="Spearman Correlation Coefficient",
    TEXT_COLOR_CHANGE=-0.2,
    COLORBAR_TICKS=np.arange(-1, 1.01, 0.25),
    FONTSIZES=None,
    ROTATION=None,
    FIG_SIZE=None,
    FIG_DPI=None,
    INVERT_YAXIS=True,
    SHOW_FIG=True,
    SAVE_FIG=False,
):
    # Spearman相关系数可视化
    # ? FONTSIZES=[Text Font Size, Ticks Font Size, Title Font Size, Colorbar Tick Font Size]

    if FONTSIZES is None:
        FONTSIZES = [15, 13, 20, 12.5]
    if ROTATION is None:
        ROTATION = [90, 0]
    if FIG_SIZE is None:
        FIG_SIZE = (9, 4)
    if FIG_DPI is None:
        FIG_DPI = 500

    # 调整图片大小和分辨率
    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    ax = plt.axes()

    x_grid, y_grid = np.meshgrid(
        np.arange(1, DATA.shape[1] + 1), np.arange(1, DATA.shape[0] + 1)
    )

    PCOLOR_RESULT = ax.pcolor(x_grid, y_grid, DATA, cmap="bwr")

    for i in np.arange(1, DATA.shape[1] + 1):
        for j in np.arange(1, DATA.shape[0] + 1):
            ax.text(
                i,
                j,
                "{:.3f}".format(DATA[j - 1, i - 1]),
                color="w" if DATA[j - 1, i - 1] < TEXT_COLOR_CHANGE else "k",
                ha="center",
                va="center",
                size=FONTSIZES[0],
            )

    if INVERT_YAXIS:
        ax.invert_yaxis()

    # 设置横轴纵轴刻度
    ax.set_xticks(np.arange(1, DATA.shape[1] + 1))
    ax.set_xticklabels(labels=TICK, rotation=ROTATION[0], fontsize=FONTSIZES[1])
    ax.set_yticks(np.arange(1, DATA.shape[0] + 1))
    ax.set_yticklabels(labels=TICK, rotation=ROTATION[1], fontsize=FONTSIZES[1])

    # 设置标题
    ax.set_title(label=TITLE, fontsize=FONTSIZES[2])

    # 设置colorbar
    COLORBAR = fig.colorbar(PCOLOR_RESULT, ticks=COLORBAR_TICKS)
    COLORBAR.ax.tick_params(labelsize=FONTSIZES[3])

    if SHOW_FIG:
        fig.show()

    if SAVE_FIG:
        plt.savefig("Images/00SpearmanCoeff.jpg")


def LossEpochPlot(
    LOSS_HISTORY,
    TITLE=None,
    LABELS=None,
    FONTSIZES=None,
    LEGEND=None,
    FIG_SIZE=None,
    FIG_DPI=None,
    LOG_YAXIS=True,
    SHOW_FIG=True,
    SAVE_FIG=False,
):
    # Plot loss-epoch diagram
    # ? FONTSIZES=[Axis Font Size, Label Font Size, Legend Font Size, Title Font Size]
    # ? LOSS_HISTORY=(LOSS_HISTORY_TRAIN,LOSS_HISTORY_TEST)

    if TITLE is None:
        TITLE = "Loss-Epoch Diagram"
    if LABELS is None:
        LABELS = ["Epoch", "Loss"]
    if FONTSIZES is None:
        FONTSIZES = [10, 13, 13, 15]
    if LEGEND is None:
        LEGEND = ["Train Set", "Test Set"]
    if FIG_SIZE is None:
        FIG_SIZE = (4, 3)
    if FIG_DPI is None:
        FIG_DPI = 500

    # 调整图片大小和分辨率
    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    ax = plt.axes()

    if LOG_YAXIS:
        ax.set_yscale("log")

    LOSS_HISTORY_TRAIN, LOSS_HISTORY_TEST = LOSS_HISTORY
    TIME = np.arange(0, len(LOSS_HISTORY_TRAIN))

    ax.plot(TIME, LOSS_HISTORY_TRAIN)
    ax.plot(TIME, LOSS_HISTORY_TEST)
    plt.xticks(fontsize=FONTSIZES[0])
    plt.yticks(fontsize=FONTSIZES[0])
    ax.set_xlabel(LABELS[0], fontsize=FONTSIZES[1])
    ax.set_ylabel(LABELS[1], fontsize=FONTSIZES[1])
    ax.legend(LEGEND, fontsize=FONTSIZES[2])
    ax.set_title(TITLE, fontsize=FONTSIZES[3])

    if SHOW_FIG:
        fig.show()
    if SAVE_FIG:
        plt.savefig("Images/02AE_Loss_Epoch.jpg")
