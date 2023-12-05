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
from utils import *

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
        plt.savefig("Images/00SpearmanCoeff.jpg", bbox_inches="tight")


def LossEpochPlot(
    LOSS_HISTORY,
    TITLE=None,
    LABELS=None,
    FONTSIZES=None,
    LEGEND=None,
    FIG_SIZE=None,
    FIG_DPI=None,
    FILE_NAME=None,
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
    if FILE_NAME is None:
        FILE_NAME = "Images/02AE_Loss_Epoch.jpg"

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
        plt.savefig(FILE_NAME, bbox_inches="tight")


def TestLossHeatMap(
    DATA,
    TITLE="Test Loss",
    LABELS=None,
    TEXT_COLOR_CHANGE=106.0,
    COLORBAR_TICKS=None,
    FONTSIZES=None,
    ROTATION=None,
    FIG_SIZE=None,
    FIG_DPI=None,
    FILE_NAME=None,
    SHOW_FIG=True,
    SAVE_FIG=False,
):
    if LABELS is None:
        LABELS = ["Hidden Layer 1", "Hidden Layer 2"]
    # Spearman相关系数可视化
    # ? FONTSIZES=[Text Font Size, Ticks Font Size,
    # ? Title Font Size, Colorbar Tick Font Size, Label Font Size]

    if FONTSIZES is None:
        FONTSIZES = [7, 13, 20, 12.5, 18]
    if ROTATION is None:
        ROTATION = [0, 0]
    if FIG_SIZE is None:
        FIG_SIZE = (10, 5)
    if FIG_DPI is None:
        FIG_DPI = 500
    if FILE_NAME is None:
        FILE_NAME = "Images/02AE_Comparison.jpg"

    LAYER_1_MIN = int(min(DATA[:, 0]))
    LAYER_1_MAX = int(max(DATA[:, 0]))
    LAYER_2_MIN = int(min(DATA[:, 1]))
    LAYER_2_MAX = int(max(DATA[:, 1]))

    LossValue = np.zeros((LAYER_1_MAX, LAYER_2_MAX))

    for i in range(DATA.shape[0]):
        LossValue[int(DATA[i, 0]) - 1, int(DATA[i, 1]) - 1] = DATA[i, 3]

    LossValue = LossValue.T
    LossValue = LossValue[1:, 1:]

    # 调整图片大小和分辨率
    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    ax = plt.axes()

    PCOLOR_RESULT = ax.pcolor(LossValue, cmap="bwr")
    for i in np.arange(1, LossValue.shape[1] + 1):
        for j in np.arange(1, LossValue.shape[0] + 1):
            ax.text(
                i - 0.5,
                j - 0.5,
                "{:.3f}".format(LossValue[j - 1, i - 1]),
                color="w" if LossValue[j - 1, i - 1] < TEXT_COLOR_CHANGE else "k",
                ha="center",
                va="center",
                size=FONTSIZES[0],
            )

    TICK_X = [f"{i}" for i in range(LAYER_1_MIN, LAYER_1_MAX + 1)]
    TICK_Y = [f"{i}" for i in range(LAYER_2_MIN, LAYER_2_MAX + 1)]

    # 设置横轴纵轴刻度
    ax.set_xticks([i - LAYER_1_MIN + 0.5 for i in range(LAYER_1_MIN, LAYER_1_MAX + 1)])
    ax.set_xticklabels(labels=TICK_X, rotation=ROTATION[0], fontsize=FONTSIZES[1])
    ax.set_xlabel(xlabel=LABELS[0], fontsize=FONTSIZES[4])
    ax.set_yticks([i - LAYER_2_MIN + 0.5 for i in range(LAYER_2_MIN, LAYER_2_MAX + 1)])
    ax.set_yticklabels(labels=TICK_Y, rotation=ROTATION[1], fontsize=FONTSIZES[1])
    ax.set_ylabel(ylabel=LABELS[1], fontsize=FONTSIZES[4])

    # 设置标题
    ax.set_title(label=TITLE, fontsize=FONTSIZES[2])

    # 设置colorbar
    COLORBAR = fig.colorbar(PCOLOR_RESULT, ticks=COLORBAR_TICKS)
    COLORBAR.ax.tick_params(labelsize=FONTSIZES[3])

    if SHOW_FIG:
        fig.show()

    if SAVE_FIG:
        plt.savefig(FILE_NAME, bbox_inches="tight")


def PlotEigenValues(
    MATRIX,
    TITLE=None,
    LABELS=None,
    FONTSIZES=None,
    FIG_SIZE=None,
    FIG_DPI=None,
    FILE_NAME=None,
    SHOW_FIG=True,
    SAVE_FIG=False,
):
    # 绘制矩阵的特征值
    # ? FONTSIZES=[Axis Font Size, Label Font Size, Title Font Size]
    if TITLE is None:
        TITLE = "Eigenvalues"
    if LABELS is None:
        LABELS = ["Re", "Im"]
    if FONTSIZES is None:
        FONTSIZES = [11, 13, 15]
    if FIG_SIZE is None:
        FIG_SIZE = (5, 5)
    if FIG_DPI is None:
        FIG_DPI = 500
    if FILE_NAME is None:
        FILE_NAME = "Images/06Eigenvalues.jpg"

    # 调整图片大小和分辨率
    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    ax = plt.axes()
    ax.grid(True, color="grey", linewidth=1, linestyle="-.")

    # 复平面上的单位圆
    t = np.linspace(0, 3 * np.pi, 3000)
    ax.plot(np.cos(t), np.sin(t), linewidth=2)

    for eigval in np.linalg.eigvals(MATRIX):
        if np.abs(eigval) < 1:
            ax.scatter(x=eigval.real, y=eigval.imag, marker="x", s=50, c="green")
        else:
            ax.scatter(x=eigval.real, y=eigval.imag, marker="x", s=50, c="red")
    ax.set_aspect("equal")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(
        labels=[f"{i}" for i in np.arange(-1, 1.5, 0.5)], fontsize=FONTSIZES[0]
    )
    ax.set_yticklabels(
        labels=[f"{i}" for i in np.arange(-1, 1.5, 0.5)], fontsize=FONTSIZES[0]
    )
    ax.set_xlabel(LABELS[0], fontsize=FONTSIZES[1])
    ax.set_ylabel(LABELS[1], fontsize=FONTSIZES[1])
    ax.set_title(TITLE, fontsize=FONTSIZES[2])

    if SHOW_FIG:
        fig.show()
    if SAVE_FIG:
        plt.savefig(FILE_NAME, bbox_inches="tight")


def ResultPlot(
    REAL_TIME_HISTORY,
    FORCAST_TIME_HISTORY,
    RMSE,
    TITLE=None,
    LABELS=None,
    T_TICKS=None,
    T_TICK_LABELS=None,
    FONTSIZES=None,
    LEGEND=None,
    FIG_SIZE=None,
    FIG_DPI=None,
    FILE_NAME=None,
    SHOW_FIG=True,
    SAVE_FIG=False,
):
    # Plot loss-epoch diagram
    # ? FONTSIZES=[Axis Font Size, Label Font Size, Legend Font Size, Title Font Size]
    # ? LOSS_HISTORY=(LOSS_HISTORY_TRAIN,LOSS_HISTORY_TEST)

    if TITLE is None:
        TITLE = "Model Accurancy"
    if LABELS is None:
        LABELS = [
            "Temp",
            "Dew Point Temp",
            "Rel Hum",
            "Wind Spd",
            "Stn Press",
        ]
    if T_TICKS is None:
        T_TICKS = [
            Time2SerialNum(YEAR=2017, MONTH=i, DAY=1, HOUR=0) for i in range(1, 13)
        ]
    if T_TICK_LABELS is None:
        T_TICK_LABELS = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sept",
            "Oct",
            "Nov",
            "Dec",
        ]
    if FONTSIZES is None:
        FONTSIZES = [8, 10, 8, 15]
    if LEGEND is None:
        LEGEND = ["Real", "Model"]
    if FIG_SIZE is None:
        FIG_SIZE = (16, 4)
    if FIG_DPI is None:
        FIG_DPI = 500
    if FILE_NAME is None:
        FILE_NAME = "Images/06FinalResult.jpg"

    fig, ax = plt.subplots(3, 2, figsize=FIG_SIZE, dpi=FIG_DPI)
    temp = np.arange(6).reshape(3, 2)
    for i in range(5):
        loc_x, loc_y = np.where(temp == i)
        loc_x = int(loc_x)
        loc_y = int(loc_y)
        ax[loc_x, loc_y].plot(
            np.arange(0, REAL_TIME_HISTORY.shape[0]), REAL_TIME_HISTORY[:, i]
        )
        ax[loc_x, loc_y].plot(
            np.arange(0, REAL_TIME_HISTORY.shape[0]), FORCAST_TIME_HISTORY[:, i]
        )
        ax[loc_x, loc_y].set_xticks(T_TICKS)
        ax[loc_x, loc_y].set_xticklabels(T_TICK_LABELS, fontsize=FONTSIZES[0])
        ax[loc_x, loc_y].set_ylabel(LABELS[i], rotation=90, fontsize=FONTSIZES[1])
        ax[loc_x, loc_y].legend(LEGEND, fontsize=FONTSIZES[2])

    RMSE_LABELS = [f"{i} RMSE" for i in LABELS]
    ax[2, 1].table(
        cellText=RMSE.reshape(1, -1),
        colLabels=RMSE_LABELS,
        loc="center",
        cellLoc="center",
    )
    ax[2, 1].axis("tight")
    ax[2, 1].axis("off")
    ax[2, 1].set_title("RMSE", fontsize=FONTSIZES[2])

    fig.suptitle(TITLE, fontsize=FONTSIZES[3])
    fig.tight_layout()
    if SHOW_FIG:
        fig.show()
    if SAVE_FIG:
        plt.savefig(FILE_NAME, bbox_inches="tight")
