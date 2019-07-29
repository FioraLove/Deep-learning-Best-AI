# -*- coding:utf-8 -*-
# matplotlib的三种分格方法：gridspec,subplot2grid,subplots
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# 创建一个图形窗口
plt.figure()
# gridspec.GridSpec将图像分割为3行，3列
gs = gridspec.GridSpec(3, 3)

# 使用plt.subplot来作图
# gs[0, :]表示这个图占第0行和所有列
ax6 = plt.subplot(gs[0, :])
# gs[1, :2]gs[1, :2]表示这个图占第1行和第2列前的所有列
ax7 = plt.subplot(gs[1, :2])
# gs[1:, 2]表示这个图占第1行后的所有行和第2列
ax8 = plt.subplot(gs[1:, 2])
# gs[-1, 0]表示这个图占倒数第1行和第0列
ax9 = plt.subplot(gs[-1, 0])
# gs[-1, -2]表示这个图占倒数第1行和倒数第2列.
ax10 = plt.subplot(gs[-1, -2])
