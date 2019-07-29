# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np

a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3, 3)
# cmap表示颜色图，origin='lower'代表的就是选择的原点的位置
plt.imshow(a, interpolation='nearest', cmap='bone', origin='upper')
# 添加一个shrink参数，使colorbar的长度变短为原来的92%；colorbar：增加颜色类标
plt.colorbar(shrink=.92)
# 设置x轴的刻度
plt.xticks(())
# 设置y轴刻度
plt.yticks(())
plt.show()
