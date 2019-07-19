# 1.import struct库
    1.1struct模块

        在Python中，『一切皆对象』，基本数据类型也不列外
        C语言的数组int a[3] = {1, 2, 4};，存储的是真正的值 
        Python的列表lyst = [1, 2, 4]，存储的是元素的指针 
    1.2 pack(),unpack()函数
        struct模块最重要的两个函数就是pack()、unpack()方法
        
        打包函数：pack(fmt, v1, v2, v3, ...)
        
        解包函数：unpack(fmt, buffer)，其中，fmt是格式字符（format的谐音），struct模块支持的格式化字符如下表
        
 # 2.matplotlib库
   2.1导入库文件  
     from matplotlib import pyplot as plt
     import numpy as np
     
   2.2 figure图像生成：
    
        # 使用import导入模块matplotlib.pyplot，并简写成plt 使用import导入模块numpy，并简写成np
        from matplotlib import pyplot as plt
        import numpy as np
        import matplotlib
        
        # 使用np.linspace定义x：范围是(-1,1);个数是50
        x = np.linspace(-1, 1, 50)
        # 函数y=x^3
        y = x ** 3
        y1 = x ** 2
        # 使用plt.figure定义一个图像窗口.num代表图像窗口编号，figsize表示窗口大小
        plt.figure(num=3, figsize=(8, 5), )
        """
        使用plt.plot画(x ,y1)曲线，曲线的颜色属性(color)为红色;
        曲线的宽度(linewidth)为1.0；曲线的类型(linestyle)为虚线. 使用plt.show显示图像
        """
        plt.plot(x, y1, color='red', linewidth='1.0', linestyle='--')
        # 使用plt.plot画(x ,y1)曲线.
        plt.plot(x, y)
        # 展示绘制图像
        plt.show()
   2.3 坐标轴：
   
       2.3.1 设置不同的名字和位置：
       
        # 使用plt.xlim设置x坐标轴范围：(-2, 2)
        plt.xlim((-2,2))
        # 使用plt.ylim设置x坐标轴范围：(-2, 2)
        plt.ylim((-2,2))
        # 定义x轴名称
        plt.xlabel('x轴')
        # 定义y轴名称
        plt.ylabel('y轴')
        new_ticks = np.linspace(-2, 2, 5)
        print(new_ticks)
        # 使用plt.xticks设置x轴刻度：范围是(-2,2);个数是8.
        plt.xticks(new_ticks)
        # 使用plt.yticks设置y轴刻度以及名称：刻度为[-2, -1.8, -1, 1.22, 3]；
        # 对应刻度的名称为[‘really bad’,’bad’,’normal’,’good’, ‘really good’]
        plt.yticks([-2, -1.8, -1, 1.22, 2],[r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
        
        2.3.2 坐标轴移至中心：
            """
            plt.gca获取当前坐标轴信息. 使用.spines设置边框；使用.set_color设置边框颜色
            """
            # 坐标轴中移
            ax = plt.gca()
            # 隐藏上边和右边
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            # 移动另外两个轴
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['bottom'].set_position(('data', 0))
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].set_position(('data',0))
        
    2.3 图例legend
        # set line syles
        l1, = plt.plot(x, y1, label='linear line')
        l2, = plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--', label='square line')
        # egend将要显示的信息来自于上面代码中的 label(表示图例的名称). 所以我们只需要简单写下一下代码, plt 就能自动的为我们添加图例.
        plt.legend(loc='upper right')

    2.4 annotate标注
        # 标注具体某点
        x0 = 0.5
        y0 = x0**2
        plt.plot([x0, x0,], [0, y0,], 'k--', linewidth=2.5)
        # set dot styles
        plt.scatter([x0, ], [y0, ], s=50, color='b')

        # 注释annotate，对(x0,y0)这个点进行标注
        plt.annotate(r'$x**2=%s$' % x0, xy=(x0, y0), xycoords='data', xytext=(+10, -10),
                     # xytext=(+10, -10) 和 textcoords='offset points' 对于标注位置的描述 和 xy 偏差值
                     textcoords='offset points', fontsize=10,
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

     
 
 
