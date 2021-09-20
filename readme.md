# OPENCV车流量计数

某鱼上的课设

代码来自 [github](https://github.com/creotiv/object_detection_projects/tree/master/opencv_traffic_counting)  [博客](https://zhuanlan.zhihu.com/p/47341761)，原本的代码用的是背景估计的检测方法，效果不太好。将检测方法改成 `opencv `级联检测后，效果好了许多

## 计数的原理

1. 轨迹维护的思路：就是最近邻，将当前帧所有检测的点 和 历史轨迹去比较，将距离最近 且 满足距离阈值的点和轨迹匹配。距离的计算方法，根据前两次的历史位置预测当前的位置，然后和检测的位置计算距离。
2. 计数的方法：遍历所有轨迹，取最近两次的位置，上一次位置在检测区外，这一次的位置在检测区内，就计数。

## 环境

解决 `opencv VideoCapture    ` 返回为 None 的问题  [参考链接](https://blog.csdn.net/dlh_sycamore/article/details/83178394)

``` 
conda install -c menpo opencv3
pip install opencv-contrib-python
```

fork后代码链接



