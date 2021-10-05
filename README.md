# Driver-Monitoring-System

- [Driver-Monitoring-System](#driver-monitoring-system)
  - [简介](#简介)
  - [环境](#环境)
  - [可能的更新](#可能的更新)
  - [demo](#demo)
  - [pyinstaller 打包的坑](#pyinstaller-打包的坑)
## 简介
主要完成了通过面部表情分析和头部姿态估计检测是否瞌睡
- 人脸检测和人脸68个关键点的检测用的都是dlib的检测器，权重也用的是dlib提供的   
- 表情分析是计算一段时间内面部关键点的相对位置关系与设定的空间阈值和帧数阈值做对比判断的 
- 头部姿态估计  用上面做的2D的面部关键点匹配3D模型，求解3D和2D的对应关系(cv2.solvePnp)  参考代码 https://github.com/lincolnhard/head-pose-estimation      
- ui是用pyqt5写的  

模型整体比较简单，码量也很少    
运行时ui界面组件大小可能不正常，这可能与电脑屏幕分辨率，屏幕比例，摄像头分辨率有关   
## 环境
opencv-python    &emsp;      4.5.3.56  
numpy            &emsp;         1.20.2  
PyQt5           &emsp;       5.15.4  
pyqtgraph        &emsp;       0.12.2  
scipy            &emsp;  1.1.0  

## 可能的更新
这里提2个可以优化的点，之后闲下来可能会做一下。（~~最近大概率是没啥时间了~~）   
1. 面部关键点检测：dlib的检测器比较古老，偏头的话检测效果不好。可以用WFLW的数据集自己训练自己的模型（https://wywu.github.io/projects/LAB/WFLW.html)。 用什么算法就见仁见智了。可以用提供该数据集的实验室提出的一种算法"Look at Boundary: A Boundary-Aware Face Alignment Algorithm"
2. 使用关键点检测睡意：加入时间序列  
    1）比较简单的一种思路：还是先把眼睛长宽比算出来，分析正常眨眼和疲劳时眨眼时眼睛长宽比在一定时间内的序列变化，用SVM做分类替代设定帧数阈值。其他部位同理。相关论文：Real-Time Eye Blink Detection using Facial Landmarks(http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)。 他的结果是   "The proposed SVM method that uses a temporal window of the eye aspect ratio (EAR),outperforms the EAR thresholding"

    2）受最近比较火的用图卷积处理基于骨架（人体关键点）的动作识别算法(如ST-GCN)的启发，可以不处理局部的关键点，直接在瞌睡视频的数据集上跑面部关键点的模型，得到所有点的位置序列，再在这些关键点位置数据上做时空图卷积。当然这完全是我意淫的，算力支不支持跑出来，效果如何还不清楚。~~好像还没人做过，要是实验结果还行貌似可以水篇论文唉~~    
    
    一个9.5小时包含正常驾驶、打哈欠、眨眼速度慢、睡着等状态的Drowsiness Detection Dataset。比较坑的是没有开源，得写小作文发邮件问他们要 (http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/index.html ) 。也可以试一下他们提出的算法“Driver Drowsiness Detection via a Hierarchical Temporal Deep Belief Network”

## demo
![eyeclosed.png](https://i.loli.net/2021/10/05/cJ537yCUv1Dq8YT.png)

![EyeclosedAndYawning.png](https://i.loli.net/2021/10/05/OBpzx5Ngt3UMDq6.png)

![nod.png](https://i.loli.net/2021/10/05/4jO2oF1JbSaARuW.png)



## pyinstaller 打包的坑
1. 找不到pyqtgraph:  
   打包时加上 -p <pyqtgraph的路径>  
   虽然和其他库放在一起，但就是找不到，要加上路径才行，迷惑  
2. scipy DLL load fail：  
   打包时加上  --add-binary .../scipy/extra-dll/*;.  
   新版本scipy应该是没有extra-dll文件夹的，只能降版本了  

 
虽然解决方法比较简单，但是试错过程非常曲折啊，这2处debug耗费了五六个小时，吐了


