# Driver-Monitoring-System


人脸检测和人脸68个关键点的检测用的都是dlib的检测器，权重也用的是dlib提供的；
睡意检测是计算面部关键点的相对位置关系与设定的阈值做对比判断的；
模型整体比较简单，码量也很少
ui是用pyqt5写的
这里提2个可以优化的点，之后闲下来可能会做一下。（~~最近大概率是没啥时间了~~）
1. 面部关键点检测：dlib的检测器比较古老，偏头的话检测效果不好。可以用WFLW的数据集自己训练自己的模型（https://wywu.github.io/projects/LAB/WFLW.html)。 用什么算法就见仁见智了。可以用提供该数据集的实验室提出的一种算法"Look at Boundary: A Boundary-Aware Face Alignment Algorithm"
2. 使用关键点检测睡意：加入时间序列
    1）比较简单的一种思路：还是先把眼睛长宽比算出来，分析正常眨眼和疲劳时眨眼时眼睛长宽比在一定时间内的序列变化，用SVM做分类替代设定帧数阈值。其他部位同理。相关论文：Real-Time Eye Blink Detection using Facial Landmarks(http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)。 他的结果是   "The proposed SVM method that uses a temporal window of the eye aspect ratio (EAR),outperforms the EAR thresholding"

    2）受最近比较火的用图卷积处理基于骨架（人体关键点）的动作识别算法(如ST-GCN)的启发，可以不处理局部的关键点，直接在瞌睡视频的数据集上跑面部关键点的模型，得到所有点的位置序列，再在这些关键点位置数据上做时空图卷积。当然这完全是我意淫的，算力支不支持跑出来，效果如何还不清楚。~~好像还没人做过，要是实验结果还行貌似可以水篇论文唉~~ 
    一个9.5小时包含正常驾驶、打哈欠、眨眼速度慢、睡着等状态的Drowsiness Detection Dataset。比较坑的是没有开源，得写小作文发邮件问他们要 (http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/index.html)。也可以试一下他们提出的算法“Driver Drowsiness Detection via a Hierarchical Temporal Deep Belief Network”

## demo
![Eyeclosed](\demo\eyeclosed.png)
![Eyeclosed](\demo\EyeclosedAndYawning.png)
![Eyeclosed](\demo\nod.png)



