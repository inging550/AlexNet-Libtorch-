# Libtorch（C++）实现AlexNet神经网络
使用的Libtorch版本为1.7.1,此代码对应的博客地址为：\

此代码对应的pybtorch（python）版本的博客地址为: \
https://blog.csdn.net/ing100/article/details/129519614 \
以及代码地址为:\
https://github.com/inging550/AlexNet-Pytorch-

博客中使用的0~9手写数据集下载地址如下：\
链接：https://pan.baidu.com/s/1-THA3tu7NHzyoS5ZIFv0Tw\ \
提取码：1234

## 训练方法
1、准备数据集（可以在上面下载，或者使用自己的数据集）数据集的摆放格式有要求，具体可参考博客\
2、result.h中修改 CLASS_NAME：各类别的名称 \
3、result.cpp中的构造函数中修改 NUM_CLASS的值（类别总数） 以及batch_size（根据自己电脑显存更改）\
4、在result.cpp文件中修改void Result::train()中的数据集路径以及数据集中图片的后缀名，对应代码27，28行 \
5、在main.py中的主函数中去掉对应注释，运行Pred类中的train()方法

## 预测方法
1、根据pt文件的来源选择执行相应的函数，在博客中有说明，然后修改函数中.pt文件的路径（模型参数的路径） \
2、在result.h中的构造函数中修改img_root（待预测的图片路径）\
3、确认CLASS_NAME（各类别的名称）以及 NUM_CLASS的值（类别总数）没有错误\
4、在main.py中的主函数中去掉对应注释，运行Pred类中的pred()或者pred1()方法
