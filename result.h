#pragma once
# include "dataset.h"
#include <opencv2/core/core.hpp>
#include <torch/script.h>
class Result
{
private:
	torch::DeviceType device_type;
	int batch_size;  // 训练的batch_size
	int NUM_CLASS;  // 类别总数
	const char* img_root;  // 待预测图片路径
	int Iter = 10;   // 训练的总迭代次数
	float m_best_test_loss = 0.0f;
	double m_learn_rate[3] = { 0.001,0.0001,0.00001 };
	const char* CLASS_NAME[10] = { "数字0","数字1","数字2","数字3","数字4","数字5","数字6","数字7","数字8","数字9" };  // 根据自己数据集进行更改，编写顺序按照训练集中文件夹名称自上而下的顺序
public:
	Result();
	void train();
	void pred();
	void pred1();
	void updata_learn_rate(torch::optim::SGD &optimizer, double alpha);
private:
	// 定义训练时的一些参数
	int now_epoch;
	float total_loss;
	at::Tensor data;  // 图像矩阵
	at::Tensor target;  // 标签值
	torch::Tensor prediction;  // 神经网络输出值
	torch::Tensor loss; // 损失值
	// 测试时的一些参数
	float validation_loss;
};
