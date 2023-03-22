#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>


class AlexNetImpl : public torch::nn::Module
{
public:
	AlexNetImpl(int NUM_CLASS, bool init_weight);  // NUM_CLASS为数据集的类别总数，init_weight为是否初始化权重参数
	torch::Tensor forward(torch::Tensor x);  // 前向传播函数
	void define_weight();  // 初始化权重参数
private:
	torch::nn::Sequential features{nullptr};  // 特征提取部分的网络结构
	torch::nn::Sequential classifier{ nullptr };   // 线性分类部分的网络结构
};
TORCH_MODULE(AlexNet);
