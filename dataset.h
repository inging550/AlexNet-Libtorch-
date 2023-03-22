#pragma once
# include "model.h"
# include <opencv2/opencv.hpp>
# include <fstream>
# include <string>
#include <io.h>


class dataSetClc : public torch::data::Dataset<dataSetClc> 
{
public:
	int class_index = 0;
	// 此函数目标为读取所有的图片以及对应标签并存储在list_images以及list_labels中
	// type为后缀名
	void load_data_from_folder(std::string path, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int label);	
	// 构造函数
	dataSetClc(std::string image_dir, std::string type) {
		load_data_from_folder(image_dir, std::string(type), image_paths, labels, class_index - 1);
	}
	// 重写get()方法，返回index对应的图片张量以及标签
	torch::data::Example<> get(size_t index) override;
	// 重写size()方法，返回数据集长度
	torch::optional<size_t> size() const override;
private:
	std::vector<std::string> image_paths;  // 按照文件夹的顺序存储数据集中所有的图片路径
	std::vector<int> labels;   // 按照文件夹的顺序存储图片路径对应的类别标签
};