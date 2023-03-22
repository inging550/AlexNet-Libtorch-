# include "dataset.h"

void dataSetClc::load_data_from_folder(std::string path, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int label)
{
	// 声明变量
	long long hFile = 0; //句柄
	struct _finddata_t fileInfo;  // _finddata_t为一个结构体
	std::string pathName;

	if ((hFile = _findfirst(pathName.assign(path).append("\\*.*").c_str(), &fileInfo)) == -1)
	{
		return;
	}
	do
	{
		const char* s = fileInfo.name;
		const char* t = type.data();

		if (fileInfo.attrib&_A_SUBDIR) //是子文件夹
		{
			//遍历子文件夹中的文件(夹)
			if (strcmp(s, ".") == 0 || strcmp(s, "..") == 0) //子文件夹目录是.或者..
				continue;
			std::string sub_path = path + "\\" + fileInfo.name;
			label++;
			load_data_from_folder(sub_path, type, list_images, list_labels, label);

		}
		else //判断是不是后缀为type文件
		{
			if (strstr(s, t))
			{
				std::string image_path = path + "\\" + fileInfo.name;
				// 将图像路径以及对应标签存进vector容器中
				list_images.push_back(image_path);
				list_labels.push_back(label);
			}
		}
	} while (_findnext(hFile, &fileInfo) == 0);
}

torch::data::Example<> dataSetClc::get(size_t index)
{
	std::string image_path = image_paths.at(index);  //vector的切片
	cv::Mat image = cv::imread(image_path);   // opencv读取图像
	cv::resize(image, image, cv::Size(224, 224)); //尺寸统一
	cv::cvtColor(image,image,cv::COLOR_BGR2RGB);   // BGR—>RGB
	int label = labels.at(index);   // 读取类别信息
	// 将opencv格式的矩阵转化为张量
	torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width
	torch::Tensor label_tensor = torch::full({ 1 }, label);
	return { img_tensor, label_tensor };
}

torch::optional<size_t> dataSetClc::size() const
{
	return image_paths.size();
};