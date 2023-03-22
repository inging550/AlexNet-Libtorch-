# include "result.h"


// 构造函数用于初始化一些参数
// 根据GitHub中的说明进行修改
Result::Result() : device_type(torch::kCUDA),img_root("F:/CCCCCProject/AlexNet/Project1/DATASET/TEST/3/3_00122.bmp"), NUM_CLASS(10) ,batch_size(16)
{
}

// 此函数目的为在训练途中更改学习率
void Result::updata_learn_rate(torch::optim::SGD &optimizer, double alpha)
{
	for (auto& pg : optimizer.param_groups())
	{
		if (pg.has_options())
		{
			auto& options = pg.options();
			static_cast<torch::optim::SGDOptions&>(pg.options()).lr() = alpha;
		}
	}
}


void Result::train()
{
	// 1、定义数据集
	auto train_dataset = dataSetClc("F:\\CCCCCProject\\AlexNet\\Project1\\DATASET\\TRAIN", ".bmp").map(torch::data::transforms::Stack<>());   // 数据集自定义 bmp为后缀名（图片的后缀名也可以为其他比如:JPG,PNG等）
	auto test_dataset = dataSetClc("F:\\CCCCCProject\\AlexNet\\Project1\\DATASET\\TEST", ".bmp").map(torch::data::transforms::Stack<>());

	auto num_train = train_dataset.size();
	auto num_test = test_dataset.size();
	std::cout << "训练集数量为： " << num_train.value() << std::endl;
	std::cout << "测试集数量为： " << num_test.value() << std::endl;
	auto train_dataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), batch_size);   // 将数据集打包
	auto test_dataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(test_dataset), batch_size);  // batch_size = 2

	
	// 2、定义网络结构 并初始化权重参数
	AlexNet m_Alex(NUM_CLASS, true);  //NUM_CLASS为数据集类别总数
	//AlexNet* m_Alex = new AlexNet(NUM_CLASS, true);
	m_Alex->to(device_type);
	// 打印模型结构
	std::cout << "模型结构如下" << std::endl;
	for (auto& i : m_Alex->named_parameters())
	{
		std::cout << i.key() << std::endl;
	}

	// 3、定义损失函数以及优化器
	torch::optim::SGD optimizer(m_Alex->parameters(), torch::optim::SGDOptions(m_learn_rate[0]));
	torch::nn::CrossEntropyLoss loss_function;  

	// 4、开始训练 （学习率随着迭代增加而减小）
	for (int now_iter = 0; now_iter < Iter; now_iter++)
	{
		if (now_iter == 4)  // 在第四次迭代时学习率设置为m_learn_rate[1]
		{
			updata_learn_rate(optimizer, m_learn_rate[1]);
		}
		if (now_iter == 8)
		{
			updata_learn_rate(optimizer, m_learn_rate[2]);
		}
		// 训练
		m_Alex->train();  
		now_epoch = 0;
		total_loss = 0.0f;
		for (auto& batch : *train_dataLoader)  // 遍历数据集
		{
			now_epoch += 1;
			data = batch.data;  // 图像矩阵
			target = batch.target.squeeze();  // 标签
			data = data.to(torch::kF32).to(device_type).div(255.0);  // 将图像转变为张量+标准化（ div(255.0) ）+ 设置为CUDA
			target = target.to(torch::kInt64).to(device_type);   // 标签设置为CUDA

			// 下面代码可以查看图像尺寸  batch_size * channal * width * height
			//c10::IntArrayRef tsize = data.sizes();
			//int a = tsize[0];
			//int b = tsize[1];
			//int c = tsize[2];
			//int d = tsize[3];
			//std::cout << a << b << c << d << std::endl;

			// 前向传播
			prediction = m_Alex->forward(data);
			// 计算损失大小
			loss = loss_function(prediction, target);
			total_loss += loss.item<float>();
			// 将梯度归零有助于梯度下降
			optimizer.zero_grad(); 
			// 反向传播 计算梯度
			loss.backward();
			// 根据梯度更新模型参数
			optimizer.step();
			// 打印训练信息
			if (now_epoch % 5 == 0)
			{
				printf("Iter [%d/%d], Epoch [%d/%d] Loss: %.4f\n",now_iter,Iter,now_epoch, num_train.value()/batch_size, total_loss / (now_epoch + 1));
				//std::cout << "Epoch" << i << " Loss=" << total_loss / (i + 1) << std::endl;
			}

		}
		//测试
		validation_loss = 0.0f;
		m_Alex->eval();
		for (auto& batch : *test_dataLoader)
		{
			data = batch.data;  // 图像矩阵
			target = batch.target.squeeze();  // 标签
			data = data.to(torch::kF32).to(device_type).div(255.0);  // 将图像转变为张量+标准化（ div(255.0) ）+ 设置为CUDA
			target = target.to(torch::kInt64).to(device_type);   // 标签设置为CUDA
			prediction = m_Alex->forward(data);
			loss = loss_function(prediction, target);
			validation_loss += loss.item<float>();
		}
		validation_loss /= num_test.value() / batch_size;  // 计算平均loss
		(m_best_test_loss > validation_loss) && (m_best_test_loss = validation_loss); // if语句的简化用法
		printf("get best test loss %.5f", m_best_test_loss);
		torch::save(m_Alex, "AlexNet_CPP.pt");
	}
}

// 使用Python转换过来的模型文件进行预测
void Result::pred()
{
	torch::jit::script::Module m_Alex = torch::jit::load("F:/CCCCCProject/AlexNet/Project1/AlexNet.pt", device_type); // 模型参数的路径根据实际情况修改
	cv::Mat img = cv::imread(img_root);  // opencv读取图片
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);   // BGR—>RGB
	cv::resize(img, img, cv::Size(224, 224));
	// 将opencv读到的图片转成Tensor并且将BGR格式转成RGB格式
	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width   
	img_tensor = torch::unsqueeze(img_tensor,0);
	img_tensor = img_tensor.to(torch::kF32).to(device_type).div(255.0);

	//  开始预测
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(img_tensor);
	m_Alex.eval();
	auto o = m_Alex.forward(std::move(inputs));
	at::Tensor result = o.toTensor();

	// 得到预测的结果 result的size = 1 * 10 的张量
	std::cout << "网络输出的结果是" << result << std::endl;
	auto class1 = torch::max(result,1);
	// 打印预测的类别
	std::cout << "预测的结果是：" << CLASS_NAME[std::get<1>(class1).item<int>()] << std::endl;
}

// 使用C++训练得到的模型文件进行预测
void Result::pred1()
{
	AlexNet m_Alex(NUM_CLASS, false);  //NUM_CLASS为数据集类别总数
	m_Alex->to(device_type);
	torch::load(m_Alex, "AlexNet_CPP.pt");  // 模型参数的路径根据实际情况修改
	cv::Mat img = cv::imread(img_root);  // opencv读取图片
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);   // BGR—>RGB
	cv::resize(img, img, cv::Size(224, 224));
	// 将opencv读到的图片转成Tensor并且将BGR格式转成RGB格式
	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width   
	img_tensor = torch::unsqueeze(img_tensor, 0);
	img_tensor = img_tensor.to(torch::kF32).to(device_type).div(255.0);
	prediction = m_Alex->forward(img_tensor);
	std::cout << "网络输出的结果是" << prediction << std::endl;
	auto class1 = torch::max(prediction, 1);
	// 打印预测的类别
	std::cout << "预测的结果是：" << CLASS_NAME[std::get<1>(class1).item<int>()] << std::endl;
}