#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
using namespace std;
using namespace cv;
using namespace cv::dnn;

float confidenceThreshold = 0.25;
string pro_dir = "/"; //项目根目录

void image_detection(string cfgFile, string weight, string clsNames, string image_path);

int main(int argc, char** argv)
{
	String cfgFile = pro_dir + "yolo.cfg";
	String weight = pro_dir + "yolov2.weights";
	string clsNames = pro_dir + "coco.names";
	string image_path = pro_dir + "1.jpg";
	image_detection(cfgFile, weight, clsNames, image_path);//图片测试
}

void image_detection(string cfgFile, string weight, string clsNames, string image_path) {
	// 加载网络模型
	dnn::Net net = readNetFromDarknet(cfgFile, weight);
	if (net.empty())
	{
		printf("Could not load net...\n");
		return;
	}
	// 加载分类信息
	vector<string> classNamesVec;
	ifstream classNamesFile(clsNames);
	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);
	}

	// 加载图像
	Mat frame = imread(image_path);
	Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
	net.setInput(inputBlob, "data");

	// 进行目标检测
	Mat detectionMat = net.forward("detection_out");
	vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;
	ostringstream ss;
	ss << "detection time: " << time << " ms";
	putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));

	// 输出结果
	for (int i = 0; i < detectionMat.rows; i++)
	{
		const int probability_index = 5;
		const int probability_size = detectionMat.cols - probability_index;
		float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
		size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
		float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
		if (confidence > confidenceThreshold)
		{
			float x = detectionMat.at<float>(i, 0);
			float y = detectionMat.at<float>(i, 1);
			float width = detectionMat.at<float>(i, 2);
			float height = detectionMat.at<float>(i, 3);
			int xLeftBottom = static_cast<int>((x - width / 2) * frame.cols);
			int yLeftBottom = static_cast<int>((y - height / 2) * frame.rows);
			int xRightTop = static_cast<int>((x + width / 2) * frame.cols);
			int yRightTop = static_cast<int>((y + height / 2) * frame.rows);
			Rect object(xLeftBottom, yLeftBottom,
				xRightTop - xLeftBottom,
				yRightTop - yLeftBottom);
			rectangle(frame, object, Scalar(0, 0, 255), 2, 8);
			if (objectClass < classNamesVec.size())
			{
				ss.str("");
				ss << confidence;
				String conf(ss.str());
				String label = String(classNamesVec[objectClass]) + ": " + conf;
				int baseLine = 0;
				Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom),
					Size(labelSize.width, labelSize.height + baseLine)),
					Scalar(255, 255, 255), -1);
				putText(frame, label, Point(xLeftBottom, yLeftBottom + labelSize.height),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			}
		}
	}
	vector<int> compression_params;
	compression_params.push_back(1);
	compression_params.push_back(100);
	imwrite("test2.jpg", frame, compression_params);
	//imshow("YOLO-Detections", frame);
	//waitKey(0);
	return;
}
