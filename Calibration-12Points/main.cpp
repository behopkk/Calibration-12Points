#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include <opencv2/highgui/highgui_c.h> 
#include <fstream>
#include <string>
#include <vector>
#include<iostream>
#include <iterator>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


struct ControlInstruction
{
	float commandX;
	float commandY;
	float commandTheta;
};


#define PI 3.14159265

struct CircleData
{
	Point2f center;
	int radius;
};


struct HomographyStruct
{
	Mat L;
	Mat R;
};


float pointTransWorldSet[2][9] = {
	0, 4, 8, 8,   4,   0,   0, 4, 8,
	0, 0, 0, 4.5, 4.5, 4.5, 9, 9, 9
};


Point2f TransToWorldAxis(Point2f point, Mat& H)
{
	Point2f resultPoint;

	resultPoint.x = H.at<float>(0, 0)*point.x + H.at<float>(0, 1)*point.y + H.at<float>(0, 2);
	resultPoint.y = H.at<float>(1, 0)*point.x + H.at<float>(1, 1)*point.y + H.at<float>(1, 2);

	return resultPoint;
}


CircleData findCircle2(Point2f pt1, Point2f pt2, Point2f pt3)
{
	float A1, A2, B1, B2, C1, C2, temp;
	A1 = pt1.x - pt2.x;
	B1 = pt1.y - pt2.y;
	C1 = (pow(pt1.x, 2) - pow(pt2.x, 2) + pow(pt1.y, 2) - pow(pt2.y, 2)) / 2;
	A2 = pt3.x - pt2.x;
	B2 = pt3.y - pt2.y;
	C2 = (pow(pt3.x, 2) - pow(pt2.x, 2) + pow(pt3.y, 2) - pow(pt2.y, 2)) / 2;

	temp = A1 * B2 - A2 * B1;

	CircleData CD;

	if (temp == 0) {
		CD.center.x = pt1.x;
		CD.center.y = pt1.y;
	}
	else {
		CD.center.x = (C1*B2 - C2 * B1) / temp;
		CD.center.y = (A1*C2 - A2 * C1) / temp;
	}

	CD.radius = sqrtf((CD.center.x - pt1.x)*(CD.center.x - pt1.x) + (CD.center.y - pt1.y)*(CD.center.y - pt1.y));
	return CD;
}

Point2f GetCrossPoint(Mat&srcImage)
{
	Mat edges;
	Mat dstImage = Mat::zeros(srcImage.size(), srcImage.type());
	// Find the edges in the image using canny detector
	Canny(srcImage, edges, 50, 200);
	// Create a vector to store lines of the image
	vector<Vec4f> lines;
	vector<Point>linePointX, linePointY;
	// Apply Hough Transform
	HoughLinesP(edges, lines, 1, CV_PI / 180, 150, 10, 250);

	// Draw lines on the image
	float epsilon = 0.001;
	for (size_t i = 0; i < lines.size(); i++) {
		Vec4f l = lines[i];
		//line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 3, LINE_AA);
		if (abs((l[3] - l[1]) / (l[2] - l[0] + epsilon)) > 5)
		{
			linePointY.push_back(Point(l[0], l[1]));
			linePointY.push_back(Point(l[2], l[3]));
		}
		else
		{
			linePointX.push_back(Point(l[0], l[1]));
			linePointX.push_back(Point(l[2], l[3]));
		}
	}

	Vec4f fitLineX, fitLineY;
	//拟合方法采用最小二乘法
	fitLine(linePointX, fitLineX, CV_DIST_HUBER, 0, 0.01, 0.01);
	fitLine(linePointY, fitLineY, CV_DIST_HUBER, 0, 0.01, 0.01);

	float ka, kb;
	ka = (float)(fitLineX[1] / (fitLineX[0] + 1e-6)); //求出LineA斜率
	kb = (float)(fitLineY[1] / (fitLineY[0] + 1e-6)); //求出LineB斜率

	float ma, mb;
	ma = fitLineX[3] - ka * fitLineX[2];
	mb = fitLineY[3] - kb * fitLineY[2];

	Point2f crossPoint;
	crossPoint.x = (mb - ma) / (ka - kb + 1e-6);
	crossPoint.y = (ma * kb - mb * ka) / (kb - ka + 1e-6);

	return crossPoint;
}




int main(int argc, char *argv[])
{
	char srcImagePathL[500], srcImagePathR[500];
	Mat srcImageL, srcImageR;

	Rect rectL, rectR;
	float pointTransSetL[3][9], pointTransSetR[3][9];
	//计算左相机的平移标定图像的角点
	for (int index = 1; index < 10; index++)
	{
		sprintf_s(srcImagePathL, "E:\\dataset\\TEST-BD\\BD\\ImgsL\\T-L-%d.jpeg", index);

		Mat srcImage = imread(srcImagePathL, 0);
		rectL = Rect(0, 0, 3086, srcImage.rows);
		srcImage(rectL).copyTo(srcImageL);

		Point2f resultPoint = GetCrossPoint(srcImageL);
		pointTransSetL[0][index - 1] = resultPoint.x;
		pointTransSetL[1][index - 1] = resultPoint.y;
		pointTransSetL[2][index - 1] = 1;

		cvtColor(srcImageL, srcImageL, COLOR_GRAY2BGR);
		circle(srcImageL, resultPoint, 8, Scalar(0, 0, 255), 8);

		//namedWindow(srcImagePathL, 0);
		//imshow(srcImagePathL, srcImageL);
	}

	//计算右相机的平移标定图像的角点
	for (int index = 1; index < 10; index++)
	{
		sprintf_s(srcImagePathR, "E:\\dataset\\TEST-BD\\BD\\ImgsR\\T-R-%d.jpeg", index);

		Mat srcImage = imread(srcImagePathR, 0);
		rectR = Rect(0, 0, 2400, srcImage.rows);
		srcImage(rectR).copyTo(srcImageR);

		Point2f resultPoint = GetCrossPoint(srcImageR);
		pointTransSetR[0][index - 1] = resultPoint.x;
		pointTransSetR[1][index - 1] = resultPoint.y;
		pointTransSetR[2][index - 1] = 1;

		cvtColor(srcImageR, srcImageR, COLOR_GRAY2BGR);
		circle(srcImageR, resultPoint, 8, Scalar(0, 0, 255), 8);

		//namedWindow(srcImagePathR, 0);
		//imshow(srcImagePathR, srcImageR);
	}

	/*标定左右相机，获取单应性矩阵H*/
	//图像像素坐标以Mat形式存储
	Mat matSetL(3, 9, CV_32FC1);
	Mat matSetR(3, 9, CV_32FC1);

	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 9; ++x) {
			matSetL.at<float>(y,x) = pointTransSetL[y][x];
			matSetR.at<float>(y,x) = pointTransSetR[y][x];
		}
	}

	//计算图像像素坐标矩阵的伪逆
	Mat imatSetL, imatSetR;
	invert(matSetL, imatSetL, DECOMP_SVD);
	invert(matSetR, imatSetR, DECOMP_SVD);

	//标定点的物理坐标以Mat形式存储
	Mat matTransWorldSet(2,9, CV_32FC1);
	//cvtColor(matTransWorldSet, matTransWorldSet, CV_BGR2GRAY);
	for (int y = 0; y < 2; ++y) {
		for (int x = 0; x < 9; ++x) {
			matTransWorldSet.at<float>(y, x) = pointTransWorldSet[y][x];
		}
	}

	/*计算左右相机下的单应性矩阵*/
	HomographyStruct H;
	H.L = matTransWorldSet * imatSetL;
	H.R = matTransWorldSet * imatSetR;


	/*计算图像像素坐标系中的左右旋转中心*/
	float pointRotateSetL[3][3], pointRotateSetR[3][3];
	for (int index = 1; index < 4; index++)
	{
		sprintf_s(srcImagePathL, "E:\\dataset\\TEST-BD\\BD\\ImgsL\\rotate-L-%d.jpeg", index);
		sprintf_s(srcImagePathR, "E:\\dataset\\TEST-BD\\BD\\ImgsR\\rotate-R-%d.jpeg", index);

		Mat srcL = imread(srcImagePathL, 0);
		Mat srcR = imread(srcImagePathR, 0);

		rectL = Rect(0, 0, 3000, srcL.rows);
		srcL(rectL).copyTo(srcImageL);
		rectR = Rect(0, 0, 3000, srcR.rows);
		srcR(rectR).copyTo(srcImageR);

		Point2f resultPointL = GetCrossPoint(srcImageL);
		Point2f resultPointR = GetCrossPoint(srcImageR);

		//circle(srcL, resultPointL, 8, Scalar(0), 8);
		//circle(srcR, resultPointR, 8, Scalar(0), 8);
		//srcL.at<uchar>(resultPointL.y, resultPointL.x) = 0;
		//srcR.at<uchar>(resultPointR.y, resultPointR.x) = 0;

		pointRotateSetL[0][index - 1] = resultPointL.x;
		pointRotateSetL[1][index - 1] = resultPointL.y;
		pointRotateSetL[2][index - 1] = 1;

		pointRotateSetR[0][index - 1] = resultPointR.x;
		pointRotateSetR[1][index - 1] = resultPointR.y;
		pointRotateSetR[2][index - 1] = 1;

		cvtColor(srcL, srcL, COLOR_GRAY2BGR);
		circle(srcL, resultPointL, 8, Scalar(0, 0, 255), 8);
		cvtColor(srcR, srcR, COLOR_GRAY2BGR);
		circle(srcR, resultPointR, 8, Scalar(0, 0, 255), 8);

		//namedWindow(srcImagePathL, 0);
		//imshow(srcImagePathL, srcL);
		//namedWindow(srcImagePathR, 0);
		//imshow(srcImagePathR, srcR);
	}

	Point2f pointL1 = Point2f(pointRotateSetL[0][0], pointRotateSetL[1][0]);
	Point2f pointL2 = Point2f(pointRotateSetL[0][1], pointRotateSetL[1][1]);
	Point2f pointL3 = Point2f(pointRotateSetL[0][2], pointRotateSetL[1][2]);
	CircleData centerL = findCircle2(pointL1, pointL2, pointL3);

	Point2f pointR1 = Point2f(pointRotateSetR[0][0], pointRotateSetR[1][0]);
	Point2f pointR2 = Point2f(pointRotateSetR[0][1], pointRotateSetR[1][1]);
	Point2f pointR3 = Point2f(pointRotateSetR[0][2], pointRotateSetR[1][2]);
	CircleData centerR = findCircle2(pointR1, pointR2, pointR3);


	/*校正对位*/
	//将旋转中心变换至世界坐标系下
	CircleData centerWorldL, centerWorldR;
	centerWorldL.center = TransToWorldAxis(centerL.center, H.L);
	centerWorldR.center = TransToWorldAxis(centerR.center, H.R);


	/*测试：将左右相机图像拼接到统一物理坐标系的一张图上*/
	//Mat leftImage = imread("E:\\dataset\\TEST-BD\\BD\\ImgsL\\rotate-L-1.jpeg", 0);
	//Mat rightImage = imread("E:\\dataset\\TEST-BD\\BD\\ImgsR\\rotate-R-1.jpeg", 0);
	//Mat castImage = Mat::zeros(Size(10000, 10000), leftImage.type());
	//Point centerLInCast = centerL.center;
	//Point centerRInCast = Point(centerR.center.x - leftImage.cols, centerR.center.y);
	//double deltaX = centerRInCast.x - centerLInCast.x;
	//double deltaY = centerRInCast.y - centerLInCast.y;
	//for (int x = 0; x < leftImage.cols; x++)
	//{
	//	for (int y = 0; y < leftImage.rows; y++)
	//	{
	//		castImage.at<uchar>(y, x) = leftImage.at<uchar>(y, x);
	//		castImage.at<uchar>(y-deltaY, x- leftImage.cols-deltaX) = rightImage.at<uchar>(y, x);
	//	}
	//}


	/*开始对位*/
	/*加载待测图像*/
	char bmImagePathL[100], bmImagePathR[100],testImagePathL[100], testImagePathR[100];
	sprintf_s(bmImagePathL, "E:\\dataset\\TEST-BD\\L\\L2.jpeg");
	sprintf_s(bmImagePathR, "E:\\dataset\\TEST-BD\\R\\R2.jpeg");
	sprintf_s(testImagePathL, "E:\\dataset\\TEST-BD\\L\\L2.jpeg");
	sprintf_s(testImagePathR, "E:\\dataset\\TEST-BD\\R\\R2.jpeg");
	//sprintf_s(testimagepathl, "e:\\dataset\\test-bd\\l\\x1y1.jpeg");
	//sprintf_s(testimagepathr, "e:\\dataset\\test-bd\\r\\x1y1.jpeg");

	Mat bmImageL = imread(bmImagePathL, 0);
	Mat bmImageR = imread(bmImagePathR, 0);
	Mat testIL = imread(testImagePathL, 0);
	Mat testIR = imread(testImagePathR, 0);

	Mat testImageL, testImageR;

	//计算基准图像中点的世界坐标 && 边缘倾斜角度
	Point2f bmCrossPointL = GetCrossPoint(bmImageL);
	Point2f bmCrossPointR = GetCrossPoint(bmImageR);
	Point2f bmCrossPointWorldL = TransToWorldAxis(bmCrossPointL, H.L);
	Point2f bmCrossPointWorldR = TransToWorldAxis(bmCrossPointR, H.R);

	//cvtColor(bmImageL, bmImageL, CV_GRAY2BGR);
	//cvtColor(bmImageR, bmImageR, CV_GRAY2BGR);
	//circle(bmImageL, bmCrossPointL, 8, Scalar(0, 0, 255), 8);
	//circle(bmImageR, bmCrossPointR, 8, Scalar(0, 0, 255), 8);

	float deltaX = centerWorldR.center.x - centerWorldL.center.x;
	float deltaY = centerWorldR.center.y - centerWorldL.center.y;

	Point2f uniformCrossPointL = bmCrossPointWorldL;
	Point2f uniformCrossPointR = Point2f(bmCrossPointWorldR.x - deltaX, bmCrossPointWorldR.y - deltaY);
	Point2f uniformCenterPoint = centerWorldL.center;

	float benchMarkPointX = (uniformCrossPointL.x + uniformCrossPointR.x) / 2;
	float benchMarkPointY = (uniformCrossPointL.y + uniformCrossPointR.y) / 2;
	Point2f benchMarkPoint = Point2f(benchMarkPointX, benchMarkPointY);

	float bmTheta = atan2(uniformCrossPointR.y - uniformCrossPointL.y,
		uniformCrossPointR.x - uniformCrossPointL.x) * 180 / PI;

	cout << "bmTheta is: " << bmTheta << "°\n" << endl;

	//计算待测图像中点的世界坐标 && 边缘倾斜角度
	Rect rectTestL, rectTestR;
	rectTestL = Rect(0, 0, 3086, testIL.rows);
	testIL(rectTestL).copyTo(testImageL);

	rectTestR = Rect(0, 0, 2400, testIR.rows);
	testIR(rectTestR).copyTo(testImageR);

	Point2f testCrossPointL = GetCrossPoint(testImageL);
	Point2f testCrossPointR = GetCrossPoint(testImageR);
	Point2f testCrossPointWorldL = TransToWorldAxis(testCrossPointL, H.L);
	Point2f testCrossPointWorldR = TransToWorldAxis(testCrossPointR, H.R);

	//cvtColor(testImageL, testImageL, CV_GRAY2BGR);
	//cvtColor(testImageR, testImageR, CV_GRAY2BGR);
	//circle(testImageL, testCrossPointL, 8, Scalar(0, 0, 255), 8);
	//circle(testImageR, testCrossPointR, 8, Scalar(0, 0, 255), 8);

	Point2f uniformTestCrossPointL = testCrossPointWorldL;
	Point2f uniformTestCrossPointR = Point2f(testCrossPointWorldR.x - deltaX, testCrossPointWorldR.y - deltaY);
	Point2f uniformTestCenterPoint = centerWorldL.center;

	float testPointX = (uniformTestCrossPointL.x + uniformTestCrossPointR.x) / 2;
	float testPointY = (uniformTestCrossPointL.y + uniformTestCrossPointR.y) / 2;
	Point2f testPoint = Point2f(testPointX, testPointY);

	float testTheta = atan2(uniformTestCrossPointR.y - uniformTestCrossPointL.y,
		uniformTestCrossPointR.x - uniformTestCrossPointL.x) * 180 / PI;


	cout << "testTheta is: " << testTheta <<"°\n"<< endl;


	cout << "Current's ΔX is:"<< testPoint.x-benchMarkPoint.x<<endl;
	cout << "Current's ΔY is:" << testPoint.y - benchMarkPoint.y << endl;
	cout << "Current's Δθ is:" << testTheta - bmTheta << "\n" << endl;


	/*计算待测工件移动至基准位置需要移动的偏移量和旋转量*/
	ControlInstruction instruction;

	//计算旋转量alpha
	float A = uniformTestCrossPointR.x - uniformTestCrossPointL.x;
	float B = uniformTestCrossPointR.y - uniformTestCrossPointL.y;
	float k = tan((uniformCrossPointR.y - uniformCrossPointL.y) / (uniformCrossPointR.x - uniformCrossPointL.x));
	instruction.commandTheta = atan((k*A - B) / (A + k * B));
	cout << "Control instruction in Theta-axis is: " << instruction.commandTheta << endl;

	//计算偏移量
	float m = uniformTestCenterPoint.x;
	float n = uniformTestCenterPoint.y;

	float uTCRotatePointLX = cos(instruction.commandTheta)*(uniformTestCrossPointL.x - m) -
		sin(instruction.commandTheta)*(uniformTestCrossPointL.y - n) + m;
	float uTCRotatePointLY = sin(instruction.commandTheta)*(uniformTestCrossPointL.x - m) +
		cos(instruction.commandTheta)*(uniformTestCrossPointL.y - n) + n;
	float uTCRotatePointRX = cos(instruction.commandTheta)*(uniformTestCrossPointR.x - m) -
		sin(instruction.commandTheta)*(uniformTestCrossPointR.y - n) + m;
	float uTCRotatePointRY = sin(instruction.commandTheta)*(uniformTestCrossPointR.x - m) +
		cos(instruction.commandTheta)*(uniformTestCrossPointR.y - n) + n;

	//旋转校正之后的中心点坐标
	Point2f testRotatePoint = Point2f((uTCRotatePointLX + uTCRotatePointRX) / 2, (uTCRotatePointLY + uTCRotatePointRY) / 2);

	instruction.commandX = testRotatePoint.x - benchMarkPoint.x;
	instruction.commandY = testRotatePoint.y - benchMarkPoint.y;
	cout << "Control instruction in X-axis is: " << instruction.commandX << endl;
	cout << "Control instruction in Y-axis is: " << instruction.commandY << "\n" << endl;

	float testRotateTheta = atan2(uTCRotatePointRY - uTCRotatePointLY,
		uTCRotatePointRX - uTCRotatePointLX) * 180 / PI;
	cout << "After Rotation 's Δθ is: " << testRotateTheta - bmTheta<< "°" << endl;


	/*对位控制指令发出以及检测对位误差*/
	

	/*对位精度补偿*/
	float deltaXEWorld = 0;
	float deltaYEWorld = 0;
	float rotateTheta = instruction.commandTheta;
	float a = uniformTestCrossPointL.x;
	float b = uniformTestCrossPointL.y;
	float c = uTCRotatePointLX;
	float d = uTCRotatePointLY;
	float m = uniformCrossPointL.x + deltaXEWorld + instruction.commandX;
	float n = uniformCrossPointL.y + deltaYEWorld + instruction.commandY;
	//补偿之后的旋转中心世界坐标
	float xW = (m + a) / 2 + (b - n)*sin(rotateTheta) / (2 * (1 - cos(rotateTheta)));
	float yW = (n + b) / 2 + (1 + cos(rotateTheta))*(m - a) / (2 * sin(rotateTheta));


	waitKey();
	return 0;
}