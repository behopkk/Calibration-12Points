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

struct CenterPoint
{
	Point2f centerL;
	Point2f centerR;
};


struct HomographyStruct
{
	Mat L;
	Mat R;
};


struct Position
{
	Point2f uniformCrossPointL;
	Point2f uniformCrossPointR;
	Point2f uniformCenterPoint;
	float theta;
};

//[xw, yw, 0, 1]
float pointWorldSet[4][9] = {
	0, 4, 8, 8,   4,   0,   0, 4, 8,
	0, 0, 0, 4.5, 4.5, 4.5, 9, 9, 9,
	0, 0, 0, 0,   0,   0,   0, 0, 0,
	1, 1, 1, 1,   1,   1,   1, 1, 1
};


Point2f TransToWorldAxis(Point2f point, Mat& invH)
{
	Point2f resultPoint;

	//invH:4*3
	resultPoint.x = invH.at<float>(0, 0)*point.x + invH.at<float>(0, 1)*point.y + invH.at<float>(0, 2);
	//cout << invH.at<float>(0, 0) << "," << invH.at<float>(0, 1) << "," << invH.at<float>(0, 2) << endl;
	resultPoint.y = invH.at<float>(1, 0)*point.x + invH.at<float>(1, 1)*point.y + invH.at<float>(1, 2);

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


HomographyStruct GetInvH()
{
	char srcImagePathL[500], srcImagePathR[500];

	/*计算单相机的平移标定角点的图像像素坐标*/
	Mat srcImageL, srcImageRectL, srcImageR, srcImageRectR;
	Rect rectL, rectR;
	float pointTransSetL[3][9], pointTransSetR[3][9];//角点图像像素坐标
	for (int index = 1; index < 10; index++)
	{
		sprintf_s(srcImagePathL, "E:\\dataset\\TEST-BD\\BD\\ImgsL\\T-L-%d.jpeg", index);
		sprintf_s(srcImagePathR, "E:\\dataset\\TEST-BD\\BD\\ImgsR\\T-R-%d.jpeg", index);

		Mat srcImageL = imread(srcImagePathL, 0);
		rectL = Rect(0, 0, 3086, srcImageL.rows);
		srcImageL(rectL).copyTo(srcImageRectL);

		Mat srcImageR = imread(srcImagePathR, 0);
		rectR = Rect(0, 0, 2400, srcImageR.rows);
		srcImageR(rectR).copyTo(srcImageRectR);

		Point2f resultPointL = GetCrossPoint(srcImageRectL);
		pointTransSetL[0][index - 1] = resultPointL.x;
		pointTransSetL[1][index - 1] = resultPointL.y;
		pointTransSetL[2][index - 1] = 1;

		Point2f resultPointR = GetCrossPoint(srcImageRectR);
		pointTransSetR[0][index - 1] = resultPointR.x;
		pointTransSetR[1][index - 1] = resultPointR.y;
		pointTransSetR[2][index - 1] = 1;

		//cvtColor(srcImageR, srcImageR, COLOR_GRAY2BGR);
		//circle(srcImageR, resultPoint, 8, Scalar(0, 0, 255), 8);
		//cvtColor(srcImageL, srcImageL, COLOR_GRAY2BGR);
		//circle(srcImageL, resultPoint, 8, Scalar(0, 0, 255), 8);

		//namedWindow(srcImagePathR, 0);
		//imshow(srcImagePathR, srcImageR);
		//namedWindow(srcImagePathL, 0);
		//imshow(srcImagePathL, srcImageL);
	}


	/*单相机标定，计算单应性矩阵H*/
	//图像像素坐标以Mat形式存储
	Mat matSetL(3, 9, CV_32FC1);
	Mat matSetR(3, 9, CV_32FC1);

	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 9; ++x) {
			matSetL.at<float>(y, x) = pointTransSetL[y][x];
			matSetR.at<float>(y, x) = pointTransSetR[y][x];
		}
	}

	//计算图像像素坐标矩阵的伪逆
	Mat imatSetL, imatSetR;
	invert(matSetL, imatSetL, DECOMP_SVD);
	invert(matSetR, imatSetR, DECOMP_SVD);

	//标定点的物理坐标以Mat形式存储
	Mat matTransWorldSet(4, 9, CV_32FC1);
	//cvtColor(matTransWorldSet, matTransWorldSet, CV_BGR2GRAY);
	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 9; ++x) {
			matTransWorldSet.at<float>(y, x) = pointWorldSet[y][x];
		}
	}

	/*计算单应性矩阵*/
	HomographyStruct invH;
	invH.L = matTransWorldSet * imatSetL;
	invH.R = matTransWorldSet * imatSetR;

	return invH;
}


CenterPoint GetCenterPoint()
{
	char srcImagePathL[500], srcImagePathR[500];

	Mat srcImageL, srcImageRectL, srcImageR, srcImageRectR;
	Rect rectL, rectR;
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

		//cvtColor(srcL, srcL, COLOR_GRAY2BGR);
		//circle(srcL, resultPointL, 8, Scalar(0, 0, 255), 8);
		//cvtColor(srcR, srcR, COLOR_GRAY2BGR);
		//circle(srcR, resultPointR, 8, Scalar(0, 0, 255), 8);

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

	CenterPoint center;
	center.centerL = centerL.center;
	center.centerR = centerR.center;

	return center;
}


Position CalPosition(Mat& imageL, Mat& imageR, HomographyStruct invH, float deltaX, float deltaY)
{
	Point2f crossPointL = GetCrossPoint(imageL);
	Point2f crossPointR = GetCrossPoint(imageR);

	Point2f crossPointWorldL = TransToWorldAxis(crossPointL, invH.L);
	Point2f crossPointWorldR = TransToWorldAxis(crossPointR, invH.R);

	//统一变换至左相机下视场
	Point2f uniformCrossPointL = crossPointWorldL;
	Point2f uniformCrossPointR = Point2f(crossPointWorldR.x - deltaX, crossPointWorldR.y - deltaY);

	float pointX = (uniformCrossPointL.x + uniformCrossPointR.x) / 2;
	float pointY = (uniformCrossPointL.y + uniformCrossPointR.y) / 2;
	Point2f uniformCenterPoint = Point2f(pointX, pointY);

	float theta = atan2(uniformCrossPointR.y - uniformCrossPointL.y,
		uniformCrossPointR.x - uniformCrossPointL.x) * 180 / PI;

	Position position;
	position.uniformCrossPointL = uniformCrossPointL;
	position.uniformCrossPointR = uniformCrossPointR;
	position.uniformCenterPoint = uniformCenterPoint;
	position.theta = theta;

	return position;
}


int main(int argc, char *argv[])
{
	/*单相机标定，计算单应性矩阵*/
	HomographyStruct invH = GetInvH();

	/*计算图像像素坐标系中的左右旋转中心*/
	CenterPoint center = GetCenterPoint();

	/*校正对位*/
	//将旋转中心变换至世界坐标系下
	CircleData centerWorldL, centerWorldR;
	centerWorldL.center = TransToWorldAxis(center.centerL, invH.L);
	centerWorldR.center = TransToWorldAxis(center.centerR, invH.R);

	float deltaX = centerWorldR.center.x - centerWorldL.center.x;
	float deltaY = centerWorldR.center.y - centerWorldL.center.y;


	/*开始对位*/
	/*加载待测图像*/
	char bmImagePathL[100], bmImagePathR[100],testImagePathL[100], testImagePathR[100];
	sprintf_s(bmImagePathL, "E:\\dataset\\TEST-BD\\L\\L0.jpeg");
	sprintf_s(bmImagePathR, "E:\\dataset\\TEST-BD\\R\\R0.jpeg");
	sprintf_s(testImagePathL, "E:\\dataset\\TEST-BD\\L\\X1Y1.jpeg");
	sprintf_s(testImagePathR, "E:\\dataset\\TEST-BD\\R\\X1Y1.jpeg");
	

	Mat bmImageL = imread(bmImagePathL, 0);
	Mat bmImageR = imread(bmImagePathR, 0);

	Position bmPosition = CalPosition(bmImageL, bmImageR, invH, deltaX, deltaY);

	cout << "bmTheta is: " << bmPosition.theta << "°\n" << endl;

	//计算待测图像中点的世界坐标 && 边缘倾斜角度
	Mat testIL = imread(testImagePathL, 0);
	Mat testIR = imread(testImagePathR, 0);
	Mat testImageL, testImageR;
	Rect rectTestL, rectTestR;

	rectTestL = Rect(0, 0, 3086, testIL.rows);
	testIL(rectTestL).copyTo(testImageL);

	rectTestR = Rect(0, 0, 2400, testIR.rows);
	testIR(rectTestR).copyTo(testImageR);

	Position testPosition = CalPosition(testImageL, testImageR, invH, deltaX, deltaY);

	cout << "testTheta is: " << testPosition.theta <<"°\n"<< endl;


	cout << "Current's ΔX is:" << -testPosition.uniformCenterPoint.x + bmPosition.uniformCenterPoint.x << endl;
	cout << "Current's ΔY is:" << -testPosition.uniformCenterPoint.y + bmPosition.uniformCenterPoint.y << endl;
	cout << "Current's Δθ is:" << -testPosition.theta + bmPosition.theta << "\n" << endl;


	/*计算待测工件移动至基准位置需要移动的偏移量和旋转量*/
	ControlInstruction instruction;

	//计算旋转量alpha
	Point2f uniformCrossPointL = bmPosition.uniformCrossPointL;
	Point2f uniformCrossPointR = bmPosition.uniformCrossPointR;
	Point2f benchMarkPoint = bmPosition.uniformCenterPoint;

	Point2f uniformTestCrossPointL = testPosition.uniformCrossPointL;
	Point2f uniformTestCrossPointR = testPosition.uniformCrossPointR;
	Point2f uniformTestCenterPoint = testPosition.uniformCenterPoint;

	float A = uniformTestCrossPointR.x - uniformTestCrossPointL.x;
	float B = uniformTestCrossPointR.y - uniformTestCrossPointL.y;
	float k = tan((uniformCrossPointR.y - uniformCrossPointL.y) / (uniformCrossPointR.x - uniformCrossPointL.x));
	instruction.commandTheta = atan((k*A - B) / (A + k * B));
	cout << "Control instruction in Theta-axis is: " << instruction.commandTheta << endl;

	//计算偏移量
	float uTCRotatePointLX = cos(instruction.commandTheta)*(uniformTestCrossPointL.x - uniformTestCenterPoint.x) -
		sin(instruction.commandTheta)*(uniformTestCrossPointL.y - uniformTestCenterPoint.y) + uniformTestCenterPoint.x;
	float uTCRotatePointLY = sin(instruction.commandTheta)*(uniformTestCrossPointL.x - uniformTestCenterPoint.x) +
		cos(instruction.commandTheta)*(uniformTestCrossPointL.y - uniformTestCenterPoint.y) + uniformTestCenterPoint.y;
	float uTCRotatePointRX = cos(instruction.commandTheta)*(uniformTestCrossPointR.x - uniformTestCenterPoint.x) -
		sin(instruction.commandTheta)*(uniformTestCrossPointR.y - uniformTestCenterPoint.y) + uniformTestCenterPoint.x;
	float uTCRotatePointRY = sin(instruction.commandTheta)*(uniformTestCrossPointR.x - uniformTestCenterPoint.x) +
		cos(instruction.commandTheta)*(uniformTestCrossPointR.y - uniformTestCenterPoint.y) + uniformTestCenterPoint.y;

	//旋转校正之后的中心点坐标
	Point2f testRotatePoint = Point2f((uTCRotatePointLX + uTCRotatePointRX) / 2, (uTCRotatePointLY + uTCRotatePointRY) / 2);

	instruction.commandX = testRotatePoint.x - benchMarkPoint.x;
	instruction.commandY = testRotatePoint.y - benchMarkPoint.y;
	cout << "Control instruction in X-axis is: " << instruction.commandX << endl;
	cout << "Control instruction in Y-axis is: " << instruction.commandY << "\n" << endl;

	float testRotateTheta = atan2(uTCRotatePointRY - uTCRotatePointLY,
		uTCRotatePointRX - uTCRotatePointLX) * 180 / PI;
	cout << "After Rotation 's Δθ is: " << testRotateTheta - bmPosition.theta<< "°" << endl;


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