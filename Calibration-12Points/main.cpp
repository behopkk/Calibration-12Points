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

/*函数功能：求两条直线交点*/
/*输入：两条Vec4i类型直线*/
/*返回：Point2f类型的点*/
Point2f getCrossPoint(Vec4i LineA, Vec4i LineB)
{
	double ka, kb;
	ka = (double)(LineA[1] / (LineA[0] + 1e-6)); //求出LineA斜率
	kb = (double)(LineB[1] / (LineB[0] + 1e-6)); //求出LineB斜率

	double ma, mb;
	ma = LineA[3] - ka * LineA[2];
	mb = LineB[3] - kb * LineB[2];

	Point2f crossPoint;
	crossPoint.x = (mb - ma) / (ka - kb + 1e-6);
	crossPoint.y = (ma * kb - mb * ka) / (kb - ka + 1e-6);
	return crossPoint;
}


int main(int argc, char *argv[])
{
	Mat srcImageL = imread("E:\\dataset\\TEST-BD\\BD\\ImgsL\\T-L-1.jpeg", 0);
	Mat dstImageL = Mat::zeros(srcImageL.size(), srcImageL.type());

	Mat edges;
	// Find the edges in the image using canny detector
	Canny(srcImageL, edges, 50, 200);
	// Create a vector to store lines of the image
	vector<Vec4i> lines;
	vector<Point>linePointX, linePointY;
	// Apply Hough Transform
	HoughLinesP(edges, lines, 1, CV_PI / 180, 150, 10, 250);

	// Draw lines on the image
	double epsilon = 0.001;
	for (size_t i = 0; i < lines.size(); i++) {
		Vec4i l = lines[i];
		line(dstImageL, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 3, LINE_AA);
		if (abs((l[3] - l[1]) / (l[2] - l[0] + epsilon))>1)
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

	Vec4i fitLineX,fitLineY;
	//拟合方法采用最小二乘法
	fitLine(linePointX, fitLineX, CV_DIST_L2, 0, 0.01, 0.01);
	fitLine(linePointY, fitLineY, CV_DIST_L2, 0, 0.01, 0.01);

	Point2f resultPoint = getCrossPoint(fitLineX, fitLineY);

	cvtColor(srcImageL, srcImageL, COLOR_GRAY2BGR);
	circle(srcImageL, resultPoint, 8, Scalar(0, 0, 255), 8);

	waitKey();
	return 0;
}

