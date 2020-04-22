//
// Created by wbzhang233 on 2020/4/2.
// https://www.cnblogs.com/riddick/p/8486223.html
//
//

#ifndef STEREOMATCH_STEREOMATCH_H
#define STEREOMATCH_STEREOMATCH_H


#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>
#include "Eigen/Dense"
#include "cstdlib"
#include "unordered_set"
#include <map>
#include <fstream>
#include "opencv4/opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>

//#define PI 3.141592653

using namespace std;
using namespace cv;
using namespace Eigen;

struct StereoResults{
    Mat disp; //类型为CV_32F
    Mat disp8u; //类型为CV_8UC1
    // Mat depth32f; //深度图
};

struct DepthResults{
    Mat depth; //深度图CV_32FC1
    Mat depth8u; //深度图灰度图CV_8UC1
    Mat pdepth;//伪彩色图
};

struct ObstacleInfo{
    vector<Vec4f> line; //表示为x1,y1;x2,y2
    float meandisp; //该线段的平均视差值

};

namespace kitti{
    const float kitti_baseline = 540; // KITTI数据集的基线距离为54cm
    const float kitti_fx = 718.856;
    const float kitti_fy = 718.856;
    const float kitti_cx = 607.1928;
    const float kitti_cy = 185.2157;
    const Mat kitti_K =(Mat_<float>(3,3)
            <<kitti_fx,0,kitti_cx,0,kitti_fy,kitti_cy,0,0,1);//直接赋初始值的方法
}

namespace MyStereoMatch{
    // 1-立体匹配
    StereoResults stereoSGBM(string img_path, string img_name,string suffix = ".png");
    StereoResults stereoBM(string img_path, string img_name,string suffix = ".png");
    // 1-1 Mat输入函数
    StereoResults stereoSGBM(Mat &imgL,Mat &imgR,string img_name);
    StereoResults stereoBM(Mat &imgL,Mat &imgR,string img_name);
    // 1-2 输入左右图片路径(先左后右)和图片名
    StereoResults stereoSGBM(vector<string> img_path,string img_name,string suffix=".png");
    StereoResults stereoBM(vector<string> img_path,string img_name,string suffix=".png");

    // 2-计算UV视差图
    void computeUDisparity(Mat &UdispMap,Mat disp);
    void computeVDisparity(Mat &VdispMap,Mat disp);
    Mat removeObstacle(Mat Disparity,Mat Udisparity);
    vector<Vec4f> getObstacleLines(Mat uvdisp,double scale = 0.8,bool useRefine = false,bool useCanny = true);
    vector<Vec4f> getObstacleLines(Mat uvdisp,Mat &draw,double angle=0.0,double scale= 0.8,bool useRefine= false,bool useCanny = true);
    void htpLines(Mat img,vector<Vec4i> &linesP,Mat &draw,int thresholdP=128);
    vector<Rect2f> matchLines(vector<Vec4f> lines1,vector<Vec4f> lines2,int maxDisp=128);


    // 3-计算深度图
    void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap, cv::Mat K,float baseline=65); //基线距离默认为65mm
    DepthResults disp2Depth(Mat dispMap,Mat K,float baseline=65);

    // 3-计算

    // 4-辅助函数
    void fillHoleDepth32f(cv::Mat& depth);//孔洞填充
    Mat Mat32fToGray(Mat disp);//32FC1转8UC1
    Mat grayToPesudo(Mat &gray,ColormapTypes colormapType=COLORMAP_JET);//灰度图转伪彩图,JET越红表示灰度值越大,越蓝灰度值越小
    void printMatToTxt(Mat img,string save_name);
    void printLines(vector<Vec4f> lines,string save_name);
    double getSlope(Vec4f line);
    template <typename _T> double getTheta(_T line);
    void filterLines(vector<Vec4f> &lines,double angle,double epsilon=3.0);

}


#endif //STEREOMATCH_STEREOMATCH_H