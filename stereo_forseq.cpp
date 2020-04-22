//
// Created by wbzhang233 on 2020/4/16.
//

#include "iostream"
#include "stereoMatch.h"

using namespace std;
using namespace MyStereoMatch;
using namespace kitti;

void help(){
    cout<<"usage: stereo_forseq <seq_path>"<<endl;
}

//int method = 1; // 0-SGBM;1-BM

int main(int argc,char* argv[])
{
    if (argc < 3)
    {
        help();
        exit(1);
    }

    string imgL_path = argv[1];
    string imgR_path = argv[2];
    int method = 0;//默认为SGBM
    if(argc==4)
         method = stoi(argv[3]);

    // 0-读取文件路径中的所有图片
    int count = 0;
//    namedWindow("disp_depth");
//    namedWindow("disp");
//    namedWindow("2-3U视差图");
//    namedWindow("2-3V视差图");
    namedWindow("2-5U视差图检测");
    namedWindow("2-5V视差图检测");
    namedWindow("2-6障碍物");
    namedWindow("2-4移除障碍后视差图");



    while(count<271){
        // 1-读取左右图片
//        char imgL_name[40];sprintf(imgL_name,"/%06dL.png",count);
//        char imgR_name[40];sprintf(imgR_name,"/%06dR.png",count);
        char img_name[40];sprintf(img_name,"/%06d.png",count);
        string imgLPath = imgL_path + img_name;
        Mat imgL = imread(imgLPath);
        string imgRPath = imgR_path + img_name;
        Mat imgR = imread(imgRPath);
        if(imgL.rows==0 || imgR.rows==0){
            break;
        }

        vector<string> img_path = {imgL_path,imgR_path};
        // 2-立体匹配计算视差
        Mat disparity;
        StereoResults stereoRes;
        switch (method){
            case 0:{
//                stereoRes = stereoSGBM(imgL,imgR,to_string(count));
                stereoRes = stereoSGBM( img_path,img_name);
                cout<<"Now is SGBM "<<endl;
                break;
            }
            case 1:{
//                stereoRes = stereoBM(imgL,imgR,to_string(count));
                stereoRes = stereoBM( img_path,img_name);
                cout<<"Now is BM "<<endl;
                break;
            }
        }
        // 2-1 对视差图进行填充
        Mat dispFilled = stereoRes.disp.clone();
        fillHoleDepth32f(dispFilled);
        dispFilled = Mat32fToGray(dispFilled);
//        cout<<"dispFilled:"<<typeToString(dispFilled.type())<<endl;
        Mat pdispFilled = grayToPesudo(dispFilled);//填充后视差图伪彩色图

//        Mat vc1;
//        vconcat(stereoRes.disp8u,dispFilled,vc1);//视差图,填充后视差图
//        imshow("disp",vc1);

        // 2-1计算UV视差图
        int mMaxDisp = 128;
        Mat udisp = cv::Mat(mMaxDisp,stereoRes.disp8u.cols,CV_16UC1);
        Mat vdisp =cv::Mat(stereoRes.disp8u.rows,mMaxDisp,CV_16UC1);

        computeUDisparity(udisp,stereoRes.disp8u);
        computeVDisparity(vdisp,stereoRes.disp8u);

        Mat udisp8u = Mat(udisp.size(),CV_8UC1);
        Mat vdisp8u = Mat(vdisp.size(),CV_8UC1);
        normalize(udisp, udisp8u, 0, 255, NORM_MINMAX, CV_8UC1);
        normalize(vdisp, vdisp8u, 0, 255, NORM_MINMAX, CV_8UC1);
        Mat rmDisp = removeObstacle(dispFilled,udisp8u);
        imshow("2-4移除障碍后视差图",rmDisp);

//        imshow("2-3U视差图",udisp8u);
//        imshow("2-3V视差图",vdisp8u);

        // 2-3 LSD线段检测
        vector<Vec4f> ulines = getObstacleLines(udisp8u,0.8);
        vector<Vec4f> vlines = getObstacleLines(vdisp8u,0.8);
        Mat udispClone = udisp8u.clone();
        Mat vdispClone = vdisp8u.clone();
        cvtColor(udispClone,udispClone,COLOR_GRAY2BGR);
        cvtColor(vdispClone,vdispClone,COLOR_GRAY2BGR);

//        cout<<"u_size:"<<ulines.size()<<endl;
//        cout<<"v_size:"<<vlines.size()<<endl;
//        printLines(ulines,"./linesResults/ulines2.txt");
//        printLines(vlines,"./linesResults/vlines2.txt");

        for (int i = 0; i < ulines.size();++i)
        {
            Point startPt(ulines[i](0),ulines[i](1));
            Point endPt(ulines[i](2),ulines[i](3));
            line(udispClone, startPt, endPt, Scalar(255,0,0));//标记像素点的类别，颜色区分
//        line(img, startPt, endPt, Scalar(255,0,0));//标记像素点的类别，颜色区分
        }
        imshow("2-5U视差图检测",udispClone);
        for (int i = 0; i < vlines.size();++i)
        {
            Point startPt(vlines[i](0),vlines[i](1));
            Point endPt(vlines[i](2),vlines[i](3));
            line(vdispClone, startPt, endPt, Scalar(255,0,0));//标记像素点的类别，颜色区分
//        line(img, startPt, endPt, Scalar(0,0,255));//标记像素点的类别，颜色区分
        }
        imshow("2-5V视差图检测",vdispClone);

        // 匹配在U V中检测到的直线
        vector<Rect2f> rects = matchLines(vlines,ulines);

        // 在原图中绘制障碍物矩形区域
        Mat imgclone = imgL.clone();
        for (int j = 0; j < rects.size(); ++j) {
            rectangle(imgclone,rects[j],Scalar(0,0,255),1,LINE_8);
        }
        imshow("2-6障碍物",imgclone);


//        // 3-计算深度图
//        DepthResults depthRes = disp2Depth(dispFilled, kitti_K, kitti_baseline);
//        Mat pdepth = grayToPesudo(depthRes.depth8u);
//
//        string save_name1 = "./depthResults/test"+ to_string(count)+".txt";
//        printMatToTxt( depthRes.depth,save_name1);
//
//
//        imshow("disp_depth",pdepth);
//        Mat vc2;
//        vconcat(pdispFilled,pdepth,vc2);//填充后视差图伪彩图,深度图伪彩色图
//        imshow("disp_depth",vc2);

        waitKey(10);
        count++;
        count = count%270;
    }

    cout<<"All images done..."<<endl;
    return 0;
}