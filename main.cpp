#include <iostream>
#include "stereoMatch.h"

using namespace std;
using namespace MyStereoMatch;

int main() {
    std::cout << "Hello, World!" << std::endl;

//    string img_path = "/home/wbzhang233/Code/StereoMatch/SGM-disparity/data/";
//    string img_name = "imR";
//    string suffix=".jpg";
    string img_path = "/home/wbzhang233/Dataset/stereo/";
    string img_name = "000020R";
    string suffix = ".png";

    Mat img  = imread(img_path+img_name+suffix);

    // 1-双目立体匹配
    StereoResults res = MyStereoMatch::stereoSGBM(img_path,img_name,suffix);
    Mat disp_filled = res.disp.clone(); //视差图32F
    MyStereoMatch::fillHoleDepth32f(disp_filled);
//    cout<<disp_filled.type()<<" "<<res.disp.type()<<endl;
    disp_filled = Mat32fToGray(disp_filled);//视差图填充后灰度图

    // 2-视差图孔洞填充
//    Mat vc21;
//    vconcat(res.disp8u,disp_filled,vc21);
//    namedWindow("2-1视差图孔洞填充前后");
//    imshow("2-1视差图孔洞填充前后",vc21);

//    Mat pdisp,fpdisp,vc22;
//    applyColorMap(res.disp8u,pdisp,COLORMAP_JET);
//    applyColorMap(disp_filled,fpdisp,COLORMAP_JET);
//    vconcat(pdisp,fpdisp,vc22);
//    namedWindow("2-2视差图伪彩图");
//    imshow("2-2视差图伪彩图",vc22);

    /// 2-1计算UV视差图
    int mMaxDisp = 128;
    Mat udisp = cv::Mat(mMaxDisp,res.disp8u.cols,CV_16UC1);
    Mat vdisp =cv::Mat(res.disp8u.rows,mMaxDisp,CV_16UC1);

    computeUDisparity(udisp,res.disp8u);
    computeVDisparity(vdisp,res.disp8u);

    Mat udisp8u = Mat(res.disp8u.size(),CV_8UC1);
    Mat vdisp8u = Mat(res.disp8u.size(),CV_8UC1);
    normalize(udisp, udisp8u, 0, 255, NORM_MINMAX, CV_8UC1);
    normalize(vdisp, vdisp8u, 0, 255, NORM_MINMAX, CV_8UC1);
//    Mat rmDisp = removeObstacle(disp_filled,udisp8u);


    namedWindow("2-3U视差图");
    imshow("2-3U视差图",udisp8u);
    namedWindow("2-3V视差图");
    imshow("2-3V视差图",vdisp8u);
//    namedWindow("2-4移除障碍后视差图");
//    imshow("2-4移除障碍后视差图",rmDisp);
//    imwrite("./disp/udisp.png",udisp8u);
//    imwrite("./disp/vdisp.png",vdisp8u);
//    imwrite("./disp/disp.png",res.disp8u);
//    imwrite("./disp/rmDisp.png",rmDisp);

    /// 2-2 霍夫变换
    vector<Vec2f> linesu;
    HoughLines(udisp8u, linesu, 1, CV_PI/180, 180, 1, 1 );
    vector<Vec2f> linesv;
    HoughLines(udisp8u, linesv, 1, CV_PI/180, 180, 1, 1);
    Mat udispClone1 = udisp8u.clone();
    Mat vdispClone1 = vdisp8u.clone();
    cvtColor(udispClone1,udispClone1,COLOR_GRAY2BGR);
    cvtColor(vdispClone1,vdispClone1,COLOR_GRAY2BGR);
    for( size_t i = 0; i < linesu.size(); i++ )
    {
        float rho = linesu[i][0], theta = linesu[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( udispClone1, pt1, pt2, Scalar(0,0,255), 1, LINE_AA);
    }
    for( size_t i = 0; i < linesv.size(); i++ )
    {
        float rho = linesv[i][0], theta = linesv[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( vdispClone1, pt1, pt2, Scalar(0,0,255), 1, LINE_AA);
    }

    // 2-2-2 霍夫变换2
//    vector<Vec4i> linesPu,linesPv;
//    htpLines(udisp8u,linesPu,udispClone1);
//    htpLines(vdisp8u,linesPv,vdispClone1);

//    namedWindow("U-HT检测");
//    imshow("U-HT检测",udispClone1);
//    namedWindow("V-HT检测");
//    imshow("V-HT检测",vdispClone1);

    /// 2-3 LSD线段检测
    Mat udispClone = udisp8u.clone();
    Mat vdispClone = vdisp8u.clone();
    cvtColor(udispClone,udispClone,COLOR_GRAY2BGR);
    cvtColor(vdispClone,vdispClone,COLOR_GRAY2BGR);
    vector<Vec4f> ulines = getObstacleLines(udisp8u,udispClone,0,0.8);
    vector<Vec4f> vlines = getObstacleLines(vdisp8u,vdispClone,90,0.8);

    printLines(ulines,"./linesResults/ulines.txt");
    printLines(vlines,"./linesResults/vlines.txt");

//    for (int i = 0; i < ulines.size();++i)
//    {
//        Point startPt(ulines[i](0),ulines[i](1));
//        Point endPt(ulines[i](2),ulines[i](3));
//        line(udispClone, startPt, endPt, Scalar(255,0,0));//标记像素点的类别，颜色区分
////        line(img, startPt, endPt, Scalar(255,0,0));//标记像素点的类别，颜色区分
//    }

//    for (int i = 0; i < vlines.size();++i)
//    {
//        Point startPt(vlines[i](0),vlines[i](1));
//        Point endPt(vlines[i](2),vlines[i](3));
//        line(vdispClone, startPt, endPt, Scalar(255,0,0));//标记像素点的类别，颜色区分
////        line(img, startPt, endPt, Scalar(0,0,255));//标记像素点的类别，颜色区分
//    }

    namedWindow("2-5U视差图检测");
    imshow("2-5U视差图检测",udispClone);
    namedWindow("2-5V视差图检测");
    imshow("2-5V视差图检测",vdispClone);

    /// 2-4匹配在UV中检测到的直线
    vector<Rect2f> rects = matchLines(vlines,ulines);
    // 在原图中绘制障碍物矩形区域
    Mat imgclone = img.clone();
    for (int j = 0; j < rects.size(); ++j) {
        rectangle(imgclone,rects[j],Scalar(0,0,255),1,LINE_8);
    }
    namedWindow("2-6障碍物");
    imshow("2-6障碍物",imgclone);


//    // 3-计算深度图
//    Mat K = kitti::kitti_K;
//    float baseline = kitti::kitti_baseline;
//
//    // 3-1 计算深度图
//    Mat depth,pdepth;
//    Mat vc30;
//    disp2Depth(res.disp8u,depth,K,baseline);//depth为16uc1图
////    depth.convertTo(depth, CV_32F, 1.0 / 16);  //除以16得到真实视差值
//    Mat depth8U = Mat(depth.rows, depth.cols, CV_8UC1); //显示
//    normalize(depth, depth8U, 0, 255, NORM_MINMAX, CV_8UC1);
//    namedWindow("3-0深度图");
//    imshow("3-0深度图",depth8U);
//
//    // 3-2 深度图
//    DepthResults depths = disp2Depth(res.disp,K,baseline);
//    fillHoleDepth32f(depths.depth); // 对深度图进行填充
//
//    string save_name1 = "./depthResults/test.txt";
//    printMatToTxt( depths.depth,save_name1);
//
//    Mat depth_filled = Mat32fToGray(depths.depth); //填充图灰度图
//    Mat pdepth_filled = grayToPesudo(depth_filled); //填充图伪彩色图
//    cout<<"depth_filled:"<<depth_filled.type()<<endl;
//    cout<<"pdepth_filled"<<pdepth_filled.type()<<endl;
//
//    Mat vc31;
//    vconcat(depths.depth8u,depth_filled,vc31);
//    namedWindow("3-1深度图");
//    imshow("3-1深度图",vc31);
//
//    Mat vc32;
//    vconcat(depths.pdepth,pdepth_filled,vc32);
//    namedWindow("3-2深度伪彩色图");
//    imshow("3-2深度伪彩色图",vc32);

    waitKey();
    return 0;
}
