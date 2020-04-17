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
    string img_name = "000000R";
    string suffix = ".png";

    // 1-双目立体匹配
    StereoResults res = MyStereoMatch::stereoSGBM(img_path,img_name,suffix);
    Mat disp_filled = res.disp.clone(); //视差图32F
    MyStereoMatch::fillHoleDepth32f(disp_filled);
//    cout<<disp_filled.type()<<" "<<res.disp.type()<<endl;
    disp_filled = Mat32fToGray(disp_filled);//视差图填充后灰度图

    // 2-视差图孔洞填充
    Mat vc21;
    vconcat(res.disp8u,disp_filled,vc21);
    namedWindow("2-1视差图孔洞填充前后");
    imshow("2-1视差图孔洞填充前后",vc21);

//    Mat pdisp,fpdisp,vc22;
//    applyColorMap(res.disp8u,pdisp,COLORMAP_JET);
//    applyColorMap(disp_filled,fpdisp,COLORMAP_JET);
//    vconcat(pdisp,fpdisp,vc22);
//    namedWindow("2-2视差图伪彩图");
//    imshow("2-2视差图伪彩图",vc22);

    // 2-1计算UV视差图
    int mMaxDisp = 128;
    Mat udisp = cv::Mat(mMaxDisp,res.disp8u.cols,CV_16UC1);
    Mat vdisp =cv::Mat(res.disp8u.rows,mMaxDisp,CV_16UC1);

    computeUDisparity(udisp,res.disp8u);
    computeVDisparity(vdisp,res.disp8u);

    Mat udisp8u = Mat(res.disp8u.size(),CV_8UC1);
    Mat vdisp8u = Mat(res.disp8u.size(),CV_8UC1);
    normalize(udisp, udisp8u, 0, 255, NORM_MINMAX, CV_8UC1);
    normalize(vdisp, vdisp8u, 0, 255, NORM_MINMAX, CV_8UC1);

    Mat rmDisp = removeObstacle(disp_filled,udisp8u);

    namedWindow("2-3U视差图");
    imshow("2-3U视差图",udisp8u);
    namedWindow("2-3V视差图");
    imshow("2-3V视差图",vdisp8u);
    namedWindow("2-4移除障碍后视差图");
    imshow("2-4移除障碍后视差图",rmDisp);
    imwrite("./disp/udisp.png",udisp8u);
    imwrite("./disp/vdisp.png",vdisp8u);
    imwrite("./disp/disp.png",res.disp8u);
    imwrite("./disp/rmDisp.png",rmDisp);


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
