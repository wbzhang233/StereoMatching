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
    namedWindow("disp_depth");
    namedWindow("disp");
    while(count<10000){
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

        Mat udisp,vdisp;
        computeUDisparity(udisp,stereoRes.disp);
        computeVDisparity(vdisp,stereoRes.disp);


        // 3-计算深度图
        DepthResults depthRes = disp2Depth(dispFilled, kitti_K, kitti_baseline);
        Mat pdepth = grayToPesudo(depthRes.depth8u);

        string save_name1 = "./depthResults/test"+ to_string(count)+".txt";
        printMatToTxt( depthRes.depth,save_name1);


        imshow("disp_depth",pdepth);
//        Mat vc2;
//        vconcat(pdispFilled,pdepth,vc2);//填充后视差图伪彩图,深度图伪彩色图
//        imshow("disp_depth",vc2);

        waitKey(10);
        count++;
    }

    cout<<"All images done..."<<endl;
    return 0;
}