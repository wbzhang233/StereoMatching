//
// Created by wbzhang233 on 2020/4/2.
//

#include "stereoMatch.h"

namespace MyStereoMatch{

/// SGBM
StereoResults stereoSGBM(string img_path, string img_name,string suffix)
{
    StereoResults res;
    string img_nameR = img_path + img_name + suffix ;
    string img_truename = img_name.substr(0,img_name.find_last_of('R'));
    string img_nameL = img_path  + img_truename+'L'+ suffix;

    Mat left = imread(img_nameL, IMREAD_GRAYSCALE);
    Mat right = imread(img_nameR, IMREAD_GRAYSCALE);
    Mat disp;
    int mindisparity = 0;
    int ndisparities = 64;
    int SADWindowSize = 11;
    //SGBM
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);
    int P1 = 8 * left.channels() * SADWindowSize* SADWindowSize;
    int P2 = 32 * left.channels() * SADWindowSize* SADWindowSize;
    sgbm->setP1(P1);
    sgbm->setP2(P2);
    sgbm->setPreFilterCap(15);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleRange(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setDisp12MaxDiff(1);
//    sgbm->setMode(cv::StereoSGBM::MODE_SGBM); //8方向
    sgbm->compute(left, right, disp);

    disp.convertTo(disp, CV_32F, 1.0 / 16);                //除以16得到真实视差值
    Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示
    normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
    string save_name1 = "results/"+img_name.substr(0,img_name.find_first_of('.')-1)+"SGBM-disp8u.png";
    string save_name2 = "results/"+img_name.substr(0,img_name.find_first_of('.')-1)+"SGBM-disp.png";
    imwrite(save_name1, disp8U);
    imwrite(save_name2, disp);

    res.disp = disp;
    res.disp8u = disp8U;
    return res;
}

StereoResults stereoSGBM(vector<string> img_path,string img_name,string suffix)
{
    StereoResults res;
    string img_nameL = img_path[0] + img_name  ;
    string img_nameR = img_path[1]  + img_name;

    Mat left = imread(img_nameL, IMREAD_GRAYSCALE);
    Mat right = imread(img_nameR, IMREAD_GRAYSCALE);
    Mat disp;
    int mindisparity = 0;
    int ndisparities = 64;
    int SADWindowSize = 11;
    //SGBM
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);
    int P1 = 8 * left.channels() * SADWindowSize* SADWindowSize;
    int P2 = 32 * left.channels() * SADWindowSize* SADWindowSize;
    sgbm->setP1(P1);
    sgbm->setP2(P2);
    sgbm->setPreFilterCap(15);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleRange(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM); //8方向
    sgbm->compute(left, right, disp);

    disp.convertTo(disp, CV_32F, 1.0 / 16);                //除以16得到真实视差值
    Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示
    normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
    string save_name1 = "results/"+img_name.substr(0,img_name.find_first_of('.')-1)+"SGBM-disp8u.png";
    string save_name2 = "results/"+img_name.substr(0,img_name.find_first_of('.')-1)+"SGBM-disp.png";
    imwrite(save_name1, disp8U);
    imwrite(save_name2, disp);

    res.disp = disp;
    res.disp8u = disp8U;
    return res;
}

StereoResults stereoSGBM(Mat &imgL,Mat &imgR,string img_name)
{
    StereoResults res;
    Mat disp;
    int mindisparity = 0;
    int ndisparities = 64;
    int SADWindowSize = 11;
    //SGBM
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);
    int P1 = 8 * imgL.channels() * SADWindowSize* SADWindowSize;
    int P2 = 32 * imgL.channels() * SADWindowSize* SADWindowSize;
    sgbm->setP1(P1);
    sgbm->setP2(P2);
    sgbm->setPreFilterCap(15);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleRange(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setDisp12MaxDiff(1);
//    sgbm->setMode(cv::StereoSGBM::MODE_SGBM); //8方向
    sgbm->compute(imgL, imgR, disp);

    disp.convertTo(disp, CV_32F, 1.0 / 16);                //除以16得到真实视差值
    Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示
    normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
    string save_name1 = "results/"+img_name.substr(0,img_name.find_first_of('.')-1)+"SGBM-disp8u.png";
    string save_name2 = "results/"+img_name.substr(0,img_name.find_first_of('.')-1)+"SGBM-disp.png";
    imwrite(save_name1, disp8U);
    imwrite(save_name2, disp);

    res.disp = disp;
    res.disp8u = disp8U;
    return res;
}

/// BM
StereoResults stereoBM(string img_path, string img_name,string suffix)
{
    StereoResults res;

    string img_nameR = img_path +img_name+suffix;
    string img_truename = img_name.substr(0,img_name.find_last_of('R'));
    string img_nameL = img_path + img_truename+"L"+suffix;

    Mat left = imread(img_nameL, IMREAD_GRAYSCALE);
    Mat right = imread(img_nameR, IMREAD_GRAYSCALE);
    Mat disp;

    int mindisparity = 0;
    int ndisparities = 64;
    int SADWindowSize = 11;

    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(ndisparities, SADWindowSize);

    // setter
    bm->setPreFilterType(1);
    bm->setBlockSize(SADWindowSize);
    bm->setMinDisparity(mindisparity);
    bm->setNumDisparities(ndisparities);
    bm->setPreFilterSize(15);
    bm->setPreFilterCap(31);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(5);
    bm->setSpeckleRange(32);
    bm->setSpeckleWindowSize(100);
    bm->setDisp12MaxDiff(1);

    copyMakeBorder(left, left, 0, 0, 80, 0, BORDER_REPLICATE);  //防止黑边
    copyMakeBorder(right, right, 0, 0, 80, 0, BORDER_REPLICATE);
    bm->compute(left, right, disp);

    disp.convertTo(disp, CV_32F, 1.0 / 16); //除以16得到真实视差值
    disp = disp.colRange(80, disp.cols);
    Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);
    normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
    string save_name1 = "results/"+img_name.substr(0,img_name.find_first_of('.')-1)+"BM-disp8u.png";
    string save_name2 = "results/"+img_name.substr(0,img_name.find_first_of('.')-1)+"BM-disp.png";
    imwrite(save_name1, disp8U);
    imwrite(save_name2, disp);

    res.disp = disp;
    res.disp8u = disp8U;
    return res;
}

StereoResults stereoBM(vector<string> img_path,string img_name,string suffix)
{
    StereoResults res;
    string img_nameL = img_path[0] + img_name  ;
    string img_nameR = img_path[1]  + img_name;

    Mat left = imread(img_nameL, IMREAD_GRAYSCALE);
    Mat right = imread(img_nameR, IMREAD_GRAYSCALE);
    Mat disp;

    int mindisparity = 0;
    int ndisparities = 64;
    int SADWindowSize = 11;

    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(ndisparities, SADWindowSize);

    // setter
    bm->setPreFilterType(1);
    bm->setBlockSize(SADWindowSize);
    bm->setMinDisparity(mindisparity);
    bm->setNumDisparities(ndisparities);
    bm->setPreFilterSize(15);
    bm->setPreFilterCap(31);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(5);
    bm->setSpeckleRange(32);
    bm->setSpeckleWindowSize(100);
    bm->setDisp12MaxDiff(1);

    copyMakeBorder(left, left, 0, 0, 80, 0, BORDER_REPLICATE);  //防止黑边
    copyMakeBorder(right, right, 0, 0, 80, 0, BORDER_REPLICATE);
    bm->compute(left, right, disp);

    disp.convertTo(disp, CV_32F, 1.0 / 16); //除以16得到真实视差值
    disp = disp.colRange(80, disp.cols);
    Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);
    normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
    string save_name1 = "results/"+img_name.substr(0,img_name.find_first_of('.')-1)+"BM-disp8u.png";
    string save_name2 = "results/"+img_name.substr(0,img_name.find_first_of('.')-1)+"BM-disp.png";
    imwrite(save_name1, disp8U);
    imwrite(save_name2, disp);

    res.disp = disp;
    res.disp8u = disp8U;
    return res;
}

StereoResults stereoBM(Mat &imgL,Mat &imgR,string img_name)
{
    StereoResults res;

    Mat disp;
    int mindisparity = 0;
    int ndisparities = 64;
    int SADWindowSize = 11;

    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(ndisparities, SADWindowSize);

    // setter
    bm->setPreFilterType(1);
    bm->setBlockSize(SADWindowSize);
    bm->setMinDisparity(mindisparity);
    bm->setNumDisparities(ndisparities);
    bm->setPreFilterSize(15);
    bm->setPreFilterCap(31);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(5);
    bm->setSpeckleRange(32);
    bm->setSpeckleWindowSize(100);
    bm->setDisp12MaxDiff(1);

    copyMakeBorder(imgL, imgL, 0, 0, 80, 0, BORDER_REPLICATE);  //防止黑边
    copyMakeBorder(imgR, imgR, 0, 0, 80, 0, BORDER_REPLICATE);
    bm->compute(imgL, imgR, disp);

    disp.convertTo(disp, CV_32F, 1.0 / 16); //除以16得到真实视差值
    disp = disp.colRange(80, disp.cols);
    Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);
    normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
    string save_name1 = "results/"+img_name.substr(0,img_name.find_first_of('.')-1)+"BM-disp8u.png";
    string save_name2 = "results/"+img_name.substr(0,img_name.find_first_of('.')-1)+"BM-disp.png";
    imwrite(save_name1, disp8U);
    imwrite(save_name2, disp);

    res.disp = disp;
    res.disp8u = disp8U;
    return res;
}

/// UV视差图
void computeUDisparity(Mat &UdispMap,Mat disp)
{
//    UdispMap = Mat(disp.size(),CV_16UC1);
    UdispMap.setTo(0);
    int width=disp.cols;
    int height=disp.rows;

    for(int row=0;row<height;row++)
    {
        auto  pRowInDisp=disp.ptr<uchar>(row);
        for(int col=0;col<width;col++)
        {
            uint8_t currDisp=pRowInDisp[col];
            if(currDisp>0&&currDisp<128)
            {
                UdispMap.at<ushort>(currDisp,col)++;
            }
        }
    }
}
void computeVDisparity(Mat &VdispMap,Mat disp)
{
//    VdispMap = Mat(disp.size(),CV_16UC1);
    VdispMap.setTo(0);
    int width=disp.cols;
    int height=disp.rows;

    for(int row=0;row<height;row++)
    {
        auto  pRowInDisp=disp.ptr<uchar>(row);
        for(int col=0;col<width;col++)
        {
            uint8_t currDisp=pRowInDisp[col];
            if(currDisp>0&&currDisp<128)
            {
                VdispMap.at<ushort>(row,currDisp)++;
            }

        }
    }
}
Mat removeObstacle(Mat Disparity,Mat Udisparity)
{
    cv::Mat mObstacleMap;
    mObstacleMap.create(Disparity.rows, Disparity.cols, CV_8UC1);

    mObstacleMap.setTo(0);

    int height = Disparity.rows;
    int width = Disparity.cols;


    for (int v = 0; v < height; v++)
    {
        uint8_t* pRowInDisp = Disparity.ptr<uchar>(v);
        uint8_t* pRowInObsMap = mObstacleMap.ptr<uchar>(v);
        for (int u = 0; u < width; u++)
        {
            uint8_t currDisp = pRowInDisp[u];
            if (currDisp < 128 && Udisparity.at<ushort>(currDisp, u) > 10)
                pRowInObsMap[u] = 255;
        }
    }
    return mObstacleMap;

}

/**
函数作用：视差图转深度图
输入：
　　dispMap ----视差图，8位单通道，CV_8UC1
　　K       ----内参矩阵，float类型
输出：
　　depthMap ----深度图，16位无符号单通道，CV_16UC1
*/
void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap, cv::Mat K,float baseline)
{
    int type = dispMap.type();

    float fx = K.at<float>(0, 0);
    float fy = K.at<float>(1, 1);
    float cx = K.at<float>(0, 2);
    float cy = K.at<float>(1, 2);

    if (type == CV_8U)
    {
        const float PI = 3.14159265358;
        int height = dispMap.rows;
        int width = dispMap.cols;
        depthMap = Mat(dispMap.size(),CV_16UC1);

//        uchar* dispData = (uchar*)dispMap.data;
//        ushort* depthData = (ushort*)depthMap.data;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
//                int id = i*width + j;
                if(dispMap.at<uchar>(i,j)!=0){
                    depthMap.at<ushort>(i,j) = ushort((float)fx *baseline / ((float)dispMap.at<uchar>(i,j)) );
                }
//                depthData[id] = ushort( (float)fx *baseline / ((float)dispData[id]) );
            }
        }
    }
    else
    {
        cout << "please confirm dispImg's type!" << endl;
        cv::waitKey(0);
    }
}

/// 计算深度图
// 输入视差图为CV_16SC1或者CV_32FC1
// 输出深度图res.depth 为32FC1
DepthResults disp2Depth(Mat dispMap,Mat K,float baseline)
{
    DepthResults res;
    res.depth = Mat(dispMap.size(),CV_32FC1);
    res.depth8u = Mat(dispMap.size(),CV_8UC1);
    res.pdepth = Mat(dispMap.size(),CV_8UC1);
    if(dispMap.type()!=CV_32FC1){
        return res;
    }

    float fx = K.at<float>(0,0);
    for (int i = 0; i < dispMap.rows; ++i) {
        for (int j = 0; j < dispMap.cols; ++j) {
            if(dispMap.at<float>(i,j)!=0){
                res.depth.at<float>(i,j) = fx*baseline/dispMap.at<float>(i,j);
            }
        }
    }

    res.depth8u = Mat32fToGray(res.depth); //此处无返回值
    res.pdepth = grayToPesudo(res.depth8u);
    return  res;
};

// 孔洞填充,输入为32F
void fillHoleDepth32f(cv::Mat& depth)
{
    const int width = depth.cols;
    const int height = depth.rows;
    float* data = (float*)depth.data;
    cv::Mat integralMap = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat ptsMap = cv::Mat::zeros(height, width, CV_32S);
    double* integral = (double*)integralMap.data;
    int* ptsIntegral = (int*)ptsMap.data;
    memset(integral, 0, sizeof(double) * width * height);
    memset(ptsIntegral, 0, sizeof(int) * width * height);
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            if (data[id2] > 1e-3)
            {
                integral[id2] = data[id2];
                ptsIntegral[id2] = 1;
            }
        }
    }
    // 积分区间
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 1; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - 1];
            ptsIntegral[id2] += ptsIntegral[id2 - 1];
        }
    }
    for (int i = 1; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - width];
            ptsIntegral[id2] += ptsIntegral[id2 - width];
        }
    }
    int wnd;
    double dWnd = 8;
    while (dWnd > 1)
    {
        wnd = int(dWnd);
        dWnd /= 2;
        for (int i = 0; i < height; ++i)
        {
            int id1 = i * width;
            for (int j = 0; j < width; ++j)
            {
                int id2 = id1 + j;
                int left = j - wnd - 1;
                int right = j + wnd;
                int top = i - wnd - 1;
                int bot = i + wnd;
                left = max(0, left);
                right = min(right, width - 1);
                top = max(0, top);
                bot = min(bot, height - 1);
                int dx = right - left;
                int dy = (bot - top) * width;
                int idLeftTop = top * width + left;
                int idRightTop = idLeftTop + dx;
                int idLeftBot = idLeftTop + dy;
                int idRightBot = idLeftBot + dx;
                int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
                double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);
                if (ptsCnt <= 0)
                {
                    continue;
                }
                data[id2] = float(sumGray / ptsCnt);
            }
        }
        int s = wnd / 2 * 2 + 1;
        if (s > 201)
        {
            s = 201;
        }
//        cv::GaussianBlur(depth, depth, cv::Size(s, s), s, s);
    }
}

// CV32F转8UC1
Mat Mat32fToGray(Mat disp)
{
    Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示
    normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
    return disp8U;
}

//// 灰度图转伪彩图
Mat grayToPesudo(Mat &gray,ColormapTypes colormapType)
{
    assert(gray.type()==CV_8UC1);
    Mat res;
    applyColorMap(gray,res,colormapType);
    return res;
};

void printMatToTxt(Mat img,string save_name)
{
    ofstream ofstream1;
    ofstream1.open(save_name);

    // 此处限定img为CV_32FC1
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            float value =img.at<float>(i,j);
            ofstream1 <<to_string(value/1000)<<" ";
        }
        ofstream1<<endl;
    }

    cout<<save_name<<" has saved!"<<endl;
}

}