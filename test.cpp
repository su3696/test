#include <opencv2/opencv.hpp>
#include include/KPlasticWrap.h

int main(){
    cv::Mat img(64,64,CV_8UC3,cv::Scalar(120,120,120));
    cv::rectangle(img, {8,8}, {56,56}, cv::Scalar(255,255,255), -1);
    KPlasticWrap pw; pw.setParams(4, 20.0, 2.0, 45.0, 5.0, 3.0);
    cv::Mat out = pw.apply(img);
    return (out.empty()?1:0);
}
