


#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 创建一个蓝色的图像
    cv::Mat img(300, 500, CV_8UC3, cv::Scalar(255, 0, 0)); // 蓝色背景

    // 创建窗口
    cv::namedWindow("Test Window", cv::WINDOW_AUTOSIZE);

    // 显示图像
    cv::imshow("Test Window", img);

    // 等待用户按下任意键
    std::cout << "窗口已弹出，按任意键关闭..." << std::endl;
    cv::waitKey(0); // 等待用户输入，按键后关闭窗口

    return 0;
}
// #include <opencv2/opencv.hpp>
// #include <iostream>

// int main() {
//     std::cout << cv::getBuildInformation() << std::endl;
//     return 0;
// }
