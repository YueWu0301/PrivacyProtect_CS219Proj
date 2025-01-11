#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// YuNet模型封装
class YuNet {
public:
    YuNet(const std::string& model_path,
          const cv::Size& input_size = cv::Size(320, 320),
          float conf_threshold = 0.6f,
          float nms_threshold = 0.3f,
          int top_k = 5000,
          int backend_id = cv::dnn::DNN_BACKEND_OPENCV,
          int target_id = cv::dnn::DNN_TARGET_CPU)
        : model_path_(model_path), input_size_(input_size),
          conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
          top_k_(top_k), backend_id_(backend_id), target_id_(target_id) {
        model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
    }

    void setInputSize(const cv::Size& input_size) {
        input_size_ = input_size;
        model->setInputSize(input_size_);
    }

    cv::Mat infer(const cv::Mat image) {
        cv::Mat res;
        model->detect(image, res);
        return res;
    }

private:
    cv::Ptr<cv::FaceDetectorYN> model;
    std::string model_path_;
    cv::Size input_size_;
    float conf_threshold_;
    float nms_threshold_;
    int top_k_;
    int backend_id_;
    int target_id_;
};

// 可视化函数：绘制人脸框、置信度
cv::Mat visualize(const cv::Mat& image, const cv::Mat& faces) {
    static cv::Scalar box_color{0, 255, 0};

    auto output_image = image.clone();
    for (int i = 0; i < faces.rows; ++i) {
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

        float conf = faces.at<float>(i, 14);
        cv::putText(output_image, cv::format("Conf: %.2f", conf), cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2);
    }
    return output_image;
}

int main() {
    // YuNet模型路径
    std::string model_path = "/root/cproject/model_test/face_detection_yunet_2023mar.onnx";
    YuNet model(model_path);

    // 测试图像路径
    std::string image_path = "test.jpg"; // 替换为你的测试图像路径
    cv::Mat image = cv::imread(image_path);

    if (image.empty()) {
        std::cerr << "无法读取图像，请检查路径：" << image_path << std::endl;
        return -1;
    }

    // 设置输入大小为图像大小
    model.setInputSize(image.size());

    // 推理检测人脸
    cv::Mat faces = model.infer(image);

    // 打印检测到的人脸信息
    std::cout << faces.rows << " 人脸检测到。" << std::endl;
    for (int i = 0; i < faces.rows; ++i) {
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));
        float conf = faces.at<float>(i, 14);
        std::cout << "人脸 " << i + 1 << ": (x=" << x1 << ", y=" << y1 << ", w=" << w << ", h=" << h 
                  << ", 置信度=" << conf << ")" << std::endl;
    }

    // 可视化结果
    cv::Mat output = visualize(image, faces);
    cv::imshow("Face Detection Result", output);

    // 保存结果到本地（可选）
    cv::imwrite("result.jpg", output);

    // 按任意键退出
    cv::waitKey(0);

    return 0;
}
