#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>

enum PrivacyMode { NONE, BLUR, PIXEL, MASK };
// 用于初始化时的提醒，纠错，etc
void print_usage(const std::string& program_name) { 
    std::cerr << "Usage: " << program_name << " [-mode mode] [-blur_size blur_size] [-pixel_size pixel_size] [-mask_image mask_image]" << std::endl;
    std::cerr << "  -mode <mode>       : Set the initial mode (blur, pixel, mask). Default is 'blur'." << std::endl;
    std::cerr << "  -blur_size <size>  : Set the blur kernel size (only valid in 'blur' mode)." << std::endl;
    std::cerr << "  -pixel_size <size> : Set the pixel size (only valid in 'pixel' mode)." << std::endl;
    std::cerr << "  -mask_image <path> : Set the mask image path (only valid in 'mask' mode)." << std::endl;
}


// 调用YuNet
class YuNet {
public:
    YuNet(const std::string &model_path,
          const cv::Size &input_size = cv::Size(320, 320),
          float conf_threshold = 0.6f, //人脸检测的置信度阈值
          float nms_threshold = 0.3f,
          int top_k = 5000,
          int backend_id = 0,
          int target_id = 0)
        : model_path_(model_path), input_size_(input_size),
          conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
          top_k_(top_k), backend_id_(backend_id), target_id_(target_id) {
        model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
    }

    void setInputSize(const cv::Size &input_size) {
        input_size_ = input_size;
        model->setInputSize(input_size_);
    }

    cv::Mat infer(const cv::Mat &image) {
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

void drawFaceBoxes(cv::Mat &image, const cv::Mat &faces) {
    for (int i = 0; i < faces.rows; ++i) {
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));

        // 绘制人脸框
        cv::rectangle(image, cv::Rect(x1, y1, w, h), cv::Scalar(0, 255, 0), 2);
    }
}


cv::Mat applyPrivacyMode(cv::Mat image, const cv::Mat &faces, int blur_kernel_size, int pixel_size, cv::Mat &mask_img, PrivacyMode mode) {
    auto output_image = image.clone();

    for (int i = 0; i < faces.rows; ++i) {
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));

        // 确保区域在图像范围内
        cv::Rect face_rect(x1, y1, w, h);
        face_rect &= cv::Rect(0, 0, image.cols, image.rows);

        switch (mode) {
            case BLUR: {
                int adjusted_kernel_size = std::max(3, blur_kernel_size | 1);
                cv::GaussianBlur(output_image(face_rect), output_image(face_rect), cv::Size(adjusted_kernel_size, adjusted_kernel_size), 0);
                break;
            }
            case PIXEL: {
                cv::Mat pixelated = output_image(face_rect);
                cv::resize(pixelated, pixelated, cv::Size(pixelated.cols / pixel_size, pixelated.rows / pixel_size));
                cv::resize(pixelated, output_image(face_rect), face_rect.size(), 0, 0, cv::INTER_NEAREST);
                break;
            }
            case MASK: {
                if (!mask_img.empty()) {
                    cv::Mat mask_resized;
                    cv::resize(mask_img, mask_resized, cv::Size(w, h));

                    // 检查并调整通道数，确保与视频帧匹配
                    if (mask_resized.channels() == 1) {
                        // 灰度图转换为 BGR
                        cv::cvtColor(mask_resized, mask_resized, cv::COLOR_GRAY2BGR);
                    } else if (mask_resized.channels() == 4) {
                        // RGBA 转换为 BGR
                        cv::cvtColor(mask_resized, mask_resized, cv::COLOR_BGRA2BGR);
                    }

                    // 将遮罩图片复制到目标区域
                    mask_resized.copyTo(output_image(face_rect));
                }
                break;
            }
            default:
                break;
        }
    }

    return output_image;
}

bool waitingForInput = false; // 标记是否等待用户输入路径
std::string new_mask_path = ""; // 新的遮罩路径

void setNonBlockingInput() {
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
}

int main(int argc, char *argv[]) {
    // 默认参数
    PrivacyMode mode = NONE;  // 默认是普通模式
    int blur_kernel_size = 15;
    int pixel_size = 10;
    std::string mask_image_path = "pika.png";  // 默认遮罩图片路径

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // 处理 -mode 或 -m 参数
        if (arg == "-mode" || arg == "-m") {
            if (i + 1 < argc) {
                std::string mode_str = argv[++i];
                if (mode_str == "blur") {
                    mode = BLUR;
                } else if (mode_str == "pixel") {
                    mode = PIXEL;
                } else if (mode_str == "mask") {
                    mode = MASK;
                } else {
                    std::cerr << "Invalid mode: " << mode_str << ". Valid options are 'blur', 'pixel', or 'mask'." << std::endl;
                    return -1;
                }
            } else {
                std::cerr << "Error: -mode requires an argument." << std::endl;
                return -1;
            }
        }
        // 处理 -blur_size 或 -b 参数
        else if (arg == "-blur_size" || arg == "-b") {
            if (mode != BLUR) {
                std::cerr << "Error: -blur_size is only valid when -mode is set to 'blur'." << std::endl;
                return -1;
            }
            if (i + 1 < argc) {
                blur_kernel_size = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: -blur_size requires an argument." << std::endl;
                return -1;
            }
        }
        // 处理 -pixel_size 或 -p 参数
        else if (arg == "-pixel_size" || arg == "-p") {
            if (mode != PIXEL) {
                std::cerr << "Error: -pixel_size is only valid when -mode is set to 'pixel'." << std::endl;
                return -1;
            }
            if (i + 1 < argc) {
                pixel_size = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: -pixel_size requires an argument." << std::endl;
                return -1;
            }
        }
        // 处理 -mask_image 或 -i 参数
        else if (arg == "-mask_image" || arg == "-i") {
            if (mode != MASK) {
                std::cerr << "Error: -mask_image is only valid when -mode is set to 'mask'." << std::endl;
                return -1;
            }
            if (i + 1 < argc) {
                mask_image_path = argv[++i];
            } else {
                std::cerr << "Error: -mask_image requires an argument." << std::endl;
                return -1;
            }
        }
        // 无效参数
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return -1;
        }
    }

    // 输出当前设置的参数
    std::cout << "Mode: " << (mode == BLUR ? "Blur" : mode == PIXEL ? "Pixel" : mode == MASK ? "Mask" : "None") << std::endl;
    std::cout << "Blur Size: " << blur_kernel_size << std::endl;
    std::cout << "Pixel Size: " << pixel_size << std::endl;
    std::cout << "Mask Image Path: " << mask_image_path << std::endl;

    std::string udp_stream = "udp://localhost:888?fifo_size=5000000&analyzeduration=1000000&probesize=1000000";

    // 加载遮罩图片
    cv::Mat mask_img = cv::imread(mask_image_path, cv::IMREAD_UNCHANGED);
    if (mask_img.empty()) {
        std::cout << "无法加载遮罩图片：" << mask_image_path << std::endl;
        return -1;
    }

    // 设置非阻塞输入
    setNonBlockingInput();

    // 模型路径
    std::string model_path = "face_detection_yunet_2023mar.onnx";
    YuNet model(model_path, cv::Size(320, 320), 0.6, 0.3, 5000, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU);

    // 打开 UDP 流
    cv::VideoCapture stream(udp_stream, cv::CAP_FFMPEG);
    if (!stream.isOpened()) {
        std::cout << "无法打开 UDP 流。" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::namedWindow("Privacy Protector", cv::WINDOW_AUTOSIZE);

    bool waitingForInput = false;
    std::string new_mask_path;
    int enter_count = 0;

    while (true) {
        // 读取 UDP 流帧
        if (!stream.read(frame)) {
            std::cout << "无法读取视频帧，退出。" << std::endl;
            break;
        }

        // 设置输入大小
        model.setInputSize(frame.size());

        // 检测人脸
        cv::Mat faces = model.infer(frame);


        

        if (mode == NONE) {
            // 默认模式：标记人脸框
            drawFaceBoxes(frame, faces);
        } else {
            // 根据模式应用隐私保护
            frame = applyPrivacyMode(frame, faces, blur_kernel_size, pixel_size, mask_img, mode);
        }
        // if (faces.empty()) {
        // // 显示原始视频流
        // std::string mode_text = (mode == NONE ? "None" : mode == BLUR ? "Blur" : mode == PIXEL ? "Pixel" : "Mask");
        // std::string text = "Mode: " + mode_text + " | Blur: " + std::to_string(blur_kernel_size) + " | Pixel: " + std::to_string(pixel_size);
        // cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        // cv::imshow("Privacy Protector", frame);
        // } else {
        // // 检测到人脸时，应用隐私保护模式
        //     if (mode == NONE) {
        //     // 默认模式：标记人脸框
        //     drawFaceBoxes(frame, faces);
        // } else {
        //     // 根据模式应用隐私保护
        //     frame = applyPrivacyMode(frame, faces, blur_kernel_size, pixel_size, mask_img, mode);
        // }
        // }






        // 显示当前模式和参数
        std::string mode_text = (mode == NONE ? "None" : mode == BLUR ? "Blur" : mode == PIXEL ? "Pixel" : "Mask");
        std::string text = "Mode: " + mode_text + " | Blur: " + std::to_string(blur_kernel_size) + " | Pixel: " + std::to_string(pixel_size);
        cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

        cv::imshow("Privacy Protector", frame);

        // 检测命令行输入
        char input;
        ssize_t bytes_read = read(STDIN_FILENO, &input, 1);
        if (bytes_read > 0) {
            // 处理输入
            if (waitingForInput) {
                // 处理换行符
                if (input == '\n' || input == '\r') {
                    enter_count++;
                    if (enter_count == 1) {
                        new_mask_path.clear();
                    } else if (enter_count == 2) {
                        std::cout << "路径输入完成: " << new_mask_path << std::endl;

                        // 加载新的遮罩图片
                        cv::Mat new_mask_img = cv::imread(new_mask_path, cv::IMREAD_UNCHANGED);
                        if (new_mask_img.empty()) {
                            std::cout << "无法加载遮罩图片：" << new_mask_path << std::endl;
                        } else {
                            mask_img = new_mask_img;
                            std::cout << "成功加载遮罩图片：" << new_mask_path << std::endl;
                        }

                        waitingForInput = false;
                        new_mask_path.clear();
                        enter_count = 0;
                    }
                } else {
                    new_mask_path += input;
                }
            }

            // 切换模式
            if (input == 'u') {
                waitingForInput = true;
                enter_count = 0;
                std::cout << "请输入图片地址: " << std::endl;
            }

            // 切换模式
            
            if (input == '1') {
                mode = BLUR;
                std::cout << "切换到模糊模式" << std::endl;
            } else if (input == '2') {
                mode = PIXEL;
                std::cout << "切换到像素化模式" << std::endl;
            } else if (input == '3') {
                mode = MASK;
                std::cout << "切换到遮罩模式" << std::endl;
            } else if (input == '0') {
                mode = NONE;
                std::cout << "关闭所有模式" << std::endl;
            }else if (input == 27) { // ESC 键退出
                break;
            } else if (input == '[') { // 减小模糊核或像素块大小
                if (mode == BLUR && blur_kernel_size > 3) blur_kernel_size -= 2;
                if (mode == PIXEL && pixel_size > 1) pixel_size -= 1;
            } else if (input == ']') { // 增大模糊核或像素块大小
                if (mode == BLUR) blur_kernel_size += 2;
                if (mode == PIXEL) pixel_size += 1;
            }
        }
        char key = cv::waitKey(1);

    }

    cv::destroyAllWindows();
    return 0;
}