# Real-Time Privacy Protection Tool 

>   武岳 12112422 wuy2021@mail.sustech.edu.cn https://yuewu0301.github.io/

## Project Introduction

This project aims to develop an efficient real-time privacy protection tool that utilizes C++ and OpenCV technologies. The tool captures video streams from a camera, performs real-time face detection and recognition. It provides multiple privacy protection modes, including blur (Blur), pixelation (Pixel), and mask overlay (Mask). Users can choose the appropriate protection mode based on their needs. In addition, users can dynamically adjust processing parameters and upload custom mask images to achieve personalized privacy protection effects.

## Environment Setup
This project runs on WSL, using CMake, and utilizes OpenCV's Yunet for face detection. Although WSL2 supports the use of CUDA, it does not support camera access. Therefore, UDP is used for video file transfer. Moreover, WSL2 requires tools like X1 to enable visualization. Next, we will introduce the necessary packages and tools for this project.

YuNet requires CMake ≥ 3.24.0. For updating CMake, refer to: https://blog.csdn.net/qq_37700257/article/details/131787671. After downloading the compressed package from the official website, unzip it, and configure the environment via vim ~/.bashrc.

YuNet requires OpenCV 4.10.0. If using the usual sudo installation, you will find that the version is too low, so manual installation is necessary. For specific instructions, refer to: https://blog.csdn.net/whitephantom1/article/details/136406214. CMake is needed for compiling. You can store multiple versions of OpenCV on the same machine by following the guidance here: https://blog.csdn.net/sylin211/article/details/108997411. You also need to update the environment variables via ~/.bashrc.

For running graphical interfaces in WSL2, Windows needs to install VcXsrv. Specific installation instructions are available at: https://blog.csdn.net/Alisebeast/article/details/106680267. After installation, type xclock in the WSL terminal to check if it displays correctly. I also wrote a display_test.cpp to verify if visualization works.

For accessing the local camera, I chose to send the video stream via UDP to WSL. Open the Windows command prompt and type the following:

```
ffmpeg -f dshow -i video="Integrated Webcam" -preset ultrafast -tune zerolatency -vcodec libx264 -f mpegts udp://172.31.238.217:888 -fflags nobuffer -flush_packets 1 -r 30 -b:v 2M -maxrate 2M -bufsize 4M -analyzeduration 100000 -probesize 100000
```
Here, the ffmpeg -f dshow -list_devices true -i dummy command can be used to list device names (in my case, the camera is called "Integrated Webcam"). You can check your local and WSL IP addresses using ipconfig/ifconfig. Note that the UDP address should be that of the WSL, and 888 is the port.

From my personal experience, the -analyzeduration 100000 parameter sets the stream analysis duration to 100000 microseconds, helping the decoder better identify the stream format and codecs. Increasing this value can help handle unstable or complex streams and is recommended for stable video. The -probesize parameter affects the number of bytes FFmpeg uses to analyze stream data when reading the input stream. A larger value helps FFmpeg better determine the stream codec information, so it is advisable to increase this value.

## Model Construction
After setting up the environment, you can call YuNet. The specific repository can be found here: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet. I also wrote a model_test.cpp to test whether the model can be invoked successfully.

Note that the ONNX file downloaded from the repository is a pointer file. Directly calling it will result in a Failed to parse onnx model in function 'ONNXImporter' error. You need to git clone opencv_zoo and then use git lfs to download the specific ONNX model. For issues related to model recognition, see: https://github.com/opencv/opencv_zoo/issues/31.

### 1. Real-time Face Detection: 
First, I created a YuNet class that is responsible for loading the face detection model and performing inference. This class contains an infer method that takes a frame (in cv::Mat format) as input and returns a matrix (also in cv::Mat format) containing information about the detected faces. Each row in this matrix corresponds to one detected face and contains four values: x, y, w, and h. These values represent the top-left coordinates of the face bounding box, as well as the width (w) and height (h), which define the dimensions of the detected face's bounding box.

To draw the face bounding boxes, I created a drawFaceBoxes method that visualizes the detected faces based on the information returned by the infer method. Each row in the faces matrix represents the coordinates of a detected face, where x, y, w, and h indicate the top-left corner and dimensions of the bounding box (width and height). Using these coordinates, I utilized the cv::rectangle method to draw a green rectangle around each detected face on the original image, highlighting the regions where faces are located.


Real-time rendering: In the main function, I initialized the face detection model and set command-line parameters. Then, I used cv::VideoCapture to read video frames from the UDP stream (frame). Each frame is processed by the infer method, which detects faces in the image. The detection result (faces) is passed to the drawFaceBoxes method to draw face boxes on the video frame. The processed frame is then displayed using cv::imshow, achieving real-time face detection and display.

By continuously reading video frames from the UDP stream and performing real-time face detection using the YuNet model, the program draws bounding boxes around the faces and displays the result.

![alt text](readme_source/image.png)

You can see here that the faces are detected and marked in real-time. The top-left corner shows the current status of blur and pixelation strength.

### 2. Privacy Protection Modes
I created a function applyPrivacyMode to implement privacy protection. The input consists of the original image, face coordinates, and privacy protection mode (blur for blur, pixel for pixelation, and mask for overlay).

For blur, I set the blur kernel size and used GaussianBlur to blur the region inside the face box. The kernel size must be odd, so I ensured it by adjusting the kernel size as int adjusted_kernel_size = std::max(3, blur_kernel_size | 1).

For pixel, I defined the pixel size, then resized the face area smaller before enlarging it back to its original size to achieve pixelation. I used INTER_NEAREST to implement nearest-neighbor interpolation to achieve the pixelation effect.

For mask, I first defined the image mask, loaded the image, resized the mask to fit the detected face area, and converted it to BGR format. Then, I copied the mask image into the original image's face detection box to overlay the mask.

Afterwards, in the while true loop, I set the current privacy mode and used this method to process each video frame, thus achieving privacy protection for the frame.

### 3 & 4. Dynamic Parameter Adjustment and Mode Switching
First, I set up a method setNonBlockingInput using fcntl and read to implement non-blocking input. This allows the program to receive user input in real-time without blocking the main thread, which handles video capture and face detection.

In the while true loop, I used:

```
char input;
ssize_t bytes_read = read(STDIN_FILENO, &input, 1);
```
to read one character from standard input and store it in input. Based on the input, I could dynamically adjust parameters. For example, pressing 1, 2, or 3 would switch between blur, pixel, and mask modes. After switching modes, the program would call applyPrivacyMode in the next loop to apply the selected privacy mode.

By reading ] or [, when in pixel or blur mode, I could increment or decrement the respective blur and pixel sizes. This would take effect in the next loop, enabling dynamic parameter adjustments.

![alt text](readme_source/image-1.png)


![alt text](readme_source/image-2.png)

![alt text](readme_source/image-3.png)

As you can see here, the program displays the current privacy mode, blur strength, and pixel size, which can be adjusted dynamically.




### 5. Uploading the Mask Image
Similarly, when we press 'u', it is recognized as a command to upload a new image. However, since only one character is detected at a time and the image path is a string, I implemented the following approach:

In the main function, I defined a bool waitingForInput variable outside of the while true loop to indicate whether the program is currently waiting for the new image address. When the program detects the character 'u', it sets this variable to true (which is false by default).

Inside the loop, if waitingForInput is true, I start recording the subsequent input, appending each character to a string until a carriage return (\r) or newline (\n) character is encountered. At that point, I consider the string as the complete image path, attempt to read the image, and if successful, replace the original mask image with the new one.


![alt text](readme_source/image-6.png)
press u to input pictures' address, we correctly change the mask picture to miaowazhongzi.


### Running Instructions Command-line Arguments
You can construct the relevant code in the main function. I created a function called print_usage to remind users of the correct input format in case of an error.



![alt text](readme_source/image-4.png)
The code could provide correct information if you input incorrect argument.

After you input: /build/privacy_protector  -mode blur -blur_size 20 
![alt text](readme_source/image-5.png)
We successfully change the mode to Blur and set the blur number to 20.


## Code Implementation
The full code has been uploaded to: https://github.com/YueWu0301/PrivacyProtect_CS219Proj

You can also visit my website: https://yuewu0301.github.io/projects/

Below is the complete code:

```
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
    std::string mask_image_path = "kedaya.png";  // 默认遮罩图片路径

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

```