// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



#include "paddle_api.h"
#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <limits>

using namespace paddle::lite_api;  // NOLINT

const std::vector<float> INPUT_MEAN = {0.f, 0.f, 0.f};
const std::vector<float> INPUT_STD = {1.f, 1.f, 1.f};

std::vector<std::string> load_labels(const std::string &path) {
    std::ifstream file;
    std::vector<std::string> labels;
    file.open(path);
    while (file) {
        std::string line;
        std::getline(file, line);
        labels.push_back(line);
    }
    file.clear();
    file.close();
    return labels;
}


void preprocess(cv::Mat &photo, float *input_data) {

    cv::resize(photo, photo, cv::Size(224, 224), 0.f, 0.f);   //resize到224x224
    cv::cvtColor(photo, photo, CV_BGRA2RGB);                  //BGR->RGB 与训练时输入一致
    photo.convertTo(photo, CV_32FC3, 1 / 255.f, 0.f);         //归一化
//     std::cout << photo << std::endl;//查看形状

    // NHWC->NCHW
    int image_size = photo.cols * photo.rows;
    const float *image_data = reinterpret_cast<const float *>(photo.data);
    float32x4_t vmean0 = vdupq_n_f32(INPUT_MEAN[0]);
    float32x4_t vmean1 = vdupq_n_f32(INPUT_MEAN[1]);
    float32x4_t vmean2 = vdupq_n_f32(INPUT_MEAN[2]);
    float32x4_t vscale0 = vdupq_n_f32(1.0f / INPUT_STD[0]);
    float32x4_t vscale1 = vdupq_n_f32(1.0f / INPUT_STD[1]);
    float32x4_t vscale2 = vdupq_n_f32(1.0f / INPUT_STD[2]);
    float *input_data_c0 = input_data;
    float *input_data_c1 = input_data + image_size;
    float *input_data_c2 = input_data + image_size * 2;
    int i = 0;
    for (; i < image_size - 3 ; i += 4) {
        float32x4x3_t vin3 = vld3q_f32(image_data);
        float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
        float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
        float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
        float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
        float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
        float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
        vst1q_f32(input_data_c0, vs0);
        vst1q_f32(input_data_c1, vs1);
        vst1q_f32(input_data_c2, vs2);
        image_data += 12;
        input_data_c0 += 4;
        input_data_c1 += 4;
        input_data_c2 += 4;
    }
    for (; i < image_size; i++) {
        *(input_data_c0++) = (*(image_data++) - INPUT_MEAN[0]) / INPUT_STD[0];
        *(input_data_c1++) = (*(image_data++) - INPUT_MEAN[1]) / INPUT_STD[1];
        *(input_data_c2++) = (*(image_data++) - INPUT_MEAN[2]) / INPUT_STD[2];
    } 
    
}


void run(cv::Mat &photo,
              std::vector<std::string> &word_labels,
              std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {

    // Get Input Tensor
    std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
    input_tensor->Resize({1, 3, 224, 224});
    auto* input_data = input_tensor->mutable_data<float>();
    preprocess(photo, input_data);

    // Detection Model Run
    predictor->Run();

    // Get Output Tensor
    std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));

    // Previous result
    auto* output_data = output_tensor->mutable_data<float>();
    std::cout << "预测结果为:" << word_labels[std::distance(output_data, std::max_element(output_data, output_data + 4))] << std::endl;

    for (int i = 0; i < 4; i++) {
        std::cout << "Original Output[" << i << "]: " << output_tensor->data<float>()[i] << std::endl;
    }

}

int main(int argc, char** argv) {

    std::string model = argv[1];

    // Create Predictor For Detction Model
    MobileConfig config;
    config.set_model_from_file(model);
    std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);


    if (argc == 4){
        std::string label_path = argv[2];
        std::vector<std::string> word_labels = load_labels(label_path);
        std::string img_path = argv[3];
        std::cout << argv[3] << std::endl;

        cv::Mat photo = imread(img_path, cv::IMREAD_COLOR);
        run(photo, word_labels, predictor);
        cv::cvtColor(photo, photo, CV_RGB2BGR);//RGB->BGR 还原正常颜色显示
        cv::imshow("bear", photo);
        cv::waitKey(0);
    } else if (argc == 3) {
        std::string label_path = argv[2];
        std::vector<std::string> word_labels = load_labels(label_path);
        cv::VideoCapture cap(-1);
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 640);
        if (!cap.isOpened()) {
            return -1;
        }
        while (1) {
            cv::Mat input_image;
            cap >> input_image;
            run(input_image, word_labels, predictor);
            cv::cvtColor(input_image, input_image, CV_RGB2BGR);//RGB->BGR 还原正常颜色显示
            cv::imshow("Mask Detection Demo", input_image);
            if (cv::waitKey(1) == char('q')) {
                break;
            }
        }
        cap.release();
        cv::destroyAllWindows();
    } else {
        exit(1);
    }
    return 0;
}
