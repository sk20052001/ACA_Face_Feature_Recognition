#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaobjdetect.hpp>

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

namespace fs = std::filesystem;

static cv::Point2f toCvPoint(const dlib::point& p) {
    return cv::Point2f((float)p.x(), (float)p.y());
}

static cv::Point2f avgPoints(const std::vector<cv::Point2f>& pts) {
    cv::Point2f s(0,0);
    for (auto &p : pts) { s.x += p.x; s.y += p.y; }
    s.x /= pts.size(); 
    s.y /= pts.size();
    return s;
}

struct ResultData {
    std::string name;
    int faces;
    double detect_ms;
};

void processImageGPU(
    const std::string& img_path,
    cv::Ptr<cv::cuda::CascadeClassifier>& face_detector,
    dlib::shape_predictor& sp,
    std::ofstream& lm_csv,
    std::ofstream& time_csv,
    cudaStream_t stream
)
{
    std::string img_name = fs::path(img_path).filename().string();

    cv::Mat bgr = cv::imread(img_path);
    if (bgr.empty()) return;

    //------------------------------------
    // Upload to GPU
    //------------------------------------

    cv::cuda::GpuMat gpu_img;
    gpu_img.upload(bgr, stream);

    //------------------------------------
    // Convert to grayscale on GPU
    //------------------------------------

    cv::cuda::GpuMat gpu_gray;

    cv::cuda::cvtColor(
        gpu_img,
        gpu_gray,
        cv::COLOR_BGR2GRAY,
        0,
        cv::cuda::StreamAccessor::wrapStream(stream)
    );

    //------------------------------------
    // GPU Face Detection
    //------------------------------------

    cv::cuda::GpuMat facesBuf;

    face_detector->detectMultiScale(
        gpu_gray,
        facesBuf,
        cv::cuda::StreamAccessor::wrapStream(stream)
    );

    //------------------------------------
    // Download detected faces
    //------------------------------------

    std::vector<cv::Rect> faces;

    face_detector->convert(facesBuf, faces);

    //------------------------------------
    // Landmark detection (CPU)
    //------------------------------------

    dlib::cv_image<dlib::bgr_pixel> dimg(bgr);

    cv::Mat overlay = bgr.clone();

    for (size_t fi = 0; fi < faces.size(); fi++) {

        dlib::rectangle r(
            faces[fi].x,
            faces[fi].y,
            faces[fi].x + faces[fi].width,
            faces[fi].y + faces[fi].height
        );

        auto shape = sp(dimg, r);

        std::vector<cv::Point2f> left_eye, right_eye;

        for (int i = 36; i <= 41; i++)
            left_eye.push_back(toCvPoint(shape.part(i)));

        for (int i = 42; i <= 47; i++)
            right_eye.push_back(toCvPoint(shape.part(i)));

        cv::Point2f nose = toCvPoint(shape.part(30));
        cv::Point2f mouth_l = toCvPoint(shape.part(48));
        cv::Point2f mouth_r = toCvPoint(shape.part(54));

        cv::Point2f le = avgPoints(left_eye);
        cv::Point2f re = avgPoints(right_eye);

        lm_csv << img_name << "," << fi << ","
               << le.x << "," << le.y << ","
               << re.x << "," << re.y << ","
               << nose.x << "," << nose.y << ","
               << mouth_l.x << "," << mouth_l.y << ","
               << mouth_r.x << "," << mouth_r.y << "\n";

        //------------------------------------
        // Draw overlay
        //------------------------------------

        cv::circle(overlay, le, 3, {0,255,0}, -1);
        cv::circle(overlay, re, 3, {0,255,0}, -1);
        cv::circle(overlay, nose, 3, {255,0,0}, -1);
        cv::circle(overlay, mouth_l, 3, {0,0,255}, -1);
        cv::circle(overlay, mouth_r, 3, {0,0,255}, -1);

        for (int i = 0; i < shape.num_parts(); i++) {
            cv::circle(overlay, toCvPoint(shape.part(i)), 1, {255,255,0}, -1);
        }
    }

    //------------------------------------
    // Save overlay image
    //------------------------------------

    cv::imwrite("results/overlay_" + img_name, overlay);

    time_csv << img_name << "," << faces.size() << "\n";
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cout << "Usage:\n";
        std::cout << "./cuda_face <landmark_model> <images_dir> <results>\n";
        return 0;
    }

    std::string model_path = argv[1];
    fs::path images_dir = argv[2];
    fs::path results_dir = argv[3];

    fs::create_directories(results_dir);

    //------------------------------------
    // Load Dlib landmark model
    //------------------------------------

    dlib::shape_predictor sp;
    dlib::deserialize(model_path) >> sp;

    //------------------------------------
    // GPU Face detector
    //------------------------------------

    cv::Ptr<cv::cuda::CascadeClassifier> face_detector =
        cv::cuda::CascadeClassifier::create(
            "haarcascade_frontalface_default.xml");

    //------------------------------------
    // CSV Outputs
    //------------------------------------

    std::ofstream time_csv(results_dir / "cuda_times.csv");
    std::ofstream lm_csv(results_dir / "landmarks.csv");

    //------------------------------------
    // Collect image paths
    //------------------------------------

    std::vector<std::string> image_paths;

    for (auto& entry : fs::directory_iterator(images_dir))
        image_paths.push_back(entry.path().string());

    //------------------------------------
    // Create CUDA streams
    //------------------------------------

    const int STREAMS = 4;
    cudaStream_t streams[STREAMS];

    for (int i = 0; i < STREAMS; i++)
        cudaStreamCreate(&streams[i]);

    //------------------------------------
    // Process images in parallel streams
    //------------------------------------

    for (size_t i = 0; i < image_paths.size(); i++)
    {
        int sid = i % STREAMS;

        processImageGPU(
            image_paths[i],
            face_detector,
            sp,
            lm_csv,
            time_csv,
            streams[sid]
        );
    }

    //------------------------------------
    // Cleanup
    //------------------------------------

    for (int i = 0; i < STREAMS; i++)
        cudaStreamDestroy(streams[i]);

    std::cout << "CUDA pipeline finished\n";
}