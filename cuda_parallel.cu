#include <dlib/image_processing/shape_predictor.h>
#include <dlib/opencv.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <set>

namespace fs = std::filesystem;

static bool isImageFile(const fs::path& p) {
    static const std::set<std::string> exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
    };
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return exts.count(ext) > 0;
}

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
    const fs::path& images_dir,
    const fs::path& results_dir,
    cv::CascadeClassifier& face_cascade,
    dlib::shape_predictor& sp,
    std::ofstream& lm_csv,
    std::ofstream& time_csv,
    cv::cuda::Stream& stream
)
{
    // Use relative path from images_dir to avoid collisions from subdirectories
    fs::path rel = fs::relative(img_path, images_dir);
    std::string img_label = rel.string(); // e.g. "person/img_0001.jpg"
    std::string img_name  = rel.filename().string();

    cv::Mat bgr = cv::imread(img_path);
    if (bgr.empty()) return;

    // Mirror subdirectory structure under results_dir
    fs::path out_dir = results_dir / rel.parent_path();
    fs::create_directories(out_dir);

    //------------------------------------
    // Upload to GPU
    //------------------------------------

    cv::cuda::GpuMat gpu_img;
    gpu_img.upload(bgr, stream);

    //------------------------------------
    // Convert to grayscale on GPU
    //------------------------------------

    cv::cuda::GpuMat gpu_gray;
    cv::cuda::cvtColor(gpu_img, gpu_gray, cv::COLOR_BGR2GRAY, 0, stream);

    // Download grayscale for CPU face detection
    cv::Mat gray;
    gpu_gray.download(gray, stream);
    stream.waitForCompletion();

    //------------------------------------
    // Face Detection (CPU)
    // Note: cv::cuda::CascadeClassifier's NCV XML parser is broken in
    // OpenCV 4.14.0-pre; CPU classifier is used with GPU grayscale output.
    //------------------------------------

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));

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

        lm_csv << img_label << "," << fi << ","
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

    cv::imwrite((out_dir / ("overlay_" + img_name)).string(), overlay);

    time_csv << img_label << "," << faces.size() << "\n";
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
    // Load face cascade
    //------------------------------------

    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error: could not load haarcascade_frontalface_default.xml\n";
        return 1;
    }

    //------------------------------------
    // CSV Outputs
    //------------------------------------

    std::ofstream time_csv(results_dir / "cuda_times.csv");
    std::ofstream lm_csv(results_dir / "landmarks.csv");

    //------------------------------------
    // Collect image paths (recursive, images only)
    //------------------------------------

    std::vector<std::string> image_paths;

    for (auto& entry : fs::recursive_directory_iterator(images_dir))
        if (entry.is_regular_file() && isImageFile(entry.path()))
            image_paths.push_back(entry.path().string());

    std::cout << "Found " << image_paths.size() << " images\n";

    const size_t SAMPLE = 500;
    if (image_paths.size() > SAMPLE) {
        std::mt19937 rng(std::random_device{}());
        std::shuffle(image_paths.begin(), image_paths.end(), rng);
        image_paths.resize(SAMPLE);
        std::cout << "Randomly sampled " << SAMPLE << " images\n";
    }

    //------------------------------------
    // Create CUDA streams
    //------------------------------------

    const int STREAMS = 4;
    cv::cuda::Stream streams[STREAMS];

    //------------------------------------
    // Process images in parallel streams
    //------------------------------------

    for (size_t i = 0; i < image_paths.size(); i++)
    {
        int sid = i % STREAMS;

        processImageGPU(
            image_paths[i],
            images_dir,
            results_dir,
            face_cascade,
            sp,
            lm_csv,
            time_csv,
            streams[sid]
        );
    }

    //------------------------------------
    // Cleanup
    //------------------------------------

    std::cout << "CUDA pipeline finished\n";
}
