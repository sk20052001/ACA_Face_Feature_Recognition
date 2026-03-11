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
#include <chrono>

namespace fs = std::filesystem;                                                             // creates scope handler for filesystem library

static bool isImageFile(const fs::path &p)                                                  // Checking whether the file is valid image
{
    static const std::set<std::string> exts = {                                             // Checks for valid extension with O(log n) and avoid duplicates
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"};
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);                         // Updates the extension to lower case in the same container 
    return exts.count(ext) > 0;
}

static cv::Point2f toCvPoint(const dlib::point &p)                                          // Converts a 2D point from the Dlib C++ Library format to the OpenCV format
{
    return cv::Point2f((float)p.x(), (float)p.y());
}

static cv::Point2f avgPoints(const std::vector<cv::Point2f> &pts)                           // Computes the center of multiple points
{
    cv::Point2f s(0, 0);
    for (auto &p : pts)
    {
        s.x += p.x;
        s.y += p.y;
    }
    s.x /= pts.size();
    s.y /= pts.size();
    return s;
}

// struct ResultData
// {
//     std::string name;
//     int faces;
//     double detect_ms;
// };

void processImageGPU(
    const std::string &img_path,
    const fs::path &images_dir,
    const fs::path &results_dir,
    cv::CascadeClassifier &face_cascade,
    dlib::shape_predictor &sp,
    std::ofstream &lm_csv,
    std::ofstream &time_csv,
    cv::cuda::Stream &stream)
{
    
    fs::path rel = fs::relative(img_path, images_dir);                                      // Use relative path from images_dir to avoid collisions from subdirectories
    std::string img_label = rel.string();
    std::string img_name = rel.filename().string();

    cv::Mat bgr = cv::imread(img_path);                                                     // Reads the image from the path
    if (bgr.empty())                                                                        // Returns if can't load the image
        return;

    auto t_start = std::chrono::steady_clock::now();                                        // Take a snapshot of the Start time

    fs::path out_dir = results_dir / rel.parent_path();                                     // Creates output directory
    fs::create_directories(out_dir);                                                        // with the same structure as dataset

    cv::cuda::GpuMat gpu_img;                                                               // Upload the image
    gpu_img.upload(bgr, stream);                                                            // to GPU Memory
    auto t_upload = std::chrono::steady_clock::now();                                       // Take a snapshot of current time

    cv::cuda::GpuMat gpu_gray;
    cv::cuda::cvtColor(gpu_img, gpu_gray, cv::COLOR_BGR2GRAY, 0, stream);                   // Convert to grayscale on GPU
    auto t_gray = std::chrono::steady_clock::now();                                         // Take a snapshot of current time

    cv::Mat gray;
    gpu_gray.download(gray, stream);                                                        // Download grayscale for CPU face detection
    stream.waitForCompletion();                                                             // Ensures GPU completes before CPU continues
    auto t_download = std::chrono::steady_clock::now();                                     // Take a snapshot of current time

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));                // Identifies objects in the images which is the faces here    
    auto t_detect = std::chrono::steady_clock::now();                                       // Take a snapshot of current time

    dlib::cv_image<dlib::bgr_pixel> dimg(bgr);                                              // Wraps the image for dlib format

    cv::Mat overlay = bgr.clone();                                                          // Clone the original image to draw the landmarks

    for (size_t fi = 0; fi < faces.size(); fi++)
    {

        dlib::rectangle r(                                                                  // Convert the openCV to dlib rectangle
            faces[fi].x,
            faces[fi].y,
            faces[fi].x + faces[fi].width,
            faces[fi].y + faces[fi].height
        );

        auto shape = sp(dimg, r);                                                           // Object for the landmark detector

        std::vector<cv::Point2f> left_eye, right_eye;

        for (int i = 36; i <= 41; i++)                                                      // Landmark for eyes
            left_eye.push_back(toCvPoint(shape.part(i)));

        for (int i = 42; i <= 47; i++)
            right_eye.push_back(toCvPoint(shape.part(i)));

        cv::Point2f nose = toCvPoint(shape.part(30));                                       // Landmark for nose
        cv::Point2f mouth_l = toCvPoint(shape.part(48));                                    // Landmark for mouth
        cv::Point2f mouth_r = toCvPoint(shape.part(54));

        cv::Point2f le = avgPoints(left_eye);                                               // Landmark for eyes center
        cv::Point2f re = avgPoints(right_eye);

        lm_csv << img_label << "," << fi << ","                                             // Save the landmarks to csv
               << le.x << "," << le.y << ","
               << re.x << "," << re.y << ","
               << nose.x << "," << nose.y << ","
               << mouth_l.x << "," << mouth_l.y << ","
               << mouth_r.x << "," << mouth_r.y << "\n";

        cv::circle(overlay, le, 3, {0, 255, 0}, -1);                                        // Draw the landmarks
        cv::circle(overlay, re, 3, {0, 255, 0}, -1);
        cv::circle(overlay, nose, 3, {255, 0, 0}, -1);
        cv::circle(overlay, mouth_l, 3, {0, 0, 255}, -1);
        cv::circle(overlay, mouth_r, 3, {0, 0, 255}, -1);

        for (int i = 0; i < shape.num_parts(); i++)
        {
            cv::circle(overlay, toCvPoint(shape.part(i)), 1, {255, 255, 0}, -1);
        }
    }
    auto t_landmarks = std::chrono::steady_clock::now();                                    // Take a snapshot of current time

    cv::imwrite((out_dir / ("overlay_" + img_name)).string(), overlay);                     // Save the landmarked image in the designated location
    auto t_save = std::chrono::steady_clock::now();                                         // Take a snapshot of current time

    auto ms = [](auto a, auto b)                                                            // Helper function to compute till millisecond
    {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    time_csv << img_label << ","                                                            // Write the timing to csv
             << faces.size() << ","
             << ms(t_start, t_upload) << ","
             << ms(t_upload, t_gray) << ","
             << ms(t_gray, t_download) << ","
             << ms(t_download, t_detect) << ","
             << ms(t_detect, t_landmarks) << ","
             << ms(t_landmarks, t_save) << ","
             << ms(t_start, t_save) << "\n";
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage:\n";
        std::cout << "./cuda_face <landmark_model> <images_dir> <results>\n";
        return 0;
    }

    std::string model_path = argv[1];
    fs::path images_dir = argv[2];
    fs::path results_dir = argv[3];

    fs::create_directories(results_dir);

    dlib::shape_predictor sp;                                                               // Dlib landmark model handler
    dlib::deserialize(model_path) >> sp;                                                    // Model path

    cv::CascadeClassifier face_cascade;                                                     // Handler for face cascade
    if (!face_cascade.load("haarcascade_frontalface_default.xml"))                          // Load the trained cascade
    {
        std::cerr << "Error: could not load haarcascade_frontalface_default.xml\n";
        return 1;
    }

    std::ofstream time_csv(results_dir / "cuda_times.csv");                                 // Output files
    std::ofstream lm_csv(results_dir / "landmarks.csv");

    lm_csv << "image,face_idx,le_x,le_y,re_x,re_y,nose_x,nose_y,mouth_l_x,mouth_l_y,mouth_r_x,mouth_r_y\n";
    time_csv << "image,num_faces,gpu_upload_ms,gpu_gray_ms,gpu_download_ms,face_detect_ms,landmarks_ms,save_ms,total_ms\n";

    std::vector<std::string> image_paths;

    for (auto &entry : fs::recursive_directory_iterator(images_dir))                        // Traverse dataset recursively and keep only the image files
        if (entry.is_regular_file() && isImageFile(entry.path()))
            image_paths.push_back(entry.path().string());                                   // Store the image path

    std::cout << "Found " << image_paths.size() << " images\n";

    const size_t SAMPLE = 500;                                                              
    if (image_paths.size() > SAMPLE)                                                        // Select random images from datasets as per the count mentioned
    {
        std::mt19937 rng(std::random_device{}());
        std::shuffle(image_paths.begin(), image_paths.end(), rng);
        image_paths.resize(SAMPLE);
        std::cout << "Randomly sampled " << SAMPLE << " images\n";
    }

    auto t_program_start = std::chrono::steady_clock::now();                                // Take snapshot of the time of program start

    const int STREAMS = 4;                                                                  // Specify the number of CUDA streams
    cv::cuda::Stream streams[STREAMS];

    for (size_t i = 0; i < image_paths.size(); i++)                                         // Start the CUDA process
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
            streams[sid]);
    }

    double total_s = std::chrono::duration<double>(                                         // Computer the total time for the process execution
        std::chrono::steady_clock::now() - t_program_start).count();

    std::cout << "CUDA pipeline finished\n";
    std::cout << "Total time: " << total_s << " s (" << total_s * 1000.0 << " ms) across " << image_paths.size() << " images\n";
}
