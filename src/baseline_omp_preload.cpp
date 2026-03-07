#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <atomic>
#include <iomanip>

#include "timer.h"

namespace fs = std::filesystem;

static cv::Point2f toCvPoint(const dlib::point& p) {
    return cv::Point2f((float)p.x(), (float)p.y());
}

static cv::Point2f avgPoints(const std::vector<cv::Point2f>& pts) {
    cv::Point2f s(0,0);
    for (auto &p : pts) { s.x += p.x; s.y += p.y; }
    s.x /= (float)pts.size(); s.y /= (float)pts.size();
    return s;
}

static bool isImageExt(std::string ext) {
    for (auto &c : ext) c = (char)tolower(c);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png");
}

struct ImageItem {
    std::string name;
    cv::Mat bgr;
};

int main(int argc, char** argv) {
    // ./baseline_omp_preload <model.dat> <images_dir> <results_dir> [num_threads]
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <shape_predictor_68_face_landmarks.dat> <images_dir> <results_dir> [num_threads]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const fs::path images_dir = argv[2];
    const fs::path results_dir = argv[3];
    fs::create_directories(results_dir);

    int num_threads = 1;
    if (argc >= 5) num_threads = std::max(1, std::atoi(argv[4]));

#ifdef USE_OPENMP
    omp_set_num_threads(num_threads);
#endif

    // Collect file list
    std::vector<fs::path> files;
    for (auto const& entry : fs::directory_iterator(images_dir)) {
        if (!entry.is_regular_file()) continue;
        if (!isImageExt(entry.path().extension().string())) continue;
        files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::cerr << "ERROR: No images found in: " << images_dir << "\n";
        return 1;
    }

    // Preload images (serial)
    Timer t_preload; t_preload.start();

    std::vector<ImageItem> items;
    items.reserve(files.size());

    for (const auto& p : files) {
        cv::Mat bgr = cv::imread(p.string(), cv::IMREAD_COLOR);
        if (bgr.empty()) {
            std::cerr << "WARN: Failed to load: " << p << "\n";
            continue;
        }
        items.push_back(ImageItem{p.filename().string(), bgr});
    }

    double preload_ms = t_preload.ms();

    if (items.empty()) {
        std::cerr << "ERROR: No valid images loaded.\n";
        return 1;
    }

    // Output CSVs
    const fs::path time_path = results_dir / "omp_preload_times.csv";
    const fs::path lm_path   = results_dir / "omp_preload_landmarks.csv";

    std::ofstream time_csv(time_path);
    time_csv << "image,detect_ms,landmarks_ms,total_compute_ms,num_faces,threads\n";

    std::ofstream lm_csv(lm_path);
    lm_csv << "image,face_idx,"
           << "left_eye_x,left_eye_y,right_eye_x,right_eye_y,nose_x,nose_y,"
           << "mouth_left_x,mouth_left_y,mouth_right_x,mouth_right_y\n";

    std::vector<std::string> time_lines(items.size());
    std::vector<std::string> lm_lines(items.size());
    std::atomic<int> processed_ok{0};

    // Batch compute timer (parallel region only)
    Timer t_batch_compute;
    t_batch_compute.start();

#ifdef USE_OPENMP
#pragma omp parallel
#endif
    {
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor sp;
        try {
            dlib::deserialize(model_path) >> sp;
        } catch (const std::exception& e) {
#ifdef USE_OPENMP
#pragma omp critical
#endif
            std::cerr << "ERROR: model load failed: " << e.what() << "\n";
        }

#ifdef USE_OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (size_t idx = 0; idx < items.size(); idx++) {
            const std::string& img_name = items[idx].name;
            const cv::Mat& bgr = items[idx].bgr;

            Timer t_total; t_total.start();

            dlib::cv_image<dlib::bgr_pixel> dimg(bgr);

            Timer t_det; t_det.start();
            std::vector<dlib::rectangle> faces = detector(dimg);
            double det_ms = t_det.ms();

            Timer t_lm; t_lm.start();
            std::vector<dlib::full_object_detection> shapes;
            shapes.reserve(faces.size());
            for (auto& r : faces) shapes.push_back(sp(dimg, r));
            double lm_ms = t_lm.ms();

            double total_compute_ms = t_total.ms();

            {
                std::ostringstream oss;
                oss << img_name << ","
                    << det_ms << "," << lm_ms << ","
                    << total_compute_ms << ","
                    << faces.size() << "," << num_threads << "\n";
                time_lines[idx] = oss.str();
            }

            std::ostringstream lmoss;
            for (size_t fi = 0; fi < shapes.size(); fi++) {
                const auto& s = shapes[fi];

                std::vector<cv::Point2f> left_eye, right_eye;
                left_eye.reserve(6);
                right_eye.reserve(6);
                for (int i = 36; i <= 41; i++) left_eye.push_back(toCvPoint(s.part(i)));
                for (int i = 42; i <= 47; i++) right_eye.push_back(toCvPoint(s.part(i)));

                cv::Point2f nose = toCvPoint(s.part(30));
                cv::Point2f mouth_l = toCvPoint(s.part(48));
                cv::Point2f mouth_r = toCvPoint(s.part(54));

                cv::Point2f le = avgPoints(left_eye);
                cv::Point2f re = avgPoints(right_eye);

                lmoss << img_name << "," << fi << ","
                      << le.x << "," << le.y << ","
                      << re.x << "," << re.y << ","
                      << nose.x << "," << nose.y << ","
                      << mouth_l.x << "," << mouth_l.y << ","
                      << mouth_r.x << "," << mouth_r.y << "\n";
            }
            lm_lines[idx] = lmoss.str();

            processed_ok.fetch_add(1, std::memory_order_relaxed);
        }
    }

    double batch_compute_ms = t_batch_compute.ms();

    for (const auto& s : time_lines) time_csv << s;
    for (const auto& s : lm_lines)   lm_csv << s;

    // run summary
    {
        std::ofstream summary(results_dir / "run_summary.csv");
        summary << "mode,threads,images,preload_ms,batch_compute_ms,batch_total_ms\n";
        summary << "omp_preload," << num_threads << "," << processed_ok.load() << ","
                << std::fixed << std::setprecision(2)
                << preload_ms << "," << batch_compute_ms << ",\n";
    }

    const double imgs_per_sec = (batch_compute_ms > 0.0)
        ? (1000.0 * processed_ok.load() / batch_compute_ms)
        : 0.0;

    std::cout << "Done.\n"
              << "Images loaded: " << items.size() << "\n"
              << "Images processed: " << processed_ok.load() << "\n"
              << "Threads: " << num_threads << "\n"
              << "Preload time (serial I/O): " << std::fixed << std::setprecision(2) << preload_ms << " ms\n"
              << "Batch compute time (parallel): " << std::fixed << std::setprecision(2) << batch_compute_ms << " ms\n"
              << "Throughput (compute-only): " << std::fixed << std::setprecision(2) << imgs_per_sec << " images/sec\n"
              << "Wrote:\n  " << time_path << "\n  " << lm_path << "\n  "
              << (results_dir / "run_summary.csv") << "\n";

    return 0;
}
