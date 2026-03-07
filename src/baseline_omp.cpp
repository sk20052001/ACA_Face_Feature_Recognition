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

int main(int argc, char** argv) {
    // ./baseline_omp <model.dat> <images_dir> <results_dir> [num_threads] [write_overlays]
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <shape_predictor_68_face_landmarks.dat> <images_dir> <results_dir> [num_threads] [write_overlays]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const fs::path images_dir = argv[2];
    const fs::path results_dir = argv[3];
    fs::create_directories(results_dir);

    int num_threads = 1;
    if (argc >= 5) num_threads = std::max(1, std::atoi(argv[4]));

    bool write_overlays = false;
    if (argc >= 6) write_overlays = (std::atoi(argv[5]) != 0);

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

    // Output CSVs
    const fs::path time_path = results_dir / "omp_times.csv";
    const fs::path lm_path   = results_dir / "omp_landmarks.csv";

    std::ofstream time_csv(time_path);
    time_csv << "image,load_ms,detect_ms,landmarks_ms,total_ms,num_faces,threads\n";

    std::ofstream lm_csv(lm_path);
    lm_csv << "image,face_idx,"
           << "left_eye_x,left_eye_y,right_eye_x,right_eye_y,nose_x,nose_y,"
           << "mouth_left_x,mouth_left_y,mouth_right_x,mouth_right_y\n";

    std::vector<std::string> time_lines(files.size());
    std::vector<std::string> lm_lines(files.size());

    std::atomic<int> processed_ok{0};

    Timer t_batch;
    t_batch.start();

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
        for (size_t idx = 0; idx < files.size(); idx++) {
            const fs::path& img_path = files[idx];
            const std::string img_name = img_path.filename().string();

            Timer t_total; t_total.start();

            Timer t_load; t_load.start();
            cv::Mat bgr = cv::imread(img_path.string(), cv::IMREAD_COLOR);
            double load_ms = t_load.ms();

            if (bgr.empty()) {
                std::ostringstream oss;
                oss << img_name << "," << load_ms << ",0,0," << t_total.ms() << ",0," << num_threads << "\n";
                time_lines[idx] = oss.str();
                lm_lines[idx].clear();
                continue;
            }

            dlib::cv_image<dlib::bgr_pixel> dimg(bgr);

            Timer t_det; t_det.start();
            std::vector<dlib::rectangle> faces = detector(dimg);
            double det_ms = t_det.ms();

            Timer t_lm; t_lm.start();
            std::vector<dlib::full_object_detection> shapes;
            shapes.reserve(faces.size());
            for (auto& r : faces) shapes.push_back(sp(dimg, r));
            double lm_ms = t_lm.ms();

            double total_ms = t_total.ms();

            {
                std::ostringstream oss;
                oss << img_name << ","
                    << load_ms << "," << det_ms << "," << lm_ms << ","
                    << total_ms << "," << faces.size() << "," << num_threads << "\n";
                time_lines[idx] = oss.str();
            }

            std::ostringstream lmoss;

            cv::Mat overlay;
            if (write_overlays) overlay = bgr.clone();

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

                if (write_overlays) {
                    cv::circle(overlay, le, 3, {0,255,0}, -1);
                    cv::circle(overlay, re, 3, {0,255,0}, -1);
                    cv::circle(overlay, nose, 3, {255,0,0}, -1);
                    cv::circle(overlay, mouth_l, 3, {0,0,255}, -1);
                    cv::circle(overlay, mouth_r, 3, {0,0,255}, -1);

                    for (int i = 0; i < s.num_parts(); i++) {
                        cv::circle(overlay, toCvPoint(s.part(i)), 1, {255,255,0}, -1);
                    }
                }
            }

            lm_lines[idx] = lmoss.str();

            if (write_overlays) {
                const fs::path out_path = results_dir / ("overlay_" + img_name);
                cv::imwrite(out_path.string(), overlay);
            }

            processed_ok.fetch_add(1, std::memory_order_relaxed);
        }
    }

    double batch_total_ms = t_batch.ms();

    for (const auto& s : time_lines) time_csv << s;
    for (const auto& s : lm_lines)   lm_csv << s;

    // run summary
    {
        std::ofstream summary(results_dir / "run_summary.csv");
        summary << "mode,threads,images,preload_ms,batch_compute_ms,batch_total_ms\n";
        summary << "omp_e2e," << num_threads << "," << processed_ok.load()
                << ",,,"
                << std::fixed << std::setprecision(2) << batch_total_ms << "\n";
    }

    std::cout << "Done. Processed " << processed_ok.load() << " / " << files.size() << " images.\n"
              << "Threads: " << num_threads << "\n"
              << "Batch total time: " << std::fixed << std::setprecision(2) << batch_total_ms << " ms\n"
              << "Wrote:\n  " << time_path << "\n  " << lm_path << "\n  "
              << (results_dir / "run_summary.csv") << "\n";

    return 0;
}
