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
#include <iomanip>

#include "timer.h"

namespace fs = std::filesystem;

// Convert dlib::point -> cv::Point2f
static cv::Point2f toCvPoint(const dlib::point& p) {
    return cv::Point2f((float)p.x(), (float)p.y());
}

// Average a set of points
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
    // Usage:
    // ./baseline_serial <shape_predictor_68_face_landmarks.dat> <images_dir> <results_dir>
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <shape_predictor_68_face_landmarks.dat> <images_dir> <results_dir>\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const fs::path images_dir = argv[2];
    const fs::path results_dir = argv[3];
    fs::create_directories(results_dir);

    // Load detectors
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor sp;
    dlib::deserialize(model_path) >> sp;

    // Collect file list (deterministic ordering)
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
    std::ofstream time_csv(results_dir / "serial_times.csv");
    time_csv << "image,load_ms,detect_ms,landmarks_ms,total_ms,num_faces\n";

    // Compute-only CSV (detect + landmarks)
    std::ofstream compute_csv(results_dir / "serial_compute_times.csv");
    compute_csv << "image,detect_ms,landmarks_ms,total_compute_ms,num_faces\n";

    std::ofstream lm_csv(results_dir / "landmarks.csv");
    lm_csv << "image,face_idx,"
           << "left_eye_x,left_eye_y,right_eye_x,right_eye_y,nose_x,nose_y,"
           << "mouth_left_x,mouth_left_y,mouth_right_x,mouth_right_y\n";

    size_t img_count = 0;

    Timer t_batch;
    t_batch.start();

    for (const auto& path : files) {
        const std::string img_name = path.filename().string();

        Timer t_total; t_total.start();

        // Load image
        Timer t_load; t_load.start();
        cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
        double load_ms = t_load.ms();

        if (bgr.empty()) {
            std::cerr << "Failed to load: " << path << "\n";
            continue;
        }

        dlib::cv_image<dlib::bgr_pixel> dimg(bgr);

        // Face detect
        Timer t_det; t_det.start();
        std::vector<dlib::rectangle> faces = detector(dimg);
        double det_ms = t_det.ms();

        // Landmarks
        Timer t_lm; t_lm.start();
        std::vector<dlib::full_object_detection> shapes;
        shapes.reserve(faces.size());
        for (auto& r : faces) shapes.push_back(sp(dimg, r));
        double lm_ms = t_lm.ms();

        double total_ms = t_total.ms();
        double total_compute_ms = det_ms + lm_ms;

        // Record timing
        time_csv << img_name << ","
                 << load_ms << "," << det_ms << "," << lm_ms << ","
                 << total_ms << "," << faces.size() << "\n";

        compute_csv << img_name << ","
                    << det_ms << "," << lm_ms << ","
                    << total_compute_ms << "," << faces.size() << "\n";

        // Overlay
        cv::Mat overlay = bgr.clone();
        for (size_t fi = 0; fi < shapes.size(); fi++) {
            const auto& s = shapes[fi];

            std::vector<cv::Point2f> left_eye, right_eye;
            for (int i = 36; i <= 41; i++) left_eye.push_back(toCvPoint(s.part(i)));
            for (int i = 42; i <= 47; i++) right_eye.push_back(toCvPoint(s.part(i)));

            cv::Point2f nose = toCvPoint(s.part(30));
            cv::Point2f mouth_l = toCvPoint(s.part(48));
            cv::Point2f mouth_r = toCvPoint(s.part(54));

            cv::Point2f le = avgPoints(left_eye);
            cv::Point2f re = avgPoints(right_eye);

            lm_csv << img_name << "," << fi << ","
                   << le.x << "," << le.y << ","
                   << re.x << "," << re.y << ","
                   << nose.x << "," << nose.y << ","
                   << mouth_l.x << "," << mouth_l.y << ","
                   << mouth_r.x << "," << mouth_r.y << "\n";

            cv::circle(overlay, le, 3, {0,255,0}, -1);
            cv::circle(overlay, re, 3, {0,255,0}, -1);
            cv::circle(overlay, nose, 3, {255,0,0}, -1);
            cv::circle(overlay, mouth_l, 3, {0,0,255}, -1);
            cv::circle(overlay, mouth_r, 3, {0,0,255}, -1);

            for (int i = 0; i < s.num_parts(); i++) {
                cv::circle(overlay, toCvPoint(s.part(i)), 1, {255,255,0}, -1);
            }
        }

        cv::imwrite((results_dir / ("overlay_" + img_name)).string(), overlay);
        img_count++;
    }

    double batch_total_ms = t_batch.ms();

    // run summary
    {
        std::ofstream summary(results_dir / "run_summary.csv");
        summary << "mode,threads,images,preload_ms,batch_compute_ms,batch_total_ms\n";
        summary << "serial,1," << img_count << ",,,"
                << std::fixed << std::setprecision(2) << batch_total_ms << "\n";
    }

    std::cout << "Done. Processed " << img_count << " images.\n"
              << "Wrote:\n  " << (results_dir / "serial_times.csv") << "\n  "
              << (results_dir / "serial_compute_times.csv") << "\n  "
              << (results_dir / "landmarks.csv") << "\n  "
              << (results_dir / "run_summary.csv") << "\n";

    return 0;
}
