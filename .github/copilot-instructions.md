# Copilot Instructions

## Project Overview

CUDA-accelerated face feature recognition pipeline written in CUDA C++ (`cuda_parallel.cu`). It detects faces using OpenCV's GPU Haar cascade classifier and extracts 68 facial landmarks using dlib's shape predictor, processing images in parallel across 4 CUDA streams.

## Build Command

```bash
nvcc cuda_parallel.cu -std=c++17 -o cuda_face \
  -I/usr/local/include/opencv4 \
  -L/usr/local/lib \
  -L/usr/lib/x86_64-linux-gnu \
  -lopencv_core \
  -lopencv_imgproc \
  -lopencv_imgcodecs \
  -lopencv_cudaimgproc \
  -lopencv_objdetect \
  -ldlib \
  -lopenblas \
  -llapack
```

## Run Command

```bash
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
./cuda_face models/shape_predictor_68_face_landmarks.dat <images_dir> <results_dir>
```

## Architecture

The pipeline in `processImageGPU()` follows this flow per image:
1. **GPU upload** → `cv::cuda::GpuMat`
2. **GPU grayscale** → `cv::cuda::cvtColor` with a CUDA stream
3. **GPU face detection** → `cv::cuda::CascadeClassifier` (Haar cascade: `haarcascade_frontalface_default.xml`)
4. **CPU landmark detection** → dlib `shape_predictor` with the 68-point model (`models/shape_predictor_68_face_landmarks.dat`)
5. **Outputs**:
   - `results/overlay_<image>.jpg` — annotated image with landmarks drawn
   - `results/landmarks.csv` — per-face landmark coordinates (left eye, right eye, nose, mouth corners)
   - `results/cuda_times.csv` — per-image face count

Parallelism is achieved via 4 CUDA streams (`STREAMS = 4`), cycling images round-robin across streams.

## Key Conventions

- **Dlib landmark indices**: Eyes use indices 36–41 (left) and 42–47 (right); nose tip is 30; mouth corners are 48 and 54.
- **CUDA stream type**: Use `cv::cuda::Stream` (not raw `cudaStream_t`) for all OpenCV CUDA APIs. `StreamAccessor::wrapStream()` returns a temporary that cannot bind to `cv::cuda::Stream&`.
- **cv::cuda::CascadeClassifier broken**: OpenCV 4.14.0-pre's NCV XML parser (`loadFromXML`) fails on standard Haar cascade XMLs. Use `cv::CascadeClassifier` (CPU) on the GPU-downloaded grayscale image instead.
- **dlib include**: Use `<dlib/image_processing/shape_predictor.h>` only — `<dlib/image_processing.h>` pulls in `scan_image_pyramid_tools.h` which fails to compile under C++17 due to a missing `typename` keyword.
- **Link flags**: dlib requires `-lopenblas -llapack` from `/usr/lib/x86_64-linux-gnu`; OpenCV objdetect for `cv::CascadeClassifier` requires `-lopencv_objdetect`.
- **Runtime library path**: OpenCV is installed to `/usr/local/lib` which is not in the default search path; run with `LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`.
