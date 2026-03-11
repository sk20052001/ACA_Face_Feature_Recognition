# ACA_Face_Feature_Recognition

GPU-accelerated face feature recognition pipeline using CUDA, OpenCV, and dlib.

## Build

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

## Run

```bash
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
./cuda_face models/shape_predictor_68_face_landmarks.dat <images_dir> <results_dir>
```

## Outputs

- `<results_dir>/overlay_<image>.jpg` — annotated image with 68 landmarks drawn
- `<results_dir>/landmarks.csv` — per-face landmark coordinates
- `<results_dir>/cuda_times.csv` — per-image face count
