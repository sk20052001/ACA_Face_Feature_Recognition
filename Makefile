CXX      := nvcc
TARGET   := cuda_face
SRC      := cuda_parallel.cu

OPENCV_INC := /usr/local/include/opencv4
OPENCV_LIB := /usr/local/lib
BLAS_LIB   := /usr/lib/x86_64-linux-gnu

CXXFLAGS := -std=c++17 -I$(OPENCV_INC)
LDFLAGS  := -L$(OPENCV_LIB) -L$(BLAS_LIB) \
            -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
            -lopencv_cudaimgproc -lopencv_objdetect \
            -ldlib -lopenblas -llapack

MODEL    := models/shape_predictor_68_face_landmarks.dat
CASCADE  := haarcascade_frontalface_default.xml
IMAGES   := test_images
RESULTS  := results

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	LD_LIBRARY_PATH=$(OPENCV_LIB):$$LD_LIBRARY_PATH \
	./$(TARGET) $(MODEL) $(IMAGES) $(RESULTS)

clean:
	rm -f $(TARGET)
	rm -rf $(RESULTS)
