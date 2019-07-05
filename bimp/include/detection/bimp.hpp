#ifndef BIMP_H
#define BIMP_H

#include "opencv2/opencv.hpp"
#include "cuda_bimp.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/xfeatures2d/cuda.hpp>
#include "detection/utils.hpp"

namespace bimp {
    using cv::cuda::GpuMat;
    using cv::Mat;
    using cv::Ptr;
    using cv::xfeatures2d::SURF;

    class Context
    {
    private:
        bool cpu_mode;
    public:
	bool resize_cpu;
        bool has_changed_keypoints, has_changed_descriptors;
        bool usingBIMP;
        cv::cuda::GpuMat current_image, current_image_color;
        float *raw_cuda_data;
        bimp::cuda::CudaStuff cuda_data;
        cv::cuda::GpuMat current_keypoints;
        cv::cuda::GpuMat current_descriptors, surf_descriptors;
        std::vector<float> lambdas;
        cv::cuda::SURF_CUDA surf;

        Context(cv::Mat &image);
        cv::cuda::GpuMat& getKeypoints();
        cv::cuda::GpuMat& getDescriptors();
        std::vector<cv::KeyPoint> getKeypointsAndDownload();
        cv::Mat getDescriptorsAndDownload();

	std::vector<bimp::cuda::CudaKernels> getResponses();

        void loadImage(cv::Mat new_image);
        void loadImage(cv::cuda::GpuMat new_image);
        void setCPUMode(bool cpu_mode);
        int width();
        int height();
        bool getCPUMode();
    };

    void showResized(cv::Mat &mat, float sf, bool showImage = true);
    void showResized(cv::cuda::GpuMat &mat, float sf, bool showImage = true);

  cv::Mat getMinSized(cv::Mat &mat);
  std::vector<cv::DMatch> matchContextsAndDownload(Context &c1, Context &c2, bool use_own, bool use_cpu);
  cv::cuda::GpuMat matchContexts(Context &c1, Context &c2, bool use_own, bool use_cpu);
  std::vector<cv::DMatch> downloadMatches(cv::cuda::GpuMat matches);

}

#endif // BIMP_H
