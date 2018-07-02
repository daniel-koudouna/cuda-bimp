#ifndef CUDA_BIMP_H
#define CUDA_BIMP_H

#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#if CUDA_VERSION >= 5000
#include <helper_cuda.h>
#else
#include <cutil_inline.h>
#define getLastCudaError cutilCheckMsg
#define checkCudaErrors cutilSafeCall
#endif /*CUDA_VERSION >= 5000*/
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#define NUM_ORI 8
#define SUBPIXEL true

#define KP_VECTOR_LENGTH 7
#define KP_MAX_POINTS 8000
#define SURF_MIN_SIZE 200.0

#define SURF_DIMENSIONS 128
#define COLOR_DIMENSIONS 24
#define DIMENSIONS (SURF_DIMENSIONS + COLOR_DIMENSIONS)

#ifndef FCOMPLEX_H
#define FCOMPLEX_H
#ifdef __CUDACC__
    typedef float2 fComplex;
#else
    typedef struct{
        float x;
        float y;
    } fComplex;
#endif
#endif
namespace bimp
{

    namespace cuda {

	// These are pointers to CUDA structures which change with each scale, and must be kept on the GPU between frames
		typedef struct _cudakernels {
			float lambda, lambda2, pyrstep;

			float *h_Data, *d_Data, *d_PaddedData;

			cv::cuda::GpuMat d_DataMat;

			fComplex *d_DataSpectrum;

			float *h_ResultDouble, *h_ResultSingle, *h_ResultLines, *h_ResultOri;
			char *h_ResultType;

			//Results stored on the GPU
			float *d_ResultDouble, *d_ResultSingle, *d_ResultLines, *d_ResultOri;
		    float *d_ResultComplex[NUM_ORI];
		    int numOri;
		    

			// Odd and even simple cell kernels
			float *h_Kernel_e, *d_Kernel_e, *d_PaddedKernel_e;
			float *h_Kernel_o, *d_Kernel_o, *d_PaddedKernel_o;

			fComplex *d_KernelSpectrum_e[NUM_ORI], *d_KernelSpectrum_o[NUM_ORI];

			cufftHandle fftPlanFwd, fftPlanInv;

			int kernelH; //= 11;
			int kernelW; //= 11;
			int kernelY; //= 5;
			int kernelX; //= 5;
			int dataH; //= tempframe.rows;
			int dataW; //= tempframe.cols;

			int fftH; //= snapTransformSize(dataH + kernelH - 1);
			int fftW; //= snapTransformSize(dataW + kernelW - 1);
		} CudaKernels;

		// These are pointers to CUDA structures which can be reused between scales and between frames. That's why we allocate them
		// only once, for largest resolution, cutting the needed memory by a lot
		typedef struct _cudaimage {
			int *d_k_count;

			// Intermediate storage in spatial domain
			float *d_simple_o[NUM_ORI], *d_simple_e[NUM_ORI], *d_complex[NUM_ORI];
			float *d_double, *d_single, *d_tan_in, *d_rad_in, *d_lat_in, *d_cro_in, *d_gauss, *d_ch, *d_lines, *d_ori;
			char *d_type;

			fComplex *d_DataSpectrum_1, *d_DataSpectrum_2, *d_complexSpectrum[NUM_ORI];
			fComplex *d_doubleSpectrum, *d_singleSpectrum, *d_inhibSpectrum;

			bool bRetVal;
			unsigned int hTimer;
		} CudaImage;

		class CudaStuff {
		public:
			std::vector<CudaKernels> ks;
			CudaImage im;

			CudaStuff(int width, int height, std::vector<float> lambdas);

			~CudaStuff();
		};

		/**
         * Extracts the keypoints from an image for further processing.
         * If not already on the GPU, the image is uploaded to the correct format.
         * The output matrix can also be specified.
         */
	    cv::cuda::GpuMat getKeypoints(cv::Mat &input, CudaStuff &cs);
	    
	    cv::cuda::GpuMat getKeypoints(cv::cuda::GpuMat &input, CudaStuff &cs, bool cpu_mode = false, bool resize_cpu = false);
	    
	    int getKeypoints(cv::cuda::GpuMat &input, cv::cuda::GpuMat &output, CudaStuff &cs, bool cpu_mode = false, bool resize_cpu = false);

		/**
         * Uploads an image from the CPU to the GPU, and converts it into
         * the appropriate format.
         */
		cv::cuda::GpuMat getGPUImage(cv::Mat &input, bool grayscale = true);

		/**
         * Draws the keypoints collected on an image.
         */
		void drawGPUKeypoints(cv::Mat &input, cv::cuda::GpuMat &res, cv::Mat &output, int count, cv::Scalar col,
							  unsigned int flags);

		/**
         * The main filtering function. Applies all the kernels to all the scales of
         * the image and extracts the keypoints, which are then stored on the destination
         * GPU image.
         */
		void DoFullGPUFiltering(CudaStuff &cs, cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);

	    void setupCUDAImages(CudaStuff &cs, cv::cuda::GpuMat intensity, bool resize_cpu);

		void cuda_gpu_keypoints(CudaKernels &k, cv::cuda::GpuMat &result, int *kpts);

		void cuda_dilate(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);

		int snapTransformSize(int dataSize);

		void DoGPUFiltering(CudaStuff &cs);

		std::vector<cv::KeyPoint> ExtractCUDAKeypoints(CudaStuff &cs);

		std::vector<float> makeLambdasLog(float lambda_start, float lambda_end, int scalesPerOctave);

		float parabolaPeak(float val1, float val2, float val3, float pos1 = -1, float pos2 = 0, float pos3 = 1);

		std::vector<cv::KeyPoint> cuda_keypoints(cv::Mat_<float> KD, cv::Mat_<float> KS, float lambda, float pyrstep);

		extern "C" void complexResponse(
				float *d_simple_e,
				float *d_simple_o,
				float *d_complex,
				int fftW, int fftH);

		extern "C" void clearfloatarray(float *d_in, int fftW, int fftH);

		extern "C" void sumArray(
				float *d_arg1,
				float *d_arg2,
				int fftW,
				int fftH,
				int stride);

		extern "C" void FilterSimpleCells(
				fComplex *result_e,
				fComplex *result_o,
				fComplex *kernel_e,
				fComplex *kernel_o,
				fComplex *data,
				int fftW,
				int fftH,
				cudaStream_t stream = 0);

		extern "C" void endStoppedResponse(
				float *d_double,
				float *d_single,
				float *d_complex,
				int offset1,
				int offset2,
				int dataW,
				int dataH,
				int stride,
				cudaStream_t stream = 0);

		extern "C" void inhibitionResponse(
				float *d_tan_in,
				float *d_rad_in,
				float *d_complex,
				float *d_complex2,
				int offset1,
				int offset2,
				int dataW,
				int dataH,
				int stride,
				cudaStream_t stream = 0);

		extern "C" void inhibitKeypoints(float *d_double, float *d_single, float *d_tan_in, float *d_rad_in,
										 int dataW, int dataH, int stride);

		extern "C" void collectCVMemory(
				cv::cuda::PtrStepSz<float> cv_image,
				float *d_result,
				int dataW,
				int dataH);

		extern "C" void thrustFindMax(float *src, int size, float *max);

		extern "C" void identifyKeypoints(
				cv::cuda::PtrStepSz<float> result,
				cv::cuda::PtrStepSz<int> resultInts,
				cv::cuda::PtrStepSz<float> keypts,
				cv::cuda::PtrStepSz<float> kLoc,
				float lambda,
				int off,
				int rowMax,
				int colMax,
				float pyrstep,
				float thresh,
				int dataSize,
				int *counter);

		extern "C" void copy(
				cv::cuda::PtrStepSz<float> src,
				cv::cuda::PtrStepSz<float> dest);

		extern "C" void addColorFeatures(
				cv::cuda::PtrStepSz<float> keypoints,
				cv::cuda::PtrStepSz<uchar3> color_image,
				cv::cuda::PtrStepSz<float> descriptors);

	} // namespace cuda
} // namespace bimp

#endif // CUDA_BIMP_H

