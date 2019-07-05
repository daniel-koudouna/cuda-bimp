#include <cstdlib>
#include "opencv2/opencv.hpp"
#include <thrust/device_ptr.h>
#include <detection/cuda_bimp.hpp>
#include "detection/cuda_bimp.hpp"
#include "detection/convolutionFFT2D_common.h"
#include "detection/util.h"
#include "detection/utils.hpp"

namespace bimp
{
    namespace cuda {

        cudaStream_t* initOrientationStreams()
        {
            cudaStream_t *res = new cudaStream_t[NUM_ORI];
            for (int i = 0; i < NUM_ORI; i++) {
              cudaStreamCreate(&res[i]);
            }
            return res;
        }

        cudaStream_t *orientationStreams = initOrientationStreams();


        cv::Ptr<cv::cuda::Filter> initializeFilter() {
            int dilationSize = 3;
            cv::Mat dilationEl = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilationSize, dilationSize));
            dilationEl.ptr(1)[1] = 0;
            cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_32F, dilationEl);

            cv::cuda::GpuMat temp(cv::Size(10, 10), CV_32F);
            dilateFilter->apply(temp, temp);

            return dilateFilter;
        }

        cv::Ptr<cv::cuda::Filter> DILATE_FILTER = initializeFilter();

        void checkExtremeValues(int height, int width, float *mat) {
            int count = 0;
            for (int j = 0; j < height / 5; ++j) {
                for (int i = 0; i < width; i++) {
                    float f = mat[j * width + i];

                    if (f > 100 || f != f) {
                        std::cout << "!";
                        count++;
                    } else if (f > 10) {
                        std::cout << "?";
                        count++;
                    } else {
                        std::cout << ".";
                    }
                }
                std::cout << std::endl;
            }

            std::cout << "found " << count << " junk values on " << width << " , " << height << std::endl;
        }

        int getKeypoints(cv::cuda::GpuMat &input, cv::cuda::GpuMat &output, bimp::cuda::CudaStuff &cs, bool cpu_mode, bool resize_cpu) {
            int rows = 0;
            output.setTo(cv::Scalar(0));

            if (!cpu_mode) {
                DoFullGPUFiltering(cs, input, output);
                cudaDeviceSynchronize();
                rows = *cs.im.d_k_count;
            } else {
                setupCUDAImages(cs, input, resize_cpu);
                DoGPUFiltering(cs);
                std::vector<cv::KeyPoint> pts = ExtractCUDAKeypoints(cs);
                std::cout << "pts size is " << pts.size() << std::endl;
                rows = pts.size();
                bimp::utils::uploadKeypoints(pts, output);
            }

            return rows;
        }

        cv::cuda::GpuMat getKeypoints(cv::cuda::GpuMat &input, CudaStuff &cs, bool cpu_mode, bool cpu_resize) {
            cv::cuda::GpuMat result(KP_VECTOR_LENGTH, KP_MAX_POINTS, CV_32F);
            int rows = getKeypoints(input, result, cs, cpu_mode, cpu_resize);
            if (rows > KP_MAX_POINTS)
            {
                cerr << "Warning: Keypoints (" << rows << ") exceed current keypoint maximum." << endl
                     << "This may be caused if your training images are high-resolution." << endl
                     << "Consider increasing the keypoint limit defined in cuda_bimp.hpp" << endl;
                rows = KP_MAX_POINTS;
            }
            return result.colRange(cv::Range(0, rows));
        }

        cv::cuda::GpuMat getKeypoints(cv::Mat &input, CudaStuff &cs) {
            cv::cuda::GpuMat d_input = getGPUImage(input);
            return getKeypoints(d_input, cs);
        }


        void drawGPUKeypoints(cv::Mat &input, cv::cuda::GpuMat &res, cv::Mat &output, int count, cv::Scalar col,
                              unsigned int flags) {

            std::vector<cv::KeyPoint> points = bimp::utils::downloadKeypoints(res);

            // TODO ignore flags
            cv::drawKeypoints(input, points, output, col, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        }

        cv::cuda::GpuMat getGPUImage(cv::Mat &in, bool grayscale) {
            cv::Mat input;
            int MIN_SIZE = 350;

            if (in.rows < MIN_SIZE && in.cols >= MIN_SIZE) {
                float sf = MIN_SIZE / (1.0 * in.rows);
                cv::resize(in, input, cv::Size(), sf, sf);
            } else if (in.cols < MIN_SIZE) {
                float sf = MIN_SIZE / (1.0 * in.cols);
                cv::resize(in, input, cv::Size(), sf, sf);
            } else {
                input = in;
            }

            cv::cuda::GpuMat d_input(input.size(), 0);
            if (input.type() > 0 && grayscale) {
                cv::cuda::GpuMat d_image(input.size(), input.type());
                d_image.upload(input);
                cv::cuda::cvtColor(d_image, d_input, cv::COLOR_BGR2GRAY);
            } else {
                d_input.upload(input);
            }

            return d_input;
        }

        void testKeypoints(cv::Mat &input, cv::Mat &output, CudaStuff &cs) {
            double t = (double)cv::getTickCount();

            cv::cuda::GpuMat d_input = getGPUImage(input);

            bimp::utils::log("Image upload", 0, &t);

            cv::cuda::GpuMat res = getKeypoints(d_input, cs);

            bimp::utils::log("Full keypoint extraction", 0, &t);

            int kptcount = (*cs.im.d_k_count);

            drawGPUKeypoints(input, res, output, kptcount, cv::Scalar(255, 255, 255), 4);

        }

        int snapTransformSize(int dataSize) {

            int hiBit;
            unsigned int lowPOT, hiPOT;

            dataSize = iAlignUp(dataSize, 16);

            for (hiBit = 31; hiBit >= 0; hiBit--)
                if (dataSize & (1U << hiBit)) break;

            lowPOT = 1U << hiBit;
            if (lowPOT == (unsigned int) dataSize)
                return dataSize;

            hiPOT = 1U << (hiBit + 1);
            if (hiPOT <= 1024)
                return hiPOT;
            else
                return iAlignUp(dataSize, 512);
        }


        CudaStuff::CudaStuff(int width, int height, std::vector<float> lambdas) {

            std::sort(lambdas.begin(), lambdas.end());

            // Allocate the memory for kernels at different lambdas which stay on the GPU all the time
            for (int i = 0; i < lambdas.size(); i++) {
                // std::cout << "lambda " << lambdas[i] << std::endl;
                CudaKernels k;

                k.pyrstep = 1;
		k.numOri = NUM_ORI;

                k.lambda2 = lambdas[i];

                while (k.lambda2 / k.pyrstep >= 7.5) {
		    k.pyrstep *= 2;
		}
                k.lambda = k.lambda2 / k.pyrstep;

                float gamma = 0.5;
                float phi = -CV_PI / 2;
                float sigma = k.lambda * 0.56;
                float sigmasq = sigma * sigma;
                float filtersize = 7 * k.lambda; //k.kernelW;
                if (((int) filtersize) % 2 == 0) filtersize += 1;
                float d = 0.6 * k.lambda;

                k.kernelX = k.kernelY = floor(filtersize / 2);
                k.kernelW = k.kernelH = filtersize;

                // Do CUDA initialisations
                k.dataH = height / k.pyrstep;
                k.dataW = width / k.pyrstep;
                k.fftH = snapTransformSize(k.dataH); // + k.kernelH - 1);
                k.fftW = snapTransformSize(k.dataW); // + k.kernelW - 1);

                //printf("...allocating memory for lambda %f\n",k.lambda*k.pyrstep);
                //std::cout << "FFT size: " << k.fftW << " x " << k.fftH << std::endl;

                // Data
                cudaMalloc((void **) &k.d_Data, k.dataH * k.dataW * sizeof(float));
                cudaMalloc((void **) &k.d_PaddedData, k.fftH * k.fftW * sizeof(float));
                cudaMalloc((void **) &k.d_DataSpectrum, k.fftH * (k.fftW / 2 + 1) * sizeof(fComplex));
                // Simple cells
                k.h_Kernel_e = (float *) malloc(k.kernelH * k.kernelW * sizeof(float));
                k.h_Kernel_o = (float *) malloc(k.kernelH * k.kernelW * sizeof(float));

                cudaMalloc((void **) &k.d_Kernel_e, k.kernelH * k.kernelW * sizeof(float));
                cudaMalloc((void **) &k.d_Kernel_o, k.kernelH * k.kernelW * sizeof(float));

                cudaMalloc((void **) &k.d_PaddedKernel_e, k.fftH * k.fftW * sizeof(float));
                cudaMalloc((void **) &k.d_PaddedKernel_o, k.fftH * k.fftW * sizeof(float));

                k.h_ResultDouble = (float *) malloc(k.dataH * k.dataW * sizeof(float));
                k.h_ResultSingle = (float *) malloc(k.dataH * k.dataW * sizeof(float));
                k.h_ResultLines = (float *) malloc(k.dataH * k.dataW * sizeof(float));
                k.h_ResultOri = (float *) malloc(k.dataH * k.dataW * sizeof(float));
                k.h_ResultType = (char *) malloc(k.dataH * k.dataW * sizeof(char));

                cudaMalloc((void **) &k.d_ResultDouble, k.dataH * k.dataW * sizeof(float));
                cudaMalloc((void **) &k.d_ResultLines, k.dataH * k.dataW * sizeof(float));
                cudaMalloc((void **) &k.d_ResultOri, k.dataH * k.dataW * sizeof(float));
                cudaMalloc((void **) &k.d_ResultSingle, k.dataH * k.dataW * sizeof(float));

                for (int o = 0; o < NUM_ORI; o++) {
                    // Simple cells
                    cudaMalloc((void **) &k.d_KernelSpectrum_e[o],
                                               k.fftH * (k.fftW / 2 + 1) * sizeof(fComplex));
                    cudaMalloc((void **) &k.d_KernelSpectrum_o[o],
                                               k.fftH * (k.fftW / 2 + 1) * sizeof(fComplex));

		    cudaMalloc((void **) &k.d_ResultComplex[o], k.fftH * k.fftW * sizeof(float));
		}

                cufftPlan2d(&k.fftPlanFwd, k.fftH, k.fftW, CUFFT_R2C);
                cufftPlan2d(&k.fftPlanInv, k.fftH, k.fftW, CUFFT_C2R);

                // Now create all the wavelets we will be using. Since they won't change, precalculate them here and transfer them
                // to the GPU

                //printf("...generating the kernels...\n");
                // Prepare the kernels they  will not change much
                for (int o = 0; o < NUM_ORI; o++) {
                    float theta = o * CV_PI / NUM_ORI;
                    float sintheta = sin(theta);
                    float costheta = cos(theta);
                    float dsintheta = d * sintheta;
                    float dcostheta = d * costheta;
                    float evensum = 0;

                    // create the wavelet we will be using
                    for (int row = 0; row < k.kernelH; row++) {
                        for (int col = 0; col < k.kernelW; col++) {
                            int x = col - k.kernelX;
                            int y = row - k.kernelY;

                            float env = exp(
                                    -1 / (2 * sigmasq) * ((x * costheta + y * sintheta) * (x * costheta + y * sintheta)
                                                          + gamma * (y * costheta - x * sintheta) *
                                                            (y * costheta - x * sintheta)));
                            float freqfactor = 2 * CV_PI * (x * costheta + y * sintheta) / k.lambda;

                            k.h_Kernel_e[row * k.kernelW + col] = env * cos(freqfactor);
                            k.h_Kernel_o[row * k.kernelW + col] = env * cos(freqfactor + phi);
                            evensum += k.h_Kernel_e[row * k.kernelW + col];
                        }
                    }
                    // Remove the DC component of the even kernel
                    for (int m = 0; m < k.kernelH * k.kernelW; m++)
                        k.h_Kernel_e[m] -= evensum / (filtersize * filtersize);

                    // Transfer the new kernels to the GPU device and pad them to POT
                    cudaMemcpy(k.d_Kernel_e, k.h_Kernel_e,
                                               k.kernelH * k.kernelW * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(k.d_Kernel_o, k.h_Kernel_o,
                                               k.kernelH * k.kernelW * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemset(k.d_PaddedKernel_e, 0, k.fftH * k.fftW * sizeof(float));
                    cudaMemset(k.d_PaddedKernel_o, 0, k.fftH * k.fftW * sizeof(float));
                    padKernel(k.d_PaddedKernel_e, k.d_Kernel_e, k.fftH, k.fftW, k.kernelH, k.kernelW, k.kernelY,
                              k.kernelX);
                    padKernel(k.d_PaddedKernel_o, k.d_Kernel_o, k.fftH, k.fftW, k.kernelH, k.kernelW, k.kernelY,
                              k.kernelX);

                    // calculate FFT of the kernels on the GPU device
                    cufftExecR2C(k.fftPlanFwd, (cufftReal *) k.d_PaddedKernel_e,
                                                                        (cufftComplex *) k.d_KernelSpectrum_e[o]);
                    cufftExecR2C(k.fftPlanFwd, (cufftReal *) k.d_PaddedKernel_o,
                                                                        (cufftComplex *) k.d_KernelSpectrum_o[o]);

                }


                (*this).ks.push_back(k);
            }

            cudaMallocManaged(((void **) &(this->im.d_k_count)), sizeof(int));

            // Allocate the memory for shared objects (frame information and reusable buffers)
            CudaKernels &k0 = this->ks[0];

                    cudaMalloc((void **) &(this->im.d_doubleSpectrum), k0.fftH * (k0.fftW / 2 + 1) * sizeof(fComplex));

                    cudaMalloc((void **) &(this->im.d_singleSpectrum), k0.fftH * (k0.fftW / 2 + 1) * sizeof(fComplex));

                    cudaMalloc((void **) &(this->im.d_inhibSpectrum), k0.fftH * (k0.fftW / 2 + 1) * sizeof(fComplex));


                    cudaMalloc((void **) &(this->im.d_DataSpectrum_1), k0.fftH * (k0.fftW / 2 + 1) * sizeof(fComplex));

                    cudaMalloc((void **) &(this->im.d_DataSpectrum_2), k0.fftH * (k0.fftW / 2 + 1) * sizeof(fComplex));

            cudaMalloc((void **) &(this->im.d_double), k0.dataH * k0.dataW * sizeof(float));
            cudaMalloc((void **) &(this->im.d_single), k0.dataH * k0.dataW * sizeof(float));
            cudaMalloc((void **) &(this->im.d_tan_in), k0.dataH * k0.dataW * sizeof(float));
            cudaMalloc((void **) &(this->im.d_rad_in), k0.dataH * k0.dataW * sizeof(float));
            cudaMalloc((void **) &(this->im.d_lat_in), k0.dataH * k0.dataW * sizeof(float));
            cudaMalloc((void **) &(this->im.d_cro_in), k0.dataH * k0.dataW * sizeof(float));
            cudaMalloc((void **) &(this->im.d_gauss), k0.dataH * k0.dataW * sizeof(float));
            cudaMalloc((void **) &(this->im.d_ch), k0.dataH * k0.dataW * sizeof(float));
            cudaMalloc((void **) &(this->im.d_lines), k0.dataH * k0.dataW * sizeof(float));
            cudaMalloc((void **) &(this->im.d_ori), k0.dataH * k0.dataW * sizeof(float));
            cudaMalloc((void **) &(this->im.d_type), k0.dataH * k0.dataW * sizeof(char));

            for (int o = 0; o < NUM_ORI; o++) {
                // Simple cells
                cudaMalloc((void **) &(this->im.d_simple_e[o]), k0.fftH * k0.fftW * sizeof(float));
                cudaMalloc((void **) &(this->im.d_simple_o[o]), k0.fftH * k0.fftW * sizeof(float));

                // Complex cells
                cudaMalloc((void **) &(this->im.d_complex[o]), k0.fftH * k0.fftW * sizeof(float));
                cudaMalloc((void **) &(this->im.d_complexSpectrum[o]),
                                           k0.fftH * (k0.fftW / 2 + 1) * sizeof(fComplex));
            }

        }

        CudaStuff::~CudaStuff() {
            cudaFree(this->im.d_double);
            cudaFree(this->im.d_single);
            cudaFree(this->im.d_tan_in);
            cudaFree(this->im.d_rad_in);
            cudaFree(this->im.d_lat_in);
            cudaFree(this->im.d_cro_in);
            cudaFree(this->im.d_gauss);
            cudaFree(this->im.d_ch);
            cudaFree(this->im.d_lines);
            cudaFree(this->im.d_ori);
            cudaFree(this->im.d_type);
            cudaFree(this->im.d_singleSpectrum);
            cudaFree(this->im.d_doubleSpectrum);
            cudaFree(this->im.d_inhibSpectrum);

            cudaFree(this->im.d_DataSpectrum_1);
            cudaFree(this->im.d_DataSpectrum_2);

            cudaFree(this->im.d_k_count);

            for (int o = 0; o < NUM_ORI; o++) {
                cudaFree(this->im.d_simple_e[o]);
                cudaFree(this->im.d_simple_o[o]);

                cudaFree(this->im.d_complex[o]);
                cudaFree(this->im.d_complexSpectrum[o]);
            }

            for (int i = 0; i < this->ks.size(); i++) {
                CudaKernels &k = this->ks[i];

                cudaFree(k.d_PaddedData);
                cudaFree(k.d_Data);
                cudaFree(k.d_DataSpectrum);

                cudaFree(k.d_ResultSingle);
                cudaFree(k.d_ResultDouble);
                cudaFree(k.d_ResultOri);
                cudaFree(k.d_ResultLines);

                // Clean up CUDA stuff
                cufftDestroy(k.fftPlanInv);
                cufftDestroy(k.fftPlanFwd);

                cudaFree(k.d_PaddedKernel_e);
                cudaFree(k.d_Kernel_e);

                cudaFree(k.d_PaddedKernel_o);
                cudaFree(k.d_Kernel_o);

                for (int o = 0; o < NUM_ORI; o++) {
                    cudaFree(k.d_KernelSpectrum_e[o]);
                    cudaFree(k.d_KernelSpectrum_o[o]);
                    cudaFree(k.d_ResultComplex[o]);
                }

                free(k.h_Kernel_e);
                free(k.h_Kernel_o);
            }

        }

        void DoFullGPUFiltering(CudaStuff &cs, cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst) {

            *cs.im.d_k_count = 0;

            for (int i = 0; i < cs.ks.size(); i++) {

                double t = (double) cv::getTickCount();
                double then = (double) cv::getTickCount();
                CudaKernels &k = cs.ks[i];

                float pyrstep = cs.ks[i].pyrstep;
                cv::cuda::GpuMat downscaled = src.clone();

                while (pyrstep > 1) {
                    cv::cuda::pyrDown(downscaled, downscaled);
                    pyrstep /= 2;

                }

                cv::cuda::GpuMat data(k.dataH, k.dataW, CV_32FC1, k.d_Data);
                downscaled.convertTo(data, CV_32FC1, 255.0 / 65536.0);

                bimp::utils::log("Pyramid downscaling", 5, &t);

                k.d_DataMat = data;
                collectCVMemory(k.d_DataMat, k.d_Data, k.dataW, k.dataH);

                bimp::utils::log("CV to CUDA conversion", 5, &t);

                // transfer image and kernel data to CPU
                cudaMemset(k.d_PaddedData, 0, k.fftH * k.fftW * sizeof(float));

                padDataClampToBorder(k.d_PaddedData, k.d_Data, k.fftH, k.fftW, k.dataH, k.dataW, k.kernelH, k.kernelW,
                                     k.kernelY, k.kernelX);

                // obtain the frequency spectrum of the image

                cufftExecR2C(k.fftPlanFwd, (cufftReal *) k.d_PaddedData,
                                                                    (cufftComplex *) k.d_DataSpectrum);

                clearfloatarray(cs.im.d_ch, k.dataW, k.dataH);

                // Filter the image in the frequency domain and obtain complex cell responses
                for (int o = 0; o < NUM_ORI; o++) {
                    FilterSimpleCells(cs.im.d_DataSpectrum_1, cs.im.d_DataSpectrum_2,
                                      k.d_KernelSpectrum_e[o], k.d_KernelSpectrum_o[o],
                                      k.d_DataSpectrum, k.fftW, k.fftH, orientationStreams[o]);

                    // Transfer the simple cell responses back to the spatial domain

                            cufftExecC2R(k.fftPlanInv, (cufftComplex *) cs.im.d_DataSpectrum_1,
                                                                (cufftReal *) cs.im.d_simple_e[o]);

                            cufftExecC2R(k.fftPlanInv, (cufftComplex *) cs.im.d_DataSpectrum_2,
                                                                (cufftReal *) cs.im.d_simple_o[o]);

                    // Calculate the complex responses
                    complexResponse(cs.im.d_simple_e[o], cs.im.d_simple_o[o], cs.im.d_complex[o], k.fftW, k.fftH);

                    sumArray(cs.im.d_ch, cs.im.d_complex[o], k.dataW, k.dataH, k.fftW);

		    cudaMemcpy(k.d_ResultComplex[o], cs.im.d_complex[o], k.fftH * k.fftW * sizeof(float),
					       cudaMemcpyDeviceToHost);
                }

                // Second round of filtering -- get single and double cell responses, as well as tangential and radial inhibition
                // We do all of this in the spatial domain to avoid additional FFTs, which are comparatively expensive compared
                // to filtering with sparse kernels

                clearfloatarray(k.d_ResultSingle, k.dataW, k.dataH);
                clearfloatarray(k.d_ResultDouble, k.dataW, k.dataH);

                for (int o = 0; o < NUM_ORI; o++) {
                    float theta = o * CV_PI / NUM_ORI;
                    int offset1 = round(0.6 * k.lambda * cos(theta));
                    int offset2 = round(0.6 * k.lambda * sin(theta));
                    endStoppedResponse(k.d_ResultDouble, k.d_ResultSingle, cs.im.d_complex[o], offset1, offset2,
                                       k.dataW, k.dataH, k.fftW, orientationStreams[o]);
                }

                clearfloatarray(cs.im.d_tan_in, k.dataW, k.dataH);
                clearfloatarray(cs.im.d_rad_in, k.dataW, k.dataH);
                for (int o = 0; o < NUM_ORI; o++) {
                    float theta = o * CV_PI / NUM_ORI;
                    int o2 = (o + NUM_ORI / 2) % NUM_ORI;
                    int offset1 = round(0.6 * k.lambda * sin(theta));
                    int offset2 = round(0.6 * k.lambda * cos(theta));
                    inhibitionResponse(cs.im.d_tan_in, cs.im.d_rad_in, cs.im.d_complex[o], cs.im.d_complex[o2],
                                       offset1, offset2, k.dataW, k.dataH, k.fftW, orientationStreams[o]);
                }
                inhibitKeypoints(k.d_ResultDouble, k.d_ResultSingle, cs.im.d_tan_in, cs.im.d_rad_in, k.dataW, k.dataH,
                                 k.dataW);
                bimp::utils::log("GPU Filtering", 5, &t);
                cuda_gpu_keypoints(k, dst, cs.im.d_k_count);
                bimp::utils::log("Keypoint extraction kernel", 5, &t);
                bimp::utils::log("Scale time", 4, &then);
            }

        }

        void DoGPUFiltering(CudaStuff &cs) {
            for (int i = 0; i < cs.ks.size(); i++) {
                CudaKernels &k = cs.ks[i];


                // transfer image and kernel data to CPU
                cudaMemset(k.d_PaddedData, 0, k.fftH * k.fftW * sizeof(float));

                padDataClampToBorder(k.d_PaddedData, k.d_Data, k.fftH, k.fftW, k.dataH, k.dataW, k.kernelH, k.kernelW,
                                     k.kernelY, k.kernelX);

                // obtain the frequency spectrum of the image
                cufftExecR2C(k.fftPlanFwd, (cufftReal *) k.d_PaddedData,
                                                                    (cufftComplex *) k.d_DataSpectrum);

                clearfloatarray(cs.im.d_ch, k.dataW, k.dataH);

                // Filter the image in the frequency domain and obtain complex cell responses
                for (int o = 0; o < NUM_ORI; o++) {
                    FilterSimpleCells(cs.im.d_DataSpectrum_1, cs.im.d_DataSpectrum_2,
                                      k.d_KernelSpectrum_e[o], k.d_KernelSpectrum_o[o],
                                      k.d_DataSpectrum, k.fftW, k.fftH);

                    // Transfer the simple cell responses back to the spatial domain

                    cufftExecC2R(k.fftPlanInv, (cufftComplex *) cs.im.d_DataSpectrum_1,
                                                                (cufftReal *) cs.im.d_simple_e[o]);

                    cufftExecC2R(k.fftPlanInv, (cufftComplex *) cs.im.d_DataSpectrum_2,
                                                                (cufftReal *) cs.im.d_simple_o[o]);

                    // Calculate the complex responses
                    complexResponse(cs.im.d_simple_e[o], cs.im.d_simple_o[o], cs.im.d_complex[o], k.fftW, k.fftH);
                    sumArray(cs.im.d_ch, cs.im.d_complex[o], k.dataW, k.dataH, k.fftW);
                }

                // Second round of filtering -- get single and double cell responses, as well as tangential and radial inhibition
                // We do all of this in the spatial domain to avoid additional FFTs, which are comparatively expensive compared
                // to filtering with sparse kernels
                clearfloatarray(k.d_ResultSingle, k.dataW, k.dataH);
                clearfloatarray(k.d_ResultDouble, k.dataW, k.dataH);
                for (int o = 0; o < NUM_ORI; o++) {
                    float theta = o * CV_PI / NUM_ORI;
                    int offset1 = round(0.6 * k.lambda * cos(theta));
                    int offset2 = round(0.6 * k.lambda * sin(theta));
                    endStoppedResponse(k.d_ResultDouble, k.d_ResultSingle, cs.im.d_complex[o], offset1, offset2,
                                       k.dataW, k.dataH, k.fftW);
                }

                clearfloatarray(cs.im.d_tan_in, k.dataW, k.dataH);
                clearfloatarray(cs.im.d_rad_in, k.dataW, k.dataH);
                for (int o = 0; o < NUM_ORI; o++) {
                    float theta = o * CV_PI / NUM_ORI;
                    int o2 = (o + NUM_ORI / 2) % NUM_ORI;
                    int offset1 = round(0.6 * k.lambda * sin(theta));
                    int offset2 = round(0.6 * k.lambda * cos(theta));
                    inhibitionResponse(cs.im.d_tan_in, cs.im.d_rad_in, cs.im.d_complex[o], cs.im.d_complex[o2],
                                       offset1, offset2, k.dataW, k.dataH, k.fftW);
                }
                inhibitKeypoints(k.d_ResultDouble, k.d_ResultSingle, cs.im.d_tan_in, cs.im.d_rad_in, k.dataW, k.dataH,
                                 k.dataW);

                std::cout << "copying gpu to cpu memory" << std::endl;
                cudaMemcpy(k.h_ResultDouble, k.d_ResultDouble, k.dataH * k.dataW * sizeof(float),
                                           cudaMemcpyDeviceToHost);
                cudaMemcpy(k.h_ResultSingle, k.d_ResultSingle, k.dataH * k.dataW * sizeof(float),
                                           cudaMemcpyDeviceToHost);
                cudaMemcpy(k.h_ResultLines, k.d_ResultLines, k.dataH * k.dataW * sizeof(float),
                                           cudaMemcpyDeviceToHost);
                cudaMemcpy(k.h_ResultOri, k.d_ResultOri, k.dataH * k.dataW * sizeof(float),
                                           cudaMemcpyDeviceToHost);

            }

        }

        /**
         * Resize the image according to the lambdas and store
         * the results on the GPU, accessed via a cuda pointer.
         */
        void setupCUDAImages(CudaStuff &cs, cv::cuda::GpuMat intensity, bool resize_cpu) {
            for (int i = 0; i < cs.ks.size(); i++) {
                CudaKernels &k = cs.ks[i];

                float pyrstep = cs.ks[i].pyrstep;

		cv::cuda::GpuMat downscaled;

		if (resize_cpu) {

		    cv::Mat cIntensity;
		    intensity.download(cIntensity);

		    while (pyrstep > 1) {
			cv::pyrDown(cIntensity, cIntensity);
			pyrstep /= 2;
		    }

		    downscaled.upload(cIntensity);
		} else {

		    downscaled = intensity.clone();

		    while (pyrstep > 1) {
			cv::cuda::pyrDown(downscaled, downscaled);
			pyrstep /= 2;
		    }


		}

		cv::cuda::GpuMat data(k.dataH, k.dataW, CV_32FC1, k.d_Data);
		downscaled.convertTo(data, CV_32FC1, 255.0 / 65536.0);
                k.d_DataMat = data;
                collectCVMemory(k.d_DataMat, k.d_Data, k.dataW, k.dataH);


            }
        }

        inline float parabolaPeak(float val1, float val2, float val3, float pos1, float pos2, float pos3) {
            float result;

            cv::Mat_<float> xs(3, 3), ys(3, 1), factors(3, 1);
            ys(0, 0) = val1;
            ys(1, 0) = val2;
            ys(2, 0) = val3;

            xs(0, 0) = pos1 * pos1;
            xs(0, 1) = pos1;
            xs(0, 2) = 1;
            xs(1, 0) = pos2 * pos2;
            xs(1, 1) = pos2;
            xs(1, 2) = 1;
            xs(2, 0) = pos3 * pos3;
            xs(2, 1) = pos3;
            xs(2, 2) = 1;

            solve(xs, ys, factors);

            result = -factors(1, 0) / (factors(0, 0) * 2);

            return result;
        }

        void cuda_gpu_keypoints(CudaKernels &k, cv::cuda::GpuMat &result, int *kpts) {
            double t = (double) cv::getTickCount();
            double thresh;

            float max;

            thrustFindMax(k.d_ResultDouble, k.dataH * k.dataW, &max);
            thresh = max * 0.104;

            bimp::utils::log("Calculate max", 5, &t);

            //cuda::GpuMat kSingles(k.dataH, k.dataW, DataType<float>::type, k.d_ResultSingle);
            cv::cuda::GpuMat kDoubles(k.dataH, k.dataW, CV_32FC1, k.d_ResultDouble);

            cv::cuda::GpuMat kDoublesDil(kDoubles.size(), kDoubles.type());

            cuda_dilate(kDoubles, kDoublesDil);

            cv::cuda::GpuMat kDLoc;
            cv::cuda::subtract(kDoubles, kDoublesDil, kDLoc);

            bimp::utils::log("Prepare images", 5, &t);

            int off = k.lambda;

            identifyKeypoints(result, result, kDoubles, kDLoc, k.lambda, off, kDLoc.rows, kDLoc.cols, k.pyrstep, thresh,
                              k.dataW * k.dataH, kpts);

            bimp::utils::log("Keypoint kernel", 5, &t);
        }

        std::vector<cv::KeyPoint> cuda_keypoints(cv::Mat_<float> KD, cv::Mat_<float> KS, float lambda, float pyrstep) {

            // extract keypoints
            std::vector<cv::KeyPoint> points_s, points_d, points_all;

            cv::Mat_<float> KDdil(KD.size()), KSdil(KS.size()), KPSloc, KPDloc;

            cv::Matx<uchar, 3, 3> element(1, 1, 1, 1, 0, 1, 1, 1, 1);
            dilate(KD, KDdil, element);
            dilate(KS, KSdil, element);
            KPSloc = KS - KSdil;
            KPDloc = KD - KDdil;

            double mins, mind, maxs, maxd;
            double threshs, threshd;
            // ORIG: 0.04
            minMaxLoc(KS, &mins, &maxs);
            threshs = 0.104 * maxs;
            minMaxLoc(KD, &mind, &maxd);
            threshd = 0.104 * maxd;

            // find all the marked maxima and add the keypoints to the list
            // use an offset (=lambda) from image edges to avoid phantom keypoints caused by interference
            int off = lambda;

            float *dp, *sp, *firstp_d, *firstp_s, *secondp_d, *secondp_s, *locp_d, *locp_s;
            for (int row = off; row < KPDloc.rows - off; row++) {
                dp = KD.ptr<float>(row);
                firstp_d = KD.ptr<float>(row - 1);
                secondp_d = KD.ptr<float>(row + 1);

                sp = KS.ptr<float>(row);
                firstp_s = KS.ptr<float>(row - 1);
                secondp_s = KS.ptr<float>(row + 1);

                locp_d = KPDloc.ptr<float>(row);
                locp_s = KPSloc.ptr<float>(row);

                for (int col = off; col < KPDloc.cols - off; col++) {
                    float xpos = col, ypos = row;
                    if (locp_s[col] > 0 && sp[col] > threshd) {
                        xpos += parabolaPeak(sp[col - 1], sp[col], sp[col + 1]);
                        ypos += parabolaPeak(firstp_s[col], sp[col], secondp_s[col]);
                        cv::KeyPoint kp(cv::Point2f(xpos * pyrstep, ypos * pyrstep), lambda * pyrstep, -1, sp[col],
                                    lambda * pyrstep);
#pragma omp critical
                        points_s.push_back(kp);
                    } else locp_s[col] = 0;

                    if (locp_d[col] > 0 && dp[col] > threshd) {
                        xpos += parabolaPeak(dp[col - 1], dp[col], dp[col + 1]);
                        ypos += parabolaPeak(firstp_d[col], dp[col], secondp_d[col]);
                        cv::KeyPoint kp(cv::Point2f(xpos * pyrstep, ypos * pyrstep), lambda * pyrstep, -1, dp[col],
                                    lambda * pyrstep);
#pragma omp critical
                        points_d.push_back(kp);
                    } else locp_d[col] = 0;
                }
            }

            return points_d;

            int moved = 0;
            // Merge the single-stopped and double-stopped responses
            // #pragma omp parallel for schedule(dynamic,1) default(shared)
            // for(unsigned i=0; i<points_d.size(); i++)
            // {
            //     double mindist=100000;
            //     double x1 = points_d[i].pt.x;
            //     double y1 = points_d[i].pt.y;

            //     KeyPoint bestmatch;
            //     for(unsigned j=0; j<points_s.size(); j++)
            //     {
            //         double x2 = points_s[j].pt.x;
            //         double y2 = points_s[j].pt.y;

            //         double dist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
            //         if( dist < mindist )
            //         {
            //             mindist = dist;
            //             bestmatch = points_s[j];
            //         }
            //     }

            //     float contrs = KS( round(bestmatch.pt.y),   round(bestmatch.pt.x)   )/maxs;
            //     float contrd = KD( round(points_d[i].pt.y), round(points_d[i].pt.x) )/maxd;

            //     // We found the corresponding single-stopped cell, interpolate a new keypoint from the two
            //     if(mindist < lambda*lambda/4)
            //     {
            //         float cont1 = contrd/(contrs+contrd);
            //         float cont2 = contrs/(contrs+contrd);
            //         if(fabs(contrs-contrd)<1)
            //         {
            //             Point2f newloc;
            //             newloc.x = (points_d[i].pt.x*cont1 + bestmatch.pt.x*cont2 );
            //             newloc.y = (points_d[i].pt.y*cont1 + bestmatch.pt.y*cont2 );
            //             KeyPoint merged_pt( newloc, lambda, -1, points_d[i].response, lambda);
            //             // merged_pt.angle = atan2(y1-bestmatch.pt.y,x1-bestmatch.pt.x)*180/CV_PI;
            //             #pragma omp critical
            //             points_all.push_back(merged_pt);
            //             moved++;
            //         }
            //     }
            //     // No corresponding single-stopped cell, use the double-stopped one as is
            //     else
            //     {
            //         #pragma omp critical
            //         points_all.push_back(points_d[i]);
            //     }
            // }
            // return points_all;


        }

        std::vector<cv::KeyPoint> ExtractCUDAKeypoints(CudaStuff &cs) {
            std::vector<cv::KeyPoint> points;

            for (int i = 0; i < cs.ks.size(); i++) {
                CudaKernels &k = cs.ks[i];
                // Rect kpROI(0,0,k.dataW,k.dataH);
                cv::Mat singles = cv::Mat(k.dataH, k.dataW, cv::DataType<float>::type, k.h_ResultSingle);
                cv::Mat doubles = cv::Mat(k.dataH, k.dataW, cv::DataType<float>::type, k.h_ResultDouble);
                // std::vector<KeyPoint> tmppoints = cuda_keypoints( doubles(kpROI), singles(kpROI), k.lambda, k.pyrstep );
                std::vector<cv::KeyPoint> tmppoints = cuda_keypoints(doubles, singles, k.lambda, k.pyrstep);
                std::copy(tmppoints.begin(), tmppoints.end(), back_inserter(points));
            }

            return points;
        }

        void cuda_dilate(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst) {
            DILATE_FILTER->apply(src, dst);
        }

        // std::vector<LineEdge>
        cv::Mat ExtractCUDALinesedges(CudaStuff &cs) {
            // std::vector<LineEdge> edges;
            cv::Mat lines;

            for (int i = 0; i < cs.ks.size(); i++) {
                CudaKernels &k = cs.ks[i];
                cv::Rect kpROI(0, 0, k.dataW, k.dataH);
                lines = cv::Mat(k.fftH, k.fftW, cv::DataType<float>::type, k.h_ResultLines);
            }

            return lines;
            // return edges;
        }


        /// \brief Creates a vector of linearly spaced wavelengths (lambda) for multiscale analysis
        ///
        /// \param lambda_start Initial wavelength of the gabor filters
        /// \param lambda_end   Ending wavelength of the gabor filters
        /// \param numScales    Number of scales (linearly spread between \a lambda_start and \a lambda_end)
        ///
        /// Returns a vector of wavelengths
        std::vector<float> makeLambdasLin(float lambda_start, float lambda_end, int numScales) {
            std::vector<float> lambdas;
            float lambda_step = 1;
            int NS = numScales;

            if (NS > 1) lambda_step = (lambda_end - lambda_start) / (NS - 1);

            for (float s = 1; s <= NS; s++) {
                float lambda = lambda_start + (s - 1) * lambda_step;
                lambdas.push_back(lambda);
            }

            return lambdas;
        }

        /// \brief Creates a vector of logarithmically spaced wavelengths (lambda) for multiscale analysis
        ///
/// \param lambda_start     Initial wavelength of the gabor filters
/// \param lambda_end       Ending wavelength of the gabor filters
/// \param scalesPerOctave  Number of scales per octave
///
/// Returns a vector of wavelengths
        std::vector<float> makeLambdasLog(float lambda_start, float lambda_end, int scalesPerOctave) {
            std::vector<float> lambdas;

            float multiplier = pow(2., 1. / scalesPerOctave);

            for (float lambda = lambda_start; lambda <= lambda_end * 1.05; lambda *= multiplier) {
                lambdas.push_back(lambda);
            }

            return lambdas;
        }

    } //namespace cuda
} // namespace bimp
