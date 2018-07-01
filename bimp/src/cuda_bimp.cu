#include <assert.h>
#include <cuda.h> /*needed in order to get CUDA_VERSION*/
#include "opencv2/opencv.hpp"
#if CUDA_VERSION >= 5000
#include <helper_cuda.h>
#else
#include <cutil_inline.h>
#define getLastCudaError cutilCheckMsg
#define checkCudaErrors cutilSafeCall
#endif /*CUDA_VERSION >= 5000*/
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>

#include "detection/cuda_bimp.cuh"
#include "detection/convolutionFFT2D_common.h"

#define BLX 256
    
struct compare_vals
{
    __host__ __device__
    bool operator()(float lhs, float rhs) const
    {
	if (lhs > 1000 || lhs != lhs)
	{
	    return 1;
	}
	else
	{
	    return lhs < rhs;
	}
    }
};

extern "C" void copy(
		cv::cuda::PtrStepSz<float> src,
		cv::cuda::PtrStepSz<float> dest)
{

	const dim3 block(16, 16);
	const dim3 grid( (src.cols + block.x)/block.x, (src.rows + block.y)/block.y);

	copy_kernel<<<grid,block>>>(src,dest);
}

extern "C" void addColorFeatures(
		cv::cuda::PtrStepSz<float> keypoints,
		cv::cuda::PtrStepSz<uchar3> color_image,
		cv::cuda::PtrStepSz<float> descriptors)
{
   	addColorFeatures_kernel<<<descriptors.rows,1>>>(keypoints,color_image,descriptors);
}

extern "C" void thrustFindMax(float *src, int size, float *max)
{
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(src);
    thrust::device_ptr<float> max_ptr = thrust::max_element(dev_ptr, dev_ptr + size);//, compare_vals());

    *max = max_ptr[0];
}

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
    int *counter
    ){

    float cols = keypts.cols;
    float rows = keypts.rows;
    
    const dim3 block(16, 16);
    const dim3 grid( (cols + block.x)/block.x, (rows + block.y)/block.y);

    //std::cout << "identifying keypoints " << keypts.cols << " , " << keypts.rows  << " pyrstep: " << pyrstep  << " max " << colMax << "," << rowMax << std::endl;
       
    identifyKeypoints_kernel<<<grid,block>>>(
	result,
	resultInts,
	keypts,
	kLoc,
	lambda,
	off,
	rowMax,
	colMax,
	pyrstep,
	thresh,
	counter
	);
}

extern "C" void collectCVMemory(
    cv::cuda::PtrStepSz<float> cv_image,
    float *d_result,
    int dataW,
    int dataH
    ){
    float cols = dataW;
    float rows = dataH;

    const dim3 block(16, 16);
    const dim3 grid( (cols + block.x)/block.x, (rows + block.y)/block.y);


    collectCVMemory_kernel<<<grid, block>>>(
	cv_image,
	d_result,
	dataW,
	dataH
	);
}

extern "C" void complexResponse(
    float *d_simple_e,
    float *d_simple_o,
    float *d_complex,
    int fftW,
    int fftH
    ){
    const int dataSize = fftH * fftW;

    complexResponse_kernel<<<iDivUp(dataSize, BLX), BLX>>>(
	d_simple_e,
	d_simple_o,
	d_complex,
	dataSize
	);
    getLastCudaError("complexResponse() execution failed\n");
}

extern "C" void cleararray(
    fComplex *d_in,
    int fftW,
    int fftH
    ){
    const int dataSize = fftH * fftW;

    cleararray_kernel<<<iDivUp(dataSize, BLX), BLX>>>(
	d_in,
	dataSize
	);
    getLastCudaError("cleararray() execution failed\n");
}

extern "C" void clearfloatarray(
    float *d_in,
    int fftW,
    int fftH
    ){
    const int dataSize = fftH * fftW;

    clearfloatarray_kernel<<<iDivUp(dataSize, BLX), BLX>>>(
	d_in,
	dataSize
	);
    getLastCudaError("clearfloatarray() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
/*extern "C" void modulateAndNormalize2(*/
/*    fComplex *d_Dst,*/
/*    fComplex *d_Src1,*/
/*    fComplex *d_Src2,*/
/*    int fftH,*/
/*    int fftW,*/
/*    int padding*/
/*){*/
/*    assert( fftW % 2 == 0 );*/
/*    const int dataSize = fftH * (fftW / 2 + padding);*/
/**/
/*    modulateAndNormalize2_kernel<<<iDivUp(dataSize, BLX), BLX>>>(*/
/*        d_Dst,*/
/*        d_Src1,*/
/*        d_Src2,*/
/*        dataSize,*/
/*        1.0f / (float)(fftW * fftH)*/
/*    );*/
/*    getLastCudaError("modulateAndNormalize() execution failed\n");*/
/*}*/

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
extern "C" void modulateAndNormalize3(
    fComplex *d_Dst,
    fComplex *d_Src1,
    fComplex *d_Src2,
    int fftW,
    int fftH
    ){
    assert( fftW % 2 == 0 );
    int padding = 1;
    const int dataSize = fftH * (fftW / 2 + padding);

    modulateAndNormalize3_kernel<<<iDivUp(dataSize, BLX), BLX>>>(
	d_Dst,
	d_Src1,
	d_Src2,
	dataSize,
	1.0f / (float)(fftW * fftH)
	);
    getLastCudaError("modulateAndNormalize() execution failed\n");
}

extern "C" void FilterSimpleCells(
    fComplex *result_e,
    fComplex *result_o,
    fComplex *kernel_e,
    fComplex *kernel_o,
    fComplex *data,
    int fftW,
    int fftH,
    cudaStream_t stream
    ){
    assert( fftW % 2 == 0 );
    int padding = 1;
    const int dataSize = fftH * (fftW / 2 + padding);

    FilterSimpleCells_kernel<<<iDivUp(dataSize, BLX), BLX>>>(
	result_e,
	result_o,
	kernel_e,
	kernel_o,
	data,
	dataSize,
	1.0f / (float)(fftW * fftH)
	);
    getLastCudaError("filterSimpleCells() execution failed\n");
}


extern "C" void inhibitSpectrum(
    fComplex *d_arg1,
    fComplex *d_arg2,
    fComplex *d_arg3,
    fComplex *d_arg4,
    int fftW,
    int fftH
    ){
    assert( fftW % 2 == 0 );
    const int dataSize = fftH * fftW;

    inhibitSpectrum_kernel<<<iDivUp(dataSize, BLX), BLX>>>(
	d_arg1,
	d_arg2,
	d_arg3,
	d_arg4,
	dataSize
	);
    getLastCudaError("inhibitSpectrum() execution failed\n");
}

extern "C" void inhibitSpatial(
    float *d_arg1,
    float *d_arg2,
    float *d_arg3,
    float *d_arg4,
    int fftW,
    int fftH
    ){
    assert( fftW % 2 == 0 );
    const int dataSize = fftH * fftW;

    inhibitSpatial_kernel<<<iDivUp(dataSize, BLX), BLX>>>(
	d_arg1,
	d_arg2,
	d_arg3,
	d_arg4,
	dataSize
	);
    getLastCudaError("inhibitSpatial() execution failed\n");
}


extern "C" void sumArray(
    float *d_arg1,
    float *d_arg2,
    int dataW,
    int dataH,
    int stride
    ){
    dim3 _blockDim(32, 32, 1);
    dim3 _gridDim((dataW + _blockDim.x - 1)/ _blockDim.x, (dataH + _blockDim.y - 1) / _blockDim.y, 1);

    sumArray_kernel<<<_gridDim, _blockDim, 0>>>(
	d_arg1,
	d_arg2,
	dataW, dataH, stride 
	);
    getLastCudaError("sumArray() execution failed\n");
}

extern "C" void endStoppedResponse( 
    float *d_double,
	float *d_single,
	float *d_complex,
	int offset1,
	int offset2,
	int dataW,
	int dataH,
	int stride,
	cudaStream_t stream)
{
    dim3 _blockDim(32, 32, 1);
    dim3 _gridDim((dataW + _blockDim.x - 1)/ _blockDim.x, (dataH + _blockDim.y - 1) / _blockDim.y, 1);

    endStoppedResponse_kernel<<<_gridDim,_blockDim, 0,stream>>>(
	d_double, d_single, d_complex, offset1, offset2, 
	dataW, dataH, stride
	);
    getLastCudaError("endStoppedResponse() execution failed\n");
}

extern "C" void inhibitionResponse( 
    float *d_tan_in, float *d_rad_in, float *d_complex, float *d_complex2, int offset1, int offset2, 
    int dataW, int dataH, int stride, cudaStream_t stream)
{
    dim3 _blockDim(32, 32, 1);
    dim3 _gridDim((dataW + _blockDim.x - 1)/ _blockDim.x, (dataH + _blockDim.y - 1) / _blockDim.y, 1);

    inhibitionResponse_kernel<<<_gridDim, _blockDim, 0, stream>>>(
	d_tan_in, d_rad_in, d_complex, d_complex2, offset1, offset2, 
	dataW, dataH, stride
	);
    getLastCudaError("inhibitionResponse() execution failed\n");
}

extern "C" void inhibitionResponseLE( 
    float *d_lat_in, float *d_cro_in, float *d_complex, float *d_complex2, int offset1, int offset2, 
    int dataW, int dataH, int stride )
{
    assert( stride % 2 == 0 );
    const int dataSize = dataH * stride;

    inhibitionResponseLE_kernel<<<iDivUp(dataSize, BLX), BLX>>>(
	d_lat_in, d_cro_in, d_complex, d_complex2, offset1, offset2, 
	dataW, dataH, stride
	);
    getLastCudaError("inhibitionResponseLE() execution failed\n");
}

extern "C" void inhibitKeypoints( float *d_double, float *d_single, float *d_tan_in, float *d_rad_in, 
				  int dataW, int dataH, int stride)
{
    dim3 _blockDim(32, 32, 1);
    dim3 _gridDim((dataW + _blockDim.x - 1)/ _blockDim.x, (dataH + _blockDim.y - 1) / _blockDim.y, 1);

    inhibitKeypoints_kernel<<<_gridDim, _blockDim, 0>>>(
	d_double, d_single, d_tan_in, d_rad_in, dataW, dataH, stride );
    getLastCudaError("inhibitionResponse() execution failed\n");
}






extern "C" void inhibitSpatialAll(
    float *result_d,
    float *result_s,
    float *gauss,
    float *tanin,
    float *radin,
    int fftW,
    int fftH
    ){
    assert( fftW % 2 == 0 );
    const int dataSize = fftH * fftW;

    inhibitSpatialAll_kernel<<<iDivUp(dataSize, BLX), BLX>>>(
	result_d,
	result_s,
	gauss,
	tanin,
	radin,
	dataSize
	);
    getLastCudaError("inhibitSpatialAll() execution failed\n");
}

/*extern "C" void detectLE(*/
/*    float *result,*/
/*    float *ch,*/
/*    int dataH,*/
/*    int dataW*/
/*){*/
/*    assert( dataW % 2 == 0 );*/
/*    const int dataSize = dataH * dataW;*/
/**/
    
/*cudaArray*/
/*    *a_Data;*/
/*cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();*/
/*CUDA_SAFE_CALL( cudaMallocArray(&a_Data, &floatTex, DATA_W, DATA_H) );*/
/**/
/*CUDA_SAFE_CALL( cudaBindTextureToArray(texData, ch) );*/

/*detectLE_kernel1<<<iDivUp(dataSize, BLX), BLX>>>(*/
/*    result,*/
/*    data,*/
/*    dataSize*/
/*);*/
/*getLastCudaError("detectLE() execution failed\n");*/
/*}*/

/*extern "C" void detectLE_old(*/
/*        float *d_Result,*/
/*        float *d_Data,*/
/*        int dataW,*/
/*        int dataH )*/
/*{*/
/*    assert( dataW % 2 == 0 );*/
/*    const int dataSize = dataH * dataW;*/
/**/
/*    detectLE_kernel<<<iDivUp(dataSize, BLX), BLX>>>(*/
/*        d_Result,*/
/*        d_Data,*/
/*        dataW,*/
/*        dataH*/
/*    );*/
/*    getLastCudaError("detectLE() execution failed\n");*/
/*}*/

extern "C" void detectLE(
    float *d_Result, float *d_Ori, char *d_Type,
    float *d_Ch, 
    float *d_c0, float *d_c1, float *d_c2, float *d_c3, float *d_c4, float *d_c5, float *d_c6, float *d_c7,
    float *d_o0, float *d_o1, float *d_o2, float *d_o3, float *d_o4, float *d_o5, float *d_o6, float *d_o7, 
    float *d_e0, float *d_e1, float *d_e2, float *d_e3, float *d_e4, float *d_e5, float *d_e6, float *d_e7,
    int dataW,
    int dataH, int stride )
{
    dim3 _blockDim(32, 32, 1);
    dim3 _gridDim((dataW + _blockDim.x - 1)/ _blockDim.x, (dataH + _blockDim.y - 1) / _blockDim.y, 1);

    detectLE_kernel<<<_gridDim, _blockDim, 0>>>(
	d_Result, d_Ori, d_Type,
	d_Ch, 
	d_c0, d_c1, d_c2, d_c3, d_c4, d_c5, d_c6, d_c7,
	d_o0, d_o1, d_o2, d_o3, d_o4, d_o5, d_o6, d_o7,
	d_e0, d_e1, d_e2, d_e3, d_e4, d_e5, d_e6, d_e7,
	dataW,
	dataH, stride
	);
    getLastCudaError("detectLE() execution failed\n");
}


extern "C" void eulerStep(
    float *field,
    float *lateral,
    float *input,
    float inhib,
    float time,
    int fftW,
    int fftH
    ){
    assert( fftW % 2 == 0 );
    const int dataSize = fftH * fftW;

    eulerStep_kernel<<<iDivUp(dataSize, BLX), BLX>>>(
	field,
	lateral,
	input,
	inhib,
	time,
	dataSize,
	1.0f / (float)(fftW * fftH)
	);
    getLastCudaError("eulerStep() execution failed\n");
}

