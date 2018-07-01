#include "convolutionFFT2D_common.h"
#include "cuda_bimp.hpp"
#include "cuda.h"
#define COLOR_FACTOR 0.5

__global__ void addColorFeatures_kernel(
    cv::cuda::PtrStepSz<float> keypoints,
    cv::cuda::PtrStepSz<uchar3> color_image,
    cv::cuda::PtrStepSz<float> descriptors
    ){
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > descriptors.rows)
        return;


    float x = keypoints.ptr(cv::cuda::SURF_CUDA::X_ROW)[i];
    float y = keypoints.ptr(cv::cuda::SURF_CUDA::Y_ROW)[i];
    float size = keypoints.ptr(cv::cuda::SURF_CUDA::SIZE_ROW)[i];

    int bins = COLOR_DIMENSIONS/3;
    float thresh = 255.0/(1.0*bins);

    float *desc = descriptors.ptr(i);
    /** Set memory to zero here to avoid memset elsewhere **/
    for (int k = 0; k < COLOR_DIMENSIONS; ++k)
        desc[k] = 0;

    float *desc_b = desc;
    float *desc_g = desc + bins;
    float *desc_r = desc + bins + bins;

    int im_x = (int)x;
    int im_y = (int)y;
    int pixels = (int)(log2(size)/2.0);
    int found_pixels = 0;

    for (int iy = im_y - pixels; iy < im_y + pixels; ++iy)
    {
        for (int ix = im_x - pixels; ix < im_x + pixels; ++ix)
        {
            if (iy < 0 || iy > color_image.rows)
                continue;
            if (ix < 0 || ix > color_image.cols)
                continue;

            uchar3 pixel = color_image(iy,ix);
            char b = pixel.x;
            char g = pixel.y;
            char r = pixel.z;

            found_pixels += 3;
            for (int k = 0; k < bins; ++k)
            {
                if (b > thresh*k && b < thresh*(k+1))
                    desc_b[k]++;

                if (g > thresh*k && g < thresh*(k+1))
                    desc_g[k]++;

                if (r > thresh*k && r < thresh*(k+1))
                    desc_r[k]++;
            }
        }
    }

    for (int k = 0; k < COLOR_DIMENSIONS; ++k) {
        desc[k] = (COLOR_FACTOR*desc[k])/(found_pixels*1.0);
    }
}

__global__ void copy_kernel(
    cv::cuda::PtrStepSz<float> src,
    cv::cuda::PtrStepSz<float> dest
    ){
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i > src.cols || j > src.rows)
        return;

    dest.ptr(j)[i] = src.ptr(j)[i];
}

__global__ void identifyKeypoints_kernel(
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
    int *counter){
    float *p, *firstp, *secondp, *locp;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
       
    if (i > off && i < keypts.cols - off && j > off && j < keypts.rows - off) {
	p = keypts.ptr(j);
	firstp = keypts.ptr(j-1);
	secondp = keypts.ptr(j+1);
	locp = kLoc.ptr(j);

	if ((locp[i] > 0 && p[i] > thresh)) {
	    int index = atomicAdd(counter, 1);

	    float xpos = i, ypos = j;

	    /** Simulate parabolapeak */
	    float xv1 = p[i-1];
	    float xv2 = p[i];
	    float xv3 = p[i+1];
	    
	    float yv1 = firstp[i];
	    float yv2 = p[i];
	    float yv3 = secondp[i];
	    
	    float denom_x = xv1 + xv3 - 2*xv2;
	    float denom_y = yv1 + yv3 - 2*yv2; 
	    
	    float num_x = 0.5*(xv1 - xv3);
	    float num_y = 0.5*(yv1 - yv3);

	    if (denom_x != 0)
		xpos += num_x/denom_x;

	    if (denom_y != 0)
		ypos += num_y/denom_y;

	    if (index < result.cols) {
		/** Add keypoint to result matrix **/
		result.ptr(cv::cuda::SURF_CUDA::X_ROW)[index] = xpos*pyrstep;
		result.ptr(cv::cuda::SURF_CUDA::Y_ROW)[index] = ypos*pyrstep;
		resultInts.ptr(cv::cuda::SURF_CUDA::LAPLACIAN_ROW)[index] = 1;
		resultInts.ptr(cv::cuda::SURF_CUDA::OCTAVE_ROW)[index] = lambda*pyrstep;
		result.ptr(cv::cuda::SURF_CUDA::SIZE_ROW)[index] = lambda*pyrstep;
		result.ptr(cv::cuda::SURF_CUDA::ANGLE_ROW)[index] = -1;
		result.ptr(cv::cuda::SURF_CUDA::HESSIAN_ROW)[index] = p[i];
	    }
	} else {
	    locp[i] = 0; 
	}
    }
}


__global__ void collectCVMemory_kernel(
	cv::cuda::PtrStepSz<float> cv_image,
	float *d_result,
	int cols,
	int rows
){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < cols && j < rows) {
		d_result[j*cols + i] = cv_image(j,i);
	}
}

__global__ void complexResponse_kernel(
    float *d_simple_e,
    float *d_simple_o,
    float *d_complex,
    int dataSize
){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= dataSize)
        return;

    float a = d_simple_e[i];
    float b = d_simple_o[i];
    float c = sqrt( a*a + b*b );

    d_complex[i] = c;
}

__global__ void cleararray_kernel(
    fComplex *d_in,
    int dataSize
){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= dataSize)
        return;

    d_in[i].x = 0;
    d_in[i].y = 0;
}

__global__ void clearfloatarray_kernel(
    float *d_in,
    int dataSize
){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= dataSize)
        return;

    d_in[i] = 0;
}

__global__ void modulateAndNormalize3_kernel(
    fComplex *d_Dst,
    fComplex *d_Src1,
    fComplex *d_Src2,
    int dataSize,
    float c
){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= dataSize)
        return;

    fComplex a = d_Src1[i];
    fComplex b = d_Src2[i];

    fComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
    t.x += d_Dst[i].x;
    t.y += d_Dst[i].y;

    d_Dst[i] = t;
}

__global__ void FilterSimpleCells_kernel(
    fComplex *result_e,
    fComplex *result_o,
    fComplex *kernel_e,
    fComplex *kernel_o,
    fComplex *data,
    int dataSize,
    float c
){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= dataSize)
        return;

    fComplex a = data[i];
    fComplex b_e = kernel_e[i];
    fComplex b_o = kernel_o[i];

    fComplex t1 = {c * (a.x * b_e.x - a.y * b_e.y), c * (a.y * b_e.x + a.x * b_e.y)};
    fComplex t2 = {c * (a.x * b_o.x - a.y * b_o.y), c * (a.y * b_o.x + a.x * b_o.y)};

    result_e[i] = t1;
    result_o[i] = t2;
}

__global__ void inhibitSpectrum_kernel(
    fComplex *d_arg1,
    fComplex *d_arg2,
    fComplex *d_arg3,
    fComplex *d_arg4,
    int dataSize
){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= dataSize)
        return;

    fComplex a = d_arg1[i];
    fComplex b = d_arg2[i];
    fComplex c = d_arg3[i];
    fComplex d = d_arg4[i];

    fComplex t;
    t.x = a.x + 2*b.x - c.x - d.x;
    t.y = a.y + 2*b.y - c.y - d.y;

    d_arg1[i] = t;
}

__global__ void inhibitSpatial_kernel(
    float *d_arg1,
    float *d_arg2,
    float *d_arg3,
    float *d_arg4,
    int dataSize
){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= dataSize)
        return;

    float a = d_arg1[i]; // > 0 ? d_arg1[i] : 0;
    float b = d_arg2[i]; // > 0 ? d_arg2[i] : 0;
    float c = d_arg3[i] > 0 ? d_arg3[i] : 0;
    float d = 2*b - 16*d_arg4[i]; 
    d = d > 0 ? d : 0;

    float t = a - c - d;

    d_arg1[i] = t;
}

__global__ void sumArray_kernel( float *d_arg1, float *d_arg2, int dataW, int dataH, int stride )
{
    int threadX = blockDim.x * blockIdx.x + threadIdx.x;
    int threadY = blockDim.y * blockIdx.y + threadIdx.y;
    const int gLoc = blockDim.y*blockIdx.y * stride + threadIdx.y*stride  + threadX; // For the fft-sized image (complex cells)
    const int sLoc = blockDim.y*blockIdx.y * dataW + threadIdx.y*dataW + threadX;  // For the data-sized image (summed complex map)

    if( threadX >= dataW || threadY >= dataH ) return;

    float t = d_arg1[sLoc] + d_arg2[gLoc];

    d_arg1[sLoc] = t;
}

__global__ void inhibitSpatialAll_kernel(
    float *result_d,
    float *result_s,
    float *gauss,
    float *tanin,
    float *radin,
    int dataSize
){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= dataSize)
        return;

    float dc = result_d[i];
    float sc = result_s[i];

    float g = gauss[i];

    float c = tanin[i]; 
    float d = 2*g - 16*radin[i]; 

    c = c > 0 ? c : 0;
    d = d > 0 ? d : 0;

    float t1 = dc - c - d;
    float t2 = sc - c - d;

    result_d[i] = t1;
    result_s[i] = t2;
}

__global__ void eulerStep_kernel(
    float *field,
    float *lateral,
    float *input,
    float inhib,
    float time,
    int dataSize,
    float c
){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= dataSize)
        return;

    float h = 0;
    float signu = +1; 
    if(field[i]==0) signu = 0;
    if(field[i]<0) signu = -1;

    float du = -field[i] + h + input[i] + c*lateral[i] - signu*inhib; 
    
    field[i] = field[i] + du*time;
}

__global__ void detectLE_kernel(
         float *d_Result,  float *d_Ori,  char *d_Type,
         float *d_Ch, 
         float *d_c0,  float *d_c1,  float *d_c2,  float *d_c3, 
         float *d_c4,  float *d_c5,  float *d_c6,  float *d_c7,
         float *d_o0,  float *d_o1,  float *d_o2,  float *d_o3, 
         float *d_o4,  float *d_o5,  float *d_o6,  float *d_o7,
         float *d_e0,  float *d_e1,  float *d_e2,  float *d_e3, 
         float *d_e4,  float *d_e5,  float *d_e6,  float *d_e7,
        int dataW, int dataH, int stride )
{
    int threadX = blockDim.x * blockIdx.x + threadIdx.x;
    int threadY = blockDim.y * blockIdx.y + threadIdx.y;
    const int gLoc = blockDim.y*blockIdx.y * stride + threadIdx.y*stride  + threadX; // For the fft-sized image (complex cells)
    const int sLoc = blockDim.y*blockIdx.y * dataW + threadIdx.y*dataW + threadX;  // For the data-sized image (summed complex map)

    if( threadX >= dataW || threadY >= dataH ) return;

    float value = 0;
    float strength = 0;

    if(d_Ch[sLoc] < 0.1) value = 0;
    else
    {
        float *c_max, *o_max, *e_max;

        // Find strongest orientation. Note -- the loop is unrolled
        int maxori = -1; float maxresp = 0; float curresp = 0;
        curresp = d_c0[gLoc]; if( curresp > maxresp) { maxresp = curresp; maxori=0; c_max = d_c0; o_max = d_o0; e_max = d_e0; }
        curresp = d_c1[gLoc]; if( curresp > maxresp) { maxresp = curresp; maxori=1; c_max = d_c1; o_max = d_o1; e_max = d_e1; }
        curresp = d_c2[gLoc]; if( curresp > maxresp) { maxresp = curresp; maxori=2; c_max = d_c2; o_max = d_o2; e_max = d_e2; }
        curresp = d_c3[gLoc]; if( curresp > maxresp) { maxresp = curresp; maxori=3; c_max = d_c3; o_max = d_o3; e_max = d_e3; }
        curresp = d_c4[gLoc]; if( curresp > maxresp) { maxresp = curresp; maxori=4; c_max = d_c4; o_max = d_o4; e_max = d_e4; }
        curresp = d_c5[gLoc]; if( curresp > maxresp) { maxresp = curresp; maxori=5; c_max = d_c5; o_max = d_o5; e_max = d_e5; }
        curresp = d_c6[gLoc]; if( curresp > maxresp) { maxresp = curresp; maxori=6; c_max = d_c6; o_max = d_o6; e_max = d_e6; }
        curresp = d_c7[gLoc]; if( curresp > maxresp) { maxresp = curresp; maxori=7; c_max = d_c7; o_max = d_o7; e_max = d_e7; }
                
        float theta = maxori * 0.392699;
        int offsetx = round(cos(theta));
        int offsety = round(sin(theta));
        
        // Find simple cell maxima and minima and zero crossings in the dominant orientation
        bool romax = ( o_max[gLoc]>o_max[gLoc+offsety*stride+offsetx] && o_max[gLoc]>o_max[gLoc-offsety*stride-offsetx] );
        bool romin = ( o_max[gLoc]<o_max[gLoc+offsety*stride+offsetx] && o_max[gLoc]<o_max[gLoc-offsety*stride-offsetx] );
        bool rozc  = ( o_max[gLoc+offsety*stride+offsetx] * o_max[gLoc-offsety*stride-offsetx] < 0 );
        
        bool remax = ( e_max[gLoc]>e_max[gLoc+offsety*stride+offsetx] && e_max[gLoc]>e_max[gLoc-offsety*stride-offsetx] );
        bool remin = ( e_max[gLoc]<e_max[gLoc+offsety*stride+offsetx] && e_max[gLoc]<e_max[gLoc-offsety*stride-offsetx] );
        bool rezc  = ( e_max[gLoc+offsety*stride+offsetx] * e_max[gLoc-offsety*stride-offsetx] < 0 );
        
        // Second and third inhibition step, remove spurius stuff
        if( ( romax || romin || remax || remin ) != true ) value=0;
        else if( ( rezc  || rozc ) != true ) value=0;
        else if( c_max[gLoc] <= c_max[gLoc+offsety*stride+offsetx] || c_max[gLoc] <= c_max[gLoc-offsety*stride-offsetx] ) value=0;
        else value = 1;

        if(value == 1)
        {
            // Determine orientation and type
            if( rozc && remax ) d_Type[sLoc] = 1;    // positive line
            if( rozc && remin ) d_Type[sLoc] = 2;    // negative line
            if( rezc && romax ) d_Type[sLoc] = 3;    // positive edge
            if( rezc && romin ) d_Type[sLoc] = 4;    // negative edge
            d_Ori[sLoc] = theta;
        }
        strength = c_max[gLoc];

    }

    d_Result[sLoc] = value * strength;
}



__global__ void endStoppedResponse_kernel( float *d_double, float *d_single, float *d_complex, 
                                         int offset1, int offset2 ,int dataW, int dataH, int stride)
{
    int threadX = blockDim.x * blockIdx.x + threadIdx.x;
    int threadY = blockDim.y * blockIdx.y + threadIdx.y;

    const int gLoc = blockDim.y*blockIdx.y * stride + threadIdx.y*stride  + threadX; // For the fft-sized image (complex cells)
    const int sLoc = blockDim.y*blockIdx.y * dataW + threadIdx.y*dataW + threadX;  // For the data-sized image (summed complex map)

    if( threadX >= dataW || threadY >= dataH ) return;
    if( threadX-offset2 < 0 || threadX-offset2 > dataW || threadX+offset2 < 0 || threadX+offset2 > dataW) return;
    if( threadY-offset1 < 0 || threadY-offset1 > dataH || threadY+offset1 < 0 || threadY+offset1 > dataH) return;
    if( threadX-offset2*2 < 0 || threadX-offset2*2 > dataW || threadX+offset2*2 < 0 || threadX+offset2*2 > dataW) return;
    if( threadY-offset1*2 < 0 || threadY-offset1*2 > dataH || threadY+offset1*2 < 0 || threadY+offset1*2 > dataH) return;

    float val = 0;

    /*float threshs = 30, threshd = 20;*/
    float threshs = 0, threshd = 0;

    // single-stopped cells
    val = d_complex[gLoc-offset1*stride+offset2] - d_complex[gLoc+offset1*stride-offset2];
    val = val>0 ? val : 0;
    if(val > threshs)
	    atomicAdd(&d_single[sLoc], val);

    // double-stopped cells
    val = d_complex[gLoc] - 0.5 * d_complex[gLoc-offset1*2*stride+offset2*2] - 0.5 * d_complex[gLoc+offset1*2*stride-offset2*2];
    val = val>0 ? val : 0;
    if(val > threshd)
	    atomicAdd(&d_double[sLoc], val);
}

__global__ void inhibitionResponse_kernel( float *d_tan_in, float *d_rad_in, 
                                         float *d_complex, float *d_complex2, 
                                         int offset1, int offset2, int dataW, int dataH, int stride )
{
    int threadX = blockDim.x * blockIdx.x + threadIdx.x;
    int threadY = blockDim.y * blockIdx.y + threadIdx.y;
    const int gLoc = blockDim.y*blockIdx.y * stride + threadIdx.y*stride  + threadX; // For the fft-sized image (complex cells)
    const int sLoc = blockDim.y*blockIdx.y * dataW + threadIdx.y*dataW + threadX;  // For the data-sized image (summed complex map)

    if( threadX >= dataW || threadY >= dataH ) return;
    if( threadX-offset2 < 0 || threadX-offset2 > dataW || threadX+offset2 < 0 || threadX+offset2 > dataW) return;
    if( threadY-offset1 < 0 || threadY-offset1 > dataH || threadY+offset1 < 0 || threadY+offset1 > dataH) return;
    if( threadX-offset2*2 < 0 || threadX-offset2*2 > dataW || threadX+offset2*2 < 0 || threadX+offset2*2 > dataW) return;
    if( threadY-offset1*2 < 0 || threadY-offset1*2 > dataH || threadY+offset1*2 < 0 || threadY+offset1*2 > dataH) return;

    float val = 0;

    // tangential inhibition
    float centre = d_complex[gLoc];
    val = d_complex[gLoc+offset1*stride+offset2] - centre;
    val = val<0 ? 0 : val;
    val = d_complex[gLoc-offset1*stride-offset2] - centre;
    val = val<0 ? 0 : val;
    atomicAdd(&d_tan_in[sLoc],val);
    
    // radial inhibition
    offset1 = round(0.5*offset1);
    offset2 = round(0.5*offset2);
    val  = centre - 16 * d_complex2[gLoc+offset1*stride+offset2];
    val = val<0 ? 0 : val;
    val += centre - 16 * d_complex2[gLoc-offset1*stride-offset2];
    val = val<0 ? 0 : val;
    atomicAdd(&d_rad_in[sLoc],val);
}

__global__ void inhibitKeypoints_kernel( float *d_double, float *d_single, 
                                       float *d_tan_in, float *d_rad_in, 
                                       int dataW, int dataH, int stride)
{
    int threadX = blockDim.x * blockIdx.x + threadIdx.x;
    int threadY = blockDim.y * blockIdx.y + threadIdx.y;
    const int gLoc = blockDim.y*blockIdx.y * stride + threadIdx.y*stride  + threadX; // For the fft-sized image (complex cells)

    if( threadX >= dataW || threadY >= dataH ) return;

    float inhib = d_tan_in[gLoc] + d_rad_in[gLoc];
    atomicAdd(&d_double[gLoc], -inhib);
    atomicAdd(&d_single[gLoc],-inhib);
}


__global__ void inhibitionResponseLE_kernel( 
    float *d_lat_in, float *d_cro_in, float *d_complex, float *d_complex2, int offset1, int offset2, 
    int dataW, int dataH, int stride )
{
    // global mem address for this thread
    const int gLoc = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * stride + blockIdx.y * blockDim.y * stride;

    float val1 = 0, val2 = 0;
    float result; 

    // lateral inhibition
    val1 = d_complex[gLoc+offset1*stride+offset2];
    val2 = d_complex[gLoc-offset1*stride-offset2];
    result = val1 - val2;
    if(result <0) result = -result;
    result -= (val1+val2)/2; 
    if(result <0) result = 0;
    d_lat_in[gLoc] += result*4; // - (val1+val2)/2; 

    // cross-orientation inhibition
    offset1 *= 2;
    offset2 *= 2;
    float centre = d_complex[gLoc];
    val1 =  d_complex2[gLoc+offset1*stride+offset2] + d_complex2[gLoc-offset1*stride-offset2] - 2*centre;
    val1 =  val1<0 ? 0 : val1;
    d_cro_in[gLoc] = val1;
}



