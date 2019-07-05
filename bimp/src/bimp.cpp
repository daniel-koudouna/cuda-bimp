#include <cuda.h>
#include <detection/bimp.hpp>
#include <detection/cuda_bimp.hpp>

namespace bimp {
    Context::Context(cv::Mat &image) :
            cuda_data(image.cols, image.rows, bimp::cuda::makeLambdasLog(4,64,2))
    {
      (*this).current_image = bimp::cuda::getGPUImage(image);
      (*this).current_image_color = bimp::cuda::getGPUImage(image, false);
      (*this).current_keypoints = cv::cuda::GpuMat();
      (*this).surf_descriptors = cv::cuda::GpuMat();
      (*this).current_descriptors = cv::cuda::GpuMat();
      (*this).has_changed_keypoints = true;
      (*this).has_changed_descriptors = true;
      (*this).cpu_mode = false;
      (*this).resize_cpu = false;
      (*this).usingBIMP = true;
    }

  int Context::width()
  {
    return current_image.cols;
  }

  int Context::height()
  {
    return current_image.rows;
  }

    cv::cuda::GpuMat & Context::getDescriptors()
    {

      cudaDeviceSynchronize();
      if (this->has_changed_descriptors)
      {
        this->surf(this->current_image, cv::cuda::GpuMat(),this->getKeypoints(), (*this).surf_descriptors, this->usingBIMP);

        /** Allocate additional space for color histogram features **/
        this->current_descriptors.create(this->surf_descriptors.rows, DIMENSIONS, CV_32F);
        bimp::cuda::copy(this->surf_descriptors, this->current_descriptors);
        bimp::cuda::addColorFeatures(this->current_keypoints,this->current_image_color, this->current_descriptors.colRange(SURF_DIMENSIONS, DIMENSIONS));

        /** Log example of a descriptor **/
        //cv::Mat cpu_descs;
        //this->current_descriptors.download(cpu_descs);
        //cv::Mat row = cpu_descs.row(0);
        //for (int i = 0; i < DIMENSIONS; ++i)
        //{
        //  std::cout << row.at<float>(i) << " ";
        //}
        //std::cout << std::endl;

        (*this).has_changed_descriptors = false;
      }

      return current_descriptors;
    }

  cv::cuda::GpuMat& Context::getKeypoints() {
    cudaDeviceSynchronize();
    if (this->has_changed_keypoints) {
      if (this->usingBIMP) {
	  (*this).current_keypoints = bimp::cuda::getKeypoints(this->current_image, this->cuda_data, this->cpu_mode, this->resize_cpu);
      } else {
        cv::cuda::SURF_CUDA surf;
        surf(this->current_image, cv::cuda::GpuMat(), this->current_keypoints);
      }
      (*this).has_changed_keypoints = false;
    }
    return current_keypoints;
  }

    cv::Mat Context::getDescriptorsAndDownload() {
      cv::Mat result;
      cv::cuda::GpuMat& descriptors = getDescriptors();
      descriptors.download(result);

      return result;
    }

    std::vector<cv::KeyPoint> Context::getKeypointsAndDownload() {
      cv::cuda::GpuMat kpts = getKeypoints();
      return bimp::utils::downloadKeypoints(kpts);
    }

    std::vector<bimp::cuda::CudaKernels> Context::getResponses() {
	std::vector<bimp::cuda::CudaKernels> result;

	std::vector<bimp::cuda::CudaKernels> kers = cuda_data.ks;

	for (int i = 0 ; i < kers.size(); ++i) {
	    result.push_back(kers[i]);
	}

	return result;
    };

  void Context::loadImage(cv::Mat new_image)
  {
    loadImage(bimp::cuda::getGPUImage(new_image));
  }

  void Context::loadImage(cv::cuda::GpuMat new_image)
  {
    (*this).current_image.release();
    (*this).current_image = new_image;
    (*this).has_changed_keypoints = true;
    (*this).has_changed_descriptors = true;
  }

  bool Context::getCPUMode()
  {
    return this->cpu_mode;
  }

  void Context::setCPUMode(bool cpu_mode)
  {
    (*this).cpu_mode = cpu_mode;
    (*this).has_changed_keypoints = true;
    (*this).has_changed_descriptors = true;
  }

  std::vector<cv::DMatch> matchContextsAndDownload(Context &c1, Context &c2, bool use_own, bool use_cpu)
  {
    if (!use_cpu)
    {
      cv::cuda::GpuMat gpuMatches = matchContexts(c1,c2,use_own,use_cpu);
      return downloadMatches(gpuMatches);
    }
    else
    {
      cv::Ptr<SURF> cpu_surf = SURF::create();

      cv::Mat i1, i2;

      c1.current_image.download(i1);
      c2.current_image.download(i2);

      cv::Mat c1_desc, c2_desc;
      std::vector<cv::KeyPoint> k1,k2;

      if (use_own)
      {
	  k1 = bimp::utils::downloadKeypoints(c1.getKeypoints());
	  k2 = bimp::utils::downloadKeypoints(c1.getKeypoints());
        cpu_surf->compute(i1, k1, c1_desc);
        cpu_surf->compute(i2, k2, c2_desc);
      }
      else
      {
        cpu_surf->detectAndCompute(i1, cv::noArray(), k1, c1_desc, false);
        cpu_surf->detectAndCompute(i2, cv::noArray(), k2, c2_desc, false);
      }

      cv::FlannBasedMatcher matcher;
      std::vector< std::vector< cv::DMatch> > matches;
      matcher.knnMatch( c1_desc, c2_desc, matches, 2 );

      std::vector< cv::DMatch > good_matches;
      for (int k = 0; k < std::min(c2_desc.rows - 1, (int)matches.size()); k++)
      {
        if ( (matches[k][0].distance < 0.6*(matches[k][1].distance)) &&
             ((int)matches[k].size() <= 2 && (int)matches[k].size()>0) )
        {
          good_matches.push_back( matches[k][0] );
        }
      }
    }
  }

  cv::cuda::GpuMat matchContexts(Context &c1, Context &c2, bool use_own, bool use_cpu)
  {
    int minHessian = 100;

    cv::cuda::GpuMat c1_desc, c2_desc;
    if (use_own)
    {
      c1.getKeypoints();
      cudaDeviceSynchronize();
      c2.getKeypoints();
      cudaDeviceSynchronize();
    }
    cv::cuda::SURF_CUDA surf( minHessian );
    surf(c1.current_image, cv::cuda::GpuMat(),c1.current_keypoints, c1_desc, use_own);
    cudaDeviceSynchronize();
    surf(c2.current_image, cv::cuda::GpuMat(),c2.current_keypoints, c2_desc, use_own);
    cudaDeviceSynchronize();
    if (!use_own)
    {
      c1.has_changed_keypoints = false;
      c2.has_changed_keypoints = false;
    }

    cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher();
    cv::cuda::GpuMat matchesGPU;
    matcher->knnMatchAsync(c1_desc, c2_desc, matchesGPU, 2, cv::cuda::GpuMat());
    return matchesGPU;

  }

  std::vector<cv::DMatch> downloadMatches(cv::cuda::GpuMat matches)
  {

    cv::Mat matchesCPU;
    matches.download(matchesCPU);

    int *trainIdxPtr = matchesCPU.ptr<int>(0);
    float *distancePtr =  matchesCPU.ptr<float>(1);

    std::vector< cv::DMatch > good_matches;

    for (int i = 0; i < matchesCPU.cols; ++i)
    {
      int idx1 = *trainIdxPtr;
      float dist1 = *distancePtr;
      ++trainIdxPtr;
      ++distancePtr;

      int idx2 = *trainIdxPtr;
      float dist2 = *distancePtr;
      ++trainIdxPtr;
      ++distancePtr;

      if (dist1 < 0.6*dist2)
      {
        good_matches.push_back(cv::DMatch(i, idx1, 0, dist1));
      }
    }

    return good_matches;
  }

  cv::Mat getMinSized(cv::Mat& mat)
  {
    if (mat.rows < SURF_MIN_SIZE)
    {
      cv::Mat res;
      float sf = SURF_MIN_SIZE/((float)mat.rows);
      cv::resize(mat, res, cv::Size(0,0), sf, sf, cv::INTER_NEAREST);

      //std::cout << "Upscaling by s.f " << sf << std::endl;
      return res;
    }
    else if (mat.cols < SURF_MIN_SIZE)
    {
      cv::Mat res;
      float sf = SURF_MIN_SIZE/((float)mat.cols);
      cv::resize(mat, res, cv::Size(0,0), sf, sf, cv::INTER_CUBIC);

      //std::cout << "Upscaling by s.f " << sf << std::endl;
      return res;
    }
    else
    {
      return mat;
    }
  }

    void showResized(cv::cuda::GpuMat &mat, float sf, bool showImage)
    {
	cv::Mat temp;
	mat.download(temp);
	showResized(temp,sf, showImage);
    }

    void showResized(cv::Mat &mat, float sf, bool showImage)
    {
	cv::Mat view_mat;
	for(int j = 0; j < mat.rows; j++){
	    for (int i = 0; i < mat.cols; i++){
		float f = mat.at<float>(j,i);
		//if (f != 0)
		//{
		if (f < 0)
		{
		    std::cout << "|";
		}
		else
		{
		    std::cout << f << "|";
		}
		    //}
		    //	else
		    //	{
		    //	    std::cout << ".";
		    //	}
	    }
	    std::cout << std::endl;
	}

	std::cout << "__________" <<  std::endl;

	if (showImage)
	{
	    cv::resize(mat, view_mat, cv::Size(), sf,sf, cv::INTER_NEAREST);
	    imshow("output",view_mat);
	    cv::waitKey();
	}
    }
}
