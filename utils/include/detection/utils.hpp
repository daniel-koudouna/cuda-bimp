#ifndef BIMPUTILS_H
#define BIMPUTILS_H

#include <cstdlib>
#include <unistd.h>
#include <string>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <cuda.h>
#include <vector>

namespace bimp {
    namespace utils {

		extern int log_depth;

		std::vector <std::string> read_directory( const std::string& path = std::string() );
		/**
         * Log to stdout with an optional time taken.
         */
		void log(std::string msg, int depth, double *t = 0, bool backtrack = false);

		/**
         * Write a matrix on the GPU to disk.
         */
		void write(std::string name, cv::cuda::GpuMat &mat);

		/**
         * Helper functions to convert keypoints from and to the GPU.
         */
		void uploadKeypoints(std::vector<cv::KeyPoint> points,cv::cuda::GpuMat &result);
		std::vector<cv::KeyPoint> downloadKeypoints(cv::cuda::GpuMat &keypoints);

		/**
         * Utility function to calculate the time difference of a timestamp.
         */
		inline double deltaT(double then)
		{
			return ((double)cv::getTickCount() - then)/cv::getTickFrequency();
		}

		/**
         * Download and show an image on the GPU.
         */
		void debugViewImage(cv::cuda::GpuMat &img);

		/**
		 * Download and print the contents of a GPU matrix.
		 */
		void debugPrintImage(cv::cuda::GpuMat &img, int type = CV_32F);

		void debugVotes(cv::cuda::GpuMat &classes_voted, int columns);

		std::string str(float arg);

        void showTimes();

		void showCUDAMem();
	}

}
#endif
