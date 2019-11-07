#include <helper_cuda.h>
#include <dirent.h>
#include "detection/utils.hpp"
#include <numeric>
#include "../../bimp/include/detection/cuda_bimp.hpp"
#include <opencv2/xfeatures2d/cuda.hpp>

namespace bimp {
	namespace utils {

        std::map<std::string, std::vector<double> > timeMap;

        void showTimes() {
			for ( auto it = timeMap.begin(); it != timeMap.end(); it++ ) {
                auto v = it->second;

				double sum = std::accumulate(v.begin(), v.end(), 0.0);
				double mean = sum / v.size();

				double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
				double stdev = std::sqrt(sq_sum / v.size() - mean * mean);

				std::cout << it->first << " mu: " << mean << "\tsigma: " << stdev << std::endl;
			}

		}

		const int OUT_LENGTH = 60;

		int log_depth = 0;

		std::vector <std::string> read_directory( const std::string& path) {
			std::vector <std::string> result;
			dirent* de;
			DIR* dp;
			errno = 0;
			dp = opendir( path.empty() ? "." : path.c_str() );
			if (dp) {
				while (true) {
					errno = 0;
					de = readdir( dp );
					if (de == NULL) break;
					std::string name = de->d_name;
					if (name != "." && name != "..")
						result.push_back( std::string( path + "/" + name ) );
				}
				closedir( dp );
				std::sort( result.begin(), result.end() );
			}
			return result;
		}


		void log(std::string msg, int depth, double *t, bool backtrack) {
		    double delta = 0;
		    if (t != 0) {
                delta = deltaT(*t);
                if (timeMap.find(msg) == timeMap.end()) {
					std::vector<double> vec;
                    timeMap[msg] = vec;
				}
				timeMap[msg].push_back(delta);
            }
		    
			if (depth > log_depth)
				return;

			if (backtrack && depth > 0)
				depth = depth - 1;

			std::cout << std::left;

			for (int i = 1; i <= depth; i++) {
				std::cout << " ";
			}

			std::cout << std::setw(OUT_LENGTH) << msg;

			if (t != 0) {
				std::cout << std::fixed << std::setprecision(6) << delta;
                *t = (double)cv::getTickCount();
			}

			std::cout << std::endl;
		}

		void write(std::string name, cv::cuda::GpuMat &mat) {
			long t= (long)cv::getTickCount();

			cv::Mat cpuMat;
			mat.download(cpuMat);

			std::stringstream ss;
			ss << "log-" << name << "-" << t;

			std::ofstream file;
			file.open(ss.str());

			for (int j = 0; j < cpuMat.rows; j++) {
				for (int i = 0; i < cpuMat.cols; i++) {
					file << cpuMat.at<float>(j,i) << " ";
				}
				file << std::endl;
			}

			file.close();
		}


		void uploadKeypoints(std::vector<cv::KeyPoint> points, cv::cuda::GpuMat &result) {
			cv::Mat keypoints(KP_VECTOR_LENGTH,points.size(),CV_32F);

			for (int j = 0; j < points.size(); ++j) {
				cv::KeyPoint point = points.at(j);

				keypoints.at<float>(cv::cuda::SURF_CUDA::X_ROW,j) = point.pt.x;
				keypoints.at<float>(cv::cuda::SURF_CUDA::Y_ROW,j) = point.pt.y;
				keypoints.at<float>(cv::cuda::SURF_CUDA::SIZE_ROW,j) = point.size;
				keypoints.at<float>(cv::cuda::SURF_CUDA::ANGLE_ROW,j) = point.angle;
				keypoints.at<float>(cv::cuda::SURF_CUDA::HESSIAN_ROW,j) = point.response;
				keypoints.at<int>(cv::cuda::SURF_CUDA::OCTAVE_ROW,j) = (int)(point.octave);
				keypoints.at<int>(cv::cuda::SURF_CUDA::LAPLACIAN_ROW,j) = 1;
			}

			result.upload(keypoints);
		}

		std::vector<cv::KeyPoint> downloadKeypoints(cv::cuda::GpuMat &keypoints) {
			cv::Mat pointsMat(keypoints.size(), keypoints.type());
			keypoints.download(pointsMat);
			std::vector<cv::KeyPoint> points;

			for (int j = 0; j < pointsMat.cols; j++) {

				cv::Mat col = pointsMat.col(j);
				cv::KeyPoint kp;
				kp.pt.x = pointsMat.at<float>(cv::cuda::SURF_CUDA::X_ROW,j);
				kp.pt.y = pointsMat.at<float>(cv::cuda::SURF_CUDA::Y_ROW,j);
				kp.class_id = pointsMat.at<int>(cv::cuda::SURF_CUDA::LAPLACIAN_ROW,j);
				kp.size = pointsMat.at<float>(cv::cuda::SURF_CUDA::SIZE_ROW,j);
				kp.angle = pointsMat.at<float>(cv::cuda::SURF_CUDA::ANGLE_ROW,j);
				kp.response = pointsMat.at<float>(cv::cuda::SURF_CUDA::HESSIAN_ROW,j);
				kp.octave = pointsMat.at<int>(cv::cuda::SURF_CUDA::OCTAVE_ROW,j);

				points.push_back(kp);
			}
			return points;
		}

		void debugVotes(cv::cuda::GpuMat classes_voted, int columns)	{
			cv::Mat res_host;
			classes_voted.download(res_host);
			for (int j = 0; j < res_host.rows; ++j) {
				int binaryVote = 0;
				for (int i = 0; i < res_host.cols; ++i) {
					if (res_host.at<int>(j,i) > 0)
						binaryVote |= 2 << i;
				}

				std::cout << binaryVote << " ";
				if (j % columns == 0 && j != 0)
					std::cout << std::endl;
			}
            std::cout << std::endl;
		}

		void debugPrintDescriptors(cv::cuda::GpuMat &img) {
			cv::Mat res_host;
			img.download(res_host);

			for (int j = 0; j < res_host.rows; ++j) {
				int zeros = 0;
				int extremes = 0;
				int healthy = 0;
				for (int i = 0; i < res_host.cols; ++i) {
					float f = res_host.at<float>(j,i);
					if (f == 0) {
						zeros++;
					} else if (f < -3 || f > 3) {
						extremes++;
					} else {
						healthy++;
					}
				}
				std::cout << "Healthy: " << healthy << "\t\tZeros: " << zeros << "\t\tExtrema: " << extremes << std::endl;
			}
		}

		void debugPrintImage(cv::cuda::GpuMat &img, int type) {
			cv::Mat res_host;
			img.download(res_host);

			for (int j = 0; j < res_host.rows; ++j) {
				for (int i = 0; i < res_host.cols; ++i) {
					if (type == CV_32F) {
                        float f = res_host.at<float>(j,i);
						if (f <= 0) {
							std::cout << ".";
						} else {
							std::cout << "*";
						}
						//std::cout << f << " ";
					}
					else {
						std::cout << res_host.at<int>(j,i) << " ";
					}
				}
                std::cout << std::endl;
			}
		}

		void debugViewImage(cv::cuda::GpuMat img) {
			cv::Mat res_host;
			img.download(res_host);
			cv::imshow("Debug", res_host);
			cv::waitKey();
		}

		std::string str(float arg) {
			std::stringstream ss;
			ss << arg;
			return ss.str ();
		}

		void showCUDAMem() {
			size_t free_byte ;
			size_t total_byte ;

			auto cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
			if ( cudaSuccess != cuda_status ){
				printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
				exit(1);
			}

			double free_db = (double)free_byte ;
			double total_db = (double)total_byte ;
			double used_db = total_db - free_db ;
			printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
				   used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
		}
	} //namespace utils
}//namespace detection
