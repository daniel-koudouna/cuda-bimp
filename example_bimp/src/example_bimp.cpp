#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sstream>
#include <iomanip>
#include <unistd.h>

#include "detection/cuda_bimp.hpp"
#include "detection/bimp.hpp"
#include "detection/utils.hpp"
#include "detection/convolutionFFT2D_common.h"

#define VK_QUIT 'q'
#define VK_TOGGLE_GPU 'g'
#define VK_TOGGLE_KPTS 'k'
#define VK_TOGGLE_SHOW_PTS 'p'
#define VK_TOGGLE_RENDER 'r'

const std::string keys =
    "{help h usage ? |      | print this message           }"
    "{v video-file   |      | use a video file             }"
    "{c cpu          |      | use the CPU for calculations }"
    "{o camera       |      | use the camera               }"
    "{m match        |      | match file1 to video output  }"
    "{w width        |600   | width to use for video       }"
    "{h height       |400   | height to use for video      }"
    "{k keypoints    |      | use SURF keypoints           }"
    "{s silent       |      | hide output (for CLI)        }"
    "{r resizecpu    |      | resize on the CPU            }"
    "{@file1         |      | first file to use            }"
    "{@file2         |      | second file to use           }"
    ;


void showSingle(cv::CommandLineParser parser)
{
    std::string path = parser.get<std::string>(0);
    bool cpu_mode = parser.has("cpu");

    cv::Mat img = cv::imread(path);
    cv::Mat output(img.size(), img.type());
    cv::Rect ROI(0,0,img.cols,img.rows);

    bool render = !parser.has("silent");

    bimp::Context context(img);

    context.setCPUMode(cpu_mode);

    context.usingBIMP = !parser.has("keypoints");
    cv::cuda::GpuMat keypoints = context.getKeypoints();
    std::vector<cv::KeyPoint> points = bimp::utils::downloadKeypoints(keypoints);

    drawKeypoints(img, points, output, cv::Scalar::all(255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    imwrite("out.png", output(ROI));

    if (render)
    {
	imshow("output", output);
	cv::waitKey();
    }
}

void matchImages(cv::CommandLineParser parser)
{
    std::string p1 = parser.get<std::string>(0);
    std::string p2 = parser.get<std::string>(1);

    bool cpu_mode = parser.has("cpu");
    bool render = !parser.has("silent");
    bool use_own = !parser.has("keypoints");

    cv::Mat h_train = cv::imread(p1);
    cv::Mat h_test = cv::imread(p2);

    cv::Mat train = bimp::getMinSized(h_train);
    cv::Mat test = bimp::getMinSized(h_test);

    bimp::Context c1(train);
    bimp::Context c2(test);

    int minHessian = 100;

    c1.setCPUMode(cpu_mode);
    c2.setCPUMode(cpu_mode);

    std::vector<cv::DMatch> good_matches = matchContextsAndDownload(c1,c2, use_own, cpu_mode);

    if (render)
    {
	cv::Mat finalImage;

	std::vector<cv::KeyPoint> train_kpts = bimp::utils::downloadKeypoints(c1.current_keypoints);
	std::vector<cv::KeyPoint> test_kpts = bimp::utils::downloadKeypoints(c2.current_keypoints);

	std::cout << "sizes train test " << train_kpts.size() << "," << test_kpts.size() << std::endl;

	if (use_own)
	{
	    drawKeypoints(train, train_kpts, train, cv::Scalar::all(255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	    drawKeypoints(test, test_kpts, test, cv::Scalar::all(255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	}

	drawMatches(train,train_kpts,test,test_kpts,good_matches,finalImage,cv::Scalar::all(255),cv::Scalar::all(255),std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);

	imwrite("out.png", finalImage);
	imshow("output", finalImage);
	cv::waitKey(0);
    }

}

int videoKeypoints(cv::VideoCapture capture, cv::CommandLineParser parser, bool endless)
{

    int width = parser.get<int>("width");
    int height = parser.get<int>("height");

    cv::Mat frame;
    capture >> frame;
    resize(frame, frame, cv::Size(width,height), cv::INTER_AREA );
    bimp::Context context(frame);

    double t = (double)cv::getTickCount();
    double t0 = t;

    int framenum=0;
    cv::Mat output(cv::Size(width,height), CV_32F);

    bool cpu_mode = parser.has("cpu");
    bool render = !parser.has("silent");
    bool use_own = !parser.has("keypoints");

    context.setCPUMode(cpu_mode);
    context.resize_cpu = parser.has("resizecpu");

    for(;;) {
        double t = (double)cv::getTickCount();

        capture >> frame; // get a new frame from camera

        bimp::utils::log("get frame",0,&t);

        if (!endless && !capture.read(frame))
            break;

        resize(frame, frame, cv::Size(width,height), cv::INTER_AREA );
        framenum++;

        bimp::utils::log("resize frame",0,&t);

        context.loadImage(frame);
        context.usingBIMP = use_own;
        cv::cuda::GpuMat keypoints = context.getKeypoints();

        bimp::utils::log("get keypoints",0,&t);

        if (render)
        {
            std::vector<cv::KeyPoint> points = bimp::utils::downloadKeypoints(keypoints);
            drawKeypoints(frame, points, output, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imshow("Output", output);
        }

        char pressed = cv::waitKey(1);

        if(pressed == VK_QUIT) break;

        switch (pressed) {
            case VK_TOGGLE_GPU:
                cpu_mode = !cpu_mode;
                std::cout << "Cpu mode is " << cpu_mode << std::endl;
                context.setCPUMode(cpu_mode);
                break;
            case VK_TOGGLE_RENDER:
                render = !render;
                break;
            case VK_TOGGLE_KPTS:
                use_own = !use_own;
                break;
        }
    }

    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;

    double time = ((double)cv::getTickCount()-t0)/cv::getTickFrequency();

    std::cout << "Execution time: " << time << std::endl;
    std::cout << framenum/time << " frames per second." << std::endl;

    return 0;
}

int video_file(cv::CommandLineParser &parser)
{

    cv::VideoCapture cap(parser.get<std::string>(0));
    return videoKeypoints(cap,parser,false);
}

int camera(cv::CommandLineParser parser)
{
    cv::VideoCapture cap(0);
    return videoKeypoints(cap, parser, true);
}

int matchToCamera(cv::CommandLineParser parser)
{
    cv::VideoCapture camera(0);

    std::string file = parser.get<std::string>(0);
    cv::Mat object = cv::imread(file);

    int width = parser.get<int>("width");
    int height = parser.get<int>("height");

    cv::Mat frame;
    camera >> frame;
    resize(frame, frame, cv::Size(width,height), cv::INTER_AREA );

    bimp::Context objectContext(object);
    bimp::Context videoContext(frame);

    double t = (double)cv::getTickCount();
    double t0 = t;

    int framenum=0;

    bool cpu_mode = parser.has("cpu");
    bool show_pts = false;
    bool use_own = true;
    bool render = !parser.has("silent");

    for (;;)
    {
	camera >> frame;
        resize(frame, frame, cv::Size(width,height), cv::INTER_AREA );
        framenum++;

	videoContext.loadImage(frame);
	cv::cuda::GpuMat camKpts = videoContext.getKeypoints();
	cv::cuda::GpuMat objKpts = objectContext.getKeypoints();
	cv::cuda::GpuMat gpuMatches = matchContexts(objectContext, videoContext, use_own, cpu_mode);

	if (render)
	{
	    cv::Mat imgMatches;
	    cv::Mat out_frame, out_object;

	    if (show_pts && use_own)
	    {
		std::vector<cv::KeyPoint> camera_points = bimp::utils::downloadKeypoints(camKpts);
		std::vector<cv::KeyPoint> object_points = bimp::utils::downloadKeypoints(objKpts);
		drawKeypoints(object, object_points, out_object, cv::Scalar::all(255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		drawKeypoints(frame, camera_points, out_frame, cv::Scalar::all(255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	    }
	    else
	    {
		out_object = object;
		out_frame = frame;
	    }

	    std::vector<cv::DMatch> good_matches = bimp::downloadMatches(gpuMatches);
	    std::vector<cv::KeyPoint> camera_kpts = bimp::utils::downloadKeypoints(camKpts);
	    std::vector<cv::KeyPoint> object_kpts = bimp::utils::downloadKeypoints(objKpts);
	    drawMatches(out_object,object_kpts,out_frame,camera_kpts,good_matches,imgMatches,cv::Scalar::all(200),cv::Scalar::all(200),std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);
	    imshow("output", imgMatches);
	}


	char pressed = cv::waitKey(1);

        if(pressed == VK_QUIT) break;

	switch (pressed)
	{
	case VK_TOGGLE_GPU:
	    cpu_mode = !cpu_mode;
	    videoContext.setCPUMode(cpu_mode);
	    objectContext.setCPUMode(cpu_mode);
	    std::cout << "CPU mode is " << cpu_mode << std::endl;
	    break;
	case VK_TOGGLE_KPTS:
	    use_own = !use_own;
	    break;
	case VK_TOGGLE_SHOW_PTS:
	    show_pts = !show_pts;
	    break;
	}
    }

    return 0;
}

int main(int argc, const char *const*argv)
{
    cv::CommandLineParser parser(argc, argv, keys);

    std::string path1 = parser.get<std::string>(0);
    std::string path2 = parser.get<std::string>(1);

    bool cpu_mode = parser.has("cpu");

    if (!parser.check())
    {
	parser.printErrors();
	return 0;
    }
    else if (parser.has("help"))
    {
	parser.printMessage();
	return 0;
    }
    else if (parser.has("camera"))
    {
	camera(parser);
    }
    else if (parser.has("video-file"))
    {
	video_file(parser);
    }
    else if (parser.has("match"))
    {
	matchToCamera(parser);
    }
    else
    {
	if (path1 == "")
	{
	    parser.printMessage();
	    return 0;
	}

	if (path2 == "")
	{
	    showSingle(parser);
	}
	else
	{
	    matchImages(parser);
	}

	return 0;
    }
}
