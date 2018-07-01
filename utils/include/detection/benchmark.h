#ifndef EVALUATION_BENCHMARK_H
#define EVALUATION_BENCHMARK_H

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace vislab{
  namespace evaluation{

    /// A 2D vector of integers containing the confusion matrix. First index is actual numerical label, the second is the predicted one.
    typedef std::vector< std::vector<int> > ConfusionMatrix;

    struct HFeature
    {
      int labelnum;   // if classified
      int x, y;
      float size;
      std::vector<float> distances;

      std::vector<int> members_classes;
      std::vector<int> members;
      cv::Mat descs;
      std::vector<cv::Point> ptlocs;
    
      cv::Mat visdesc;    // for general visual appearance of this region, like a single SIFT descriptor
      
      HFeature() { labelnum = x = y = -1; }
      
      HFeature( int _ln, int _x, int _y, cv::Mat _d, std::vector<cv::Point> _ptl )
      {
        labelnum = _ln;
        x = _x;
        y = _y;
        descs = _d.clone();
        ptlocs = std::vector<cv::Point>(_ptl);
      }
    };

    /// Structure representing an image used for object recognition
    struct ImgInfo
    {
      std::string filename;   ///< Filename of the image containing the object
      std::string label;      ///< The textual label of the image as a whole
      int labelnum;           ///< A numerical label used for actual classification
      
      int width, height;
      
      int x, y;   // Used for sub-images. Should be replaced with a proper feature class

      std::vector<cv::Rect> ROI;              ///< Region of interest
      std::vector<std::string> objlabels;     ///< text label of individual objects in the image
      std::vector<int> objlabelnums;          ///< num label of individual objects in the image
      
      cv::MatND hist;         ///< Colour histogram of the image, temporary

      std::vector<cv::KeyPoint> points;
      std::vector<cv::Point> ptlocs;
      // std::vector<cv::KeyPoint> points2;
      // std::vector<cv::KeyPoint> points3;
      
      std::vector<cv::Point2f> offsets;
      std::vector<cv::Rect> rects;
      
      // std::vector<std::vector<unsigned char> > descs;
      // std::vector<cv::Mat_<double> > descs2;
      cv::Mat descs3;
      // cv::Mat descs3plus;
      // std::vector<cv::MatND> descs4;
      
      std::vector<cv::Mat> hier_descs;
      std::vector<std::vector<cv::Point> > hier_locs;
      std::vector<std::vector<HFeature> > hier_features;
      
      // std::vector<int> cbhist;
      
      // cv::Mat_<float> kploc;
      // cv::Mat_<float> kploc2;
      // cv::Mat_<float> kploc3;
      
      ImgInfo()   { labelnum=-1; width=-1; height=-1; x=-1; y=-1;}
    };

    

    void printConfusionMatrix( ConfusionMatrix array, std::string filename, std::vector<std::string> labels );
    float getClassificationRate( ConfusionMatrix array );

    // Coil 100 dataset
    std::vector<std::string> selectCoilObjects( int num );
    std::vector<int> selectCoilViews( int num, bool random );
    
    int getCoilDatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                         std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, 
                         bool random = false, int imagelimit = 1000 );
    
    // Caltech 101 dataset
    std::vector<std::string> selectCaltechObjects( int num );
    std::vector<int> selectCaltechViews( int num, int max, bool random );
    
    int getCaltechDatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                            std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, 
                            bool random = false, int imagelimit = 1000 );

      int convertCaltechDataset( std::string base);
    
    // Caltech 256 dataset
    std::vector<std::string> selectCaltech256Objects( int num );
    
    int getCaltech256Datasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                               std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, 
                               bool random = false, int imagelimit = 1000 );
    
    // PASCAL VOC 2011 dataset
    std::vector<std::string> selectVOC2011Objects( int num );
    std::vector<int> selectVOC2011Views( int num, bool random );
    std::vector<ImgInfo> VOC2011GetAnnotations( std::string imgfile, std::string annofile, std::string curobj, int i );
    
    int getVOC2011Datasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                            std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, 
                            bool random = false, int imagelimit = 1000 );
    
    // ETHZ dataset
    std::vector<std::string> selectETHZObjects( int num );
    std::vector<int> selectETHZViews( int num, int max, bool random );
    
    int getETHZDatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                         std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, 
                         bool random = false, int imagelimit = 1000 );
    
    // Extended ETHZ dataset
    std::vector<std::string> selectETHZextObjects( int num );
    std::vector<int> selectETHZextViews( int num, int max, bool random );
    
    int getETHZextDatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                            std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, 
                            bool random = false, int imagelimit = 1000 );
    
    // INRIA Horses dataset
    int getHorsesDatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                           std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, 
                           bool random = false, int imagelimit = 1000);

    // Bochum's BOIL dataset of 30 household items
    std::vector<std::string> selectBOILObjects( int num );
    int getBOILDatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                         std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, bool random, int imagelimit );

    // IIIA30 Robotic object detection dataset
    int getIIIADatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                         std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, 
                         bool random = false, int imagelimit = 1000, int number = 1);

    // Generic dataset loaded from a set of directories named after classes
    int getGenericDatasets( std::string base, int numClasses, int numViews, 
                            std::vector<std::string> &objects,
                            std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, 
                            std::string ext = "jpg", bool random = false, int imagelimit = 1000 );

    int getGenericDetectionTests( std::string base, std::vector<ImgInfo> &files_test, 
                                  std::string ext = "jpg", int imagelimit = 1000 );
    int getUIUCDetectionTests( std::string base, std::vector<ImgInfo> &files_test, 
                               std::string ext, int imagelimit );
  }//namespace evaluation
}//namespace vislab

#endif // EVALUATION_BENCHMARK_H
