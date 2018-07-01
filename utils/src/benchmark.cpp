#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <dirent.h>
#include "detection/util.h"
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include "detection/benchmark.h"

/*! \file benchmark.cpp 
  \brief Tools for dealing with benchmark datasets*/

namespace vislab{
  namespace evaluation{


    //
    // COIL 100
    //

    // Internal: selects a set of numbers, used for the Coil dataset
    std::vector<std::string> selectCoilObjects( int num )
    {
      std::vector<std::string> result;

      for(int objnum = 1; objnum <= num; objnum++)
        {
          std::stringstream str;
          str << objnum;
          result.push_back(str.str());
        }

      return result;
    }

    // Internal: selects a set of numbers, which determine the training images; used for the Coil dataset
    std::vector<int> selectCoilViews( int num, bool random )
    {
      std::vector<int> result;
    
      if(!random)
        {
          for(int angle = 0; angle < 360; angle += 360/num)
            result.push_back(angle);
        }
      else
        {
          for(int i=0; i<num; i++)
            {
              bool problem = false;
              int randnum = -1;

              do 
                {
                  problem = false;
                  randnum = rand()%72 * 5;
                  if(std::find(result.begin(), result.end(), randnum) != result.end())
                    {
                      problem = true;
                      break;
                    }
                }
              while(problem);

              result.push_back(randnum);
            }
        }

      return result;
    }

    /// \brief Creates a training and a testing set for the COIL 100 benchmark  
    ///
    /// \param base         Base directory of the Coil 100 dataset, where the images are located
    /// \param numClasses   Number of classes to test
    /// \param numViews     Number of images (per class) to use for the training set. The remaining images will be used for testing
    /// \param objects      Will contain a vector of strings listing the object classes being used
    /// \param files_train  Will contain a list of training objects with filenames and corresponding labels 
    /// \param files_test   Will contain a list of testing objects with filenames and corresponding labels 
    /// \param random       Whether the split should be randomised
    ///
    /// Returns 0 if all goes well
    int getCoilDatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                         std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, bool random, int imagelimit )
    {
      // Get a list of objects
      objects = selectCoilObjects( numClasses );

      // Get a list of views used for learning, or as example templates
      std::vector<int> learningViews = selectCoilViews( numViews, random );

      // Sort the learning views to make lookups easier
      std::sort(learningViews.begin(), learningViews.end());

      // Now loop through all the objects and get the filenames with the selected orientations
      for(unsigned i=0; i<objects.size(); i++)
        {
          std::string curobj = objects[i];
        
          int number = 0;
        
          for(int angle=0; angle<360 && number<imagelimit; angle+=5)
            {
              number++;
              ImgInfo inf;

              std::stringstream str;
              str << base << "/obj" << curobj << "__" << angle << ".png";

              inf.filename = str.str();
              inf.label = curobj;
              inf.labelnum = i;

              // angles reserved for learning
              if(std::binary_search(learningViews.begin(), learningViews.end(), angle))
                files_train.push_back(inf);
              else     // the rest is for testing 
                files_test.push_back(inf);
            }
        }

      return 0;
    }


    //
    // CALTECH 101
    //

    // Internal: selects a set of Caltech labels, used for the Caltech dataset
    std::vector<std::string> selectCaltechObjects( int num )
    {
      std::vector<std::string> result;

      const char *objc[] = { "accordion", "butterfly", "crocodile_head", "ferry", "joshua_tree", "minaret", "rooster", "sunflower", "airplanes", "camera", "cup", "flamingo", "kangaroo", "Motorbikes", "saxophone", "tick", "anchor", "cannon", "dalmatian", "flamingo_head", "ketch", "nautilus", "schooner", "trilobite", "ant", "car_side", "dollar_bill", "garfield", "lamp", "octopus", "scissors", "umbrella", "ceiling_fan", "dolphin", "gerenuk", "laptop", "okapi", "scorpion", "watch", "barrel", "cellphone", "dragonfly", "gramophone", "Leopards", "pagoda", "sea_horse", "water_lilly", "bass", "chair", "electric_guitar", "grand_piano", "llama", "panda", "snoopy", "wheelchair", "beaver", "chandelier", "elephant", "hawksbill", "lobster", "pigeon", "soccer_ball", "wild_cat", "binocular", "cougar_body", "emu", "headphone", "lotus", "pizza", "stapler", "windsor_chair", "bonsai", "cougar_face", "euphonium", "hedgehog", "mandolin", "platypus", "starfish", "wrench", "brain", "crab", "ewer", "helicopter", "mayfly", "pyramid", "stegosaurus", "yin_yang", "brontosaurus", "crayfish", "Faces", "ibis", "menorah", "revolver", "stop_sign", "buddha", "crocodile", "Faces_easy", "inline_skate", "metronome", "rhino", "strawberry", "BACKGROUND_Google" };

      for(int i=0; i<101 && i<num; i++)
        {
          result.push_back(objc[i]);
        }

      return result;
    }

    // Internal: selects a set of numbers, which determine the images used for training; used for the Caltech dataset
    std::vector<int> selectCaltechViews( int num, int max, bool random )
    {
      std::vector<int> result;
    
      if(!random)
        {
          for(int v = 1; v <= num && v <= max; v++)
            result.push_back(v);
        }
      else
        {
          // for(int i=1; i<=num && i<=max; i++)
          for(int i=1; i<=num ; i++)
            {
              bool problem = false;
              int randnum = -1;

              randnum = rand()%max+1;
              do 
                {
                  problem = false;
                  randnum = rand()%max+1;
                  if(std::find(result.begin(), result.end(), randnum) != result.end())
                    {
                      problem = true;
                    }
                }
              while(problem);

              result.push_back(randnum);
            }
        }
    
      if(result.size() != (unsigned) num)
        std::cout << "Warning: wanted " << num << " images, but got " << result.size() << "!\n";

      return result;
    }

    /// \brief Creates a training and a testing set for the Caltech 101 benchmark  
    ///
    /// \param base         Base directory of the Caltech 101 dataset, where the "101_ObjectCategories" dir is located
    /// \param numClasses   Number of classes to test
    /// \param numViews     Number of images (per class) to use for the training set. The remaining images will be used for testing
    /// \param objects      Will contain a vector of strings listing the object classes being used
    /// \param files_train  Will contain a list of training objects containing filenames and corresponding labels 
    /// \param files_test   Will contain a list of testing objects containing filenames and corresponding labels 
    /// \param random       Whether the split should be randomised
    ///
    /// Returns 0 if all goes well
    int getCaltechDatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                            std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, bool random, int imagelimit )
    {
      // whether we should use the bbox annotations
      bool anno = true;

      // Get a list of objects
      objects = selectCaltechObjects( numClasses );

      // Now loop through all the objects and get the filenames with the selected orientations
      for(unsigned i=0; i<objects.size(); i++)
        {
          std::string curobj = objects[i];
          std::string curdir = base + "/101_ObjectCategories/" + curobj + "/"; 
          std::string curannodir = base + "/Annotations_txt/" + curobj + "/";

          int numfiles = vislab::util::dirfiles(curdir).size();
          if(numfiles > imagelimit) numfiles = imagelimit;

          // Get a list of views used for learning, or as example templates
          // The list has to be generated for each class separately, since objects have different number of images
          std::vector<int> learningViews = selectCaltechViews( numViews, numfiles, random );

          // Sort the learning views to make lookups easier
          std::sort(learningViews.begin(), learningViews.end());
        
          for(int view=1; view<=numfiles; view++)
            {
              ImgInfo inf;

              std::stringstream str;
              str << curdir << "image_" << std::setfill('0') << std::setw(4) << view << ".jpg";

              inf.filename = str.str();
              inf.label = curobj;
              inf.labelnum = i;

              // Extract the annotated bounding box, if needed
              // N.B. this assumes that the .mat files have been converted to an ascii representation beforehand
              // Don't want to deal with binary matlab dumps directly
              if(anno)
                {
                  double ymin,ymax,xmin,xmax;
                  std::stringstream str2;
                  str2 << curannodir << "annotation_" << std::setfill('0') << std::setw(4) << view << ".txt";
                  ifstream myfile (str2.str().c_str());
                  myfile >> ymin; //= temp;
                  myfile >> ymax; //= temp;
                  myfile >> xmin; //= temp;
                  myfile >> xmax; //= temp;
                  inf.ROI.push_back(cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin));
                  inf.objlabels.push_back(curobj);
                  inf.objlabelnums.push_back(i);
                }

              // angles reserved for learning
              if(std::binary_search(learningViews.begin(), learningViews.end(), view))
                files_train.push_back(inf);
              else     // the rest is for testing 
                files_test.push_back(inf);
            }
        }

      return 0;
    }


    int convertCaltechDataset( std::string base)
    {
      // whether we should use the bbox annotations

      vector<string> directories = vislab::util::readDir(base + "/101_ObjectCategories/");
      // Get a list of objects

      std::cout << "dir size " << directories.size() << std::endl;
      
      // Now loop through all the objects and get the filenames with the selected orientations
      for(unsigned i=0; i<directories.size(); i++)
      {
	  std::string curobj = directories.at(i);
	  std::string curdir = base + "/101_ObjectCategories/" + curobj + "/"; 
	  std::string curannodir = base + "/Annotations_txt/" + curobj + "/";

	  std::string targetdir = base + "/converted/" + curobj + "/";

	  vector<string> files = vislab::util::readDir(curdir);
	    
	    
	  for(int view=1; view<=files.size(); view++)
	  {
	      ImgInfo inf;
	      
	      std::stringstream str;
	      str << curdir << "image_" << std::setfill('0') << std::setw(4) << view << ".jpg";
	      
	      inf.filename = str.str();
	      inf.label = curobj;
	      inf.labelnum = i;

	      cv::Mat image = cv::imread(inf.filename);
	      // Extract the annotated bounding box, if needed
	      // N.B. this assumes that the .mat files have been converted to an ascii representation beforehand
	      // Don't want to deal with binary matlab dumps directly
	      double ymin,ymax,xmin,xmax;
	      std::stringstream str2;
	      str2 << curannodir << "annotation_" << std::setfill('0') << std::setw(4) << view << ".txt";
	      ifstream myfile (str2.str().c_str());
	      myfile >> ymin; //= temp;
	      myfile >> ymax; //= temp;
	      myfile >> xmin; //= temp;
	      myfile >> xmax; //= temp;

	      if (xmin < 0) {
		  xmin = 0;
	      }
	      if (ymin < 0) {
		  ymin = 0;
	      }
	      if (xmax > image.cols) {
		  xmax = image.cols -1;
	      }
	      if (ymax > image.rows) {
		  ymax = image.rows - 1;
	      }
	      cv::Rect rect(xmin,ymin,xmax-xmin,ymax-ymin);
	      
	      inf.ROI.push_back(rect);
	      inf.objlabels.push_back(curobj);
	      inf.objlabelnums.push_back(i);


	      std::cout << "image name " << inf.filename << endl;
	      std::cout << "image size " << image.size() << endl;
	      std::cout << "rect size  " << rect << endl;

	      if (rect.x >= 0 && rect.y >= 0 && rect.width > 0 && rect.height > 0) {
		  cv::Mat cropped = image(rect);
		  string command = "mkdir -p " + targetdir;
		  int r = system(command.c_str());

		  std::stringstream target;
		  target << targetdir << "image_" << std::setfill('0') << std::setw(4) << view << ".jpg";

		  std::cout << target.str() << std::endl;
		  cv::imwrite(target.str(), cropped);
	      }

	  }
      }

      return 0;
    }

    //
    // CALTECH 256
    //

    // Internal: selects a set of Caltech labels, used for the Caltech dataset
    std::vector<std::string> selectCaltech256Objects( int num )
    {
      std::vector<std::string> result;

      const char *objc[] = { 
        "001.ak47", "002.american-flag", "003.backpack", "004.baseball-bat", "005.baseball-glove", "006.basketball-hoop", "007.bat", "008.bathtub", "009.bear", "010.beer-mug", "011.billiards", "012.binoculars", "013.birdbath", "014.blimp", "015.bonsai-101", "016.boom-box", "017.bowling-ball", "018.bowling-pin", "019.boxing-glove", "020.brain-101", "021.breadmaker", "022.buddha-101", "023.bulldozer", "024.butterfly", "025.cactus", "026.cake", "027.calculator", "028.camel", "029.cannon", "030.canoe", "031.car-tire", "032.cartman", "033.cd", "034.centipede", "035.cereal-box", "036.chandelier-101", "037.chess-board", "038.chimp", "039.chopsticks", "040.cockroach", "041.coffee-mug", "042.coffin", "043.coin", "044.comet", "045.computer-keyboard", "046.computer-monitor", "047.computer-mouse", "048.conch", "049.cormorant", "050.covered-wagon", "051.cowboy-hat", "052.crab-101", "053.desk-globe", "054.diamond-ring", "055.dice", "056.dog", "057.dolphin-101", "058.doorknob", "059.drinking-straw", "060.duck", "061.dumb-bell", "062.eiffel-tower", "063.electric-guitar-101", "064.elephant-101", "065.elk", "066.ewer-101", "067.eyeglasses", "068.fern", "069.fighter-jet", "070.fire-extinguisher", "071.fire-hydrant", "072.fire-truck", "073.fireworks", "074.flashlight", "075.floppy-disk", "076.football-helmet", "077.french-horn", "078.fried-egg", "079.frisbee", "080.frog", "081.frying-pan", "082.galaxy", "083.gas-pump", "084.giraffe", "085.goat", "086.golden-gate-bridge", "087.goldfish", "088.golf-ball", "089.goose", "090.gorilla", "091.grand-piano-101", "092.grapes", "093.grasshopper", "094.guitar-pick", "095.hamburger", "096.hammock", "097.harmonica", "098.harp", "099.harpsichord", "100.hawksbill-101", "101.head-phones", "102.helicopter-101", "103.hibiscus", "104.homer-simpson", "105.horse", "106.horseshoe-crab", "107.hot-air-balloon", "108.hot-dog", "109.hot-tub", "110.hourglass", "111.house-fly", "112.human-skeleton", "113.hummingbird", "114.ibis-101", "115.ice-cream-cone", "116.iguana", "117.ipod", "118.iris", "119.jesus-christ", "120.joy-stick", "121.kangaroo-101", "122.kayak", "123.ketch-101", "124.killer-whale", "125.knife", "126.ladder", "127.laptop-101", "128.lathe", "129.leopards-101", "130.license-plate", "131.lightbulb", "132.light-house", "133.lightning", "134.llama-101", "135.mailbox", "136.mandolin", "137.mars", "138.mattress", "139.megaphone", "140.menorah-101", "141.microscope", "142.microwave", "143.minaret", "144.minotaur", "145.motorbikes-101", "146.mountain-bike", "147.mushroom", "148.mussels", "149.necktie", "150.octopus", "151.ostrich", "152.owl", "153.palm-pilot", "154.palm-tree", "155.paperclip", "156.paper-shredder", "157.pci-card", "158.penguin", "159.people", "160.pez-dispenser", "161.photocopier", "162.picnic-table", "163.playing-card", "164.porcupine", "165.pram", "166.praying-mantis", "167.pyramid", "168.raccoon", "169.radio-telescope", "170.rainbow", "171.refrigerator", "172.revolver-101", "173.rifle", "174.rotary-phone", "175.roulette-wheel", "176.saddle", "177.saturn", "178.school-bus", "179.scorpion-101", "180.screwdriver", "181.segway", "182.self-propelled-lawn-mower", "183.sextant", "184.sheet-music", "185.skateboard", "186.skunk", "187.skyscraper", "188.smokestack", "189.snail", "190.snake", "191.sneaker", "192.snowmobile", "193.soccer-ball", "194.socks", "195.soda-can", "196.spaghetti", "197.speed-boat", "198.spider", "199.spoon", "200.stained-glass", "201.starfish-101", "202.steering-wheel", "203.stirrups", "204.sunflower-101", "205.superman", "206.sushi", "207.swan", "208.swiss-army-knife", "209.sword", "210.syringe", "211.tambourine", "212.teapot", "213.teddy-bear", "214.teepee", "215.telephone-box", "216.tennis-ball", "217.tennis-court", "218.tennis-racket", "219.theodolite", "220.toaster", "221.tomato", "222.tombstone", "223.top-hat", "224.touring-bike", "225.tower-pisa", "226.traffic-light", "227.treadmill", "228.triceratops", "229.tricycle", "230.trilobite-101", "231.tripod", "232.t-shirt", "233.tuning-fork", "234.tweezer", "235.umbrella-101", "236.unicorn", "237.vcr", "238.video-projector", "239.washing-machine", "240.watch-101", "241.waterfall", "242.watermelon", "243.welding-mask", "244.wheelbarrow", "245.windmill", "246.wine-bottle", "247.xylophone", "248.yarmulke", "249.yo-yo", "250.zebra", "251.airplanes-101", "252.car-side-101", "253.faces-easy-101", "254.greyhound", "255.tennis-shoes", "256.toad", "257.clutter" };

      for(int i=0; i<257 && i<num; i++)
        {
          result.push_back(objc[i]);
        }

      return result;
    }


    /// \brief Creates a training and a testing set for the Caltech 256 benchmark  
    ///
    /// \param base         Base directory of the Caltech 256 dataset, where the "256_ObjectCategories" dir is located
    /// \param numClasses   Number of classes to test
    /// \param numViews     Number of images (per class) to use for the training set. The remaining images will be used for testing
    /// \param objects      Will contain a vector of strings listing the object classes being used
    /// \param files_train  Will contain a list of training objects containing filenames and corresponding labels 
    /// \param files_test   Will contain a list of testing objects containing filenames and corresponding labels 
    /// \param random       Whether the split should be randomised
    ///
    /// Returns 0 if all goes well
    int getCaltech256Datasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                               std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, bool random, int imagelimit )
    {
      // Get a list of objects
      objects = selectCaltech256Objects( numClasses );

      // Now loop through all the objects and get the filenames with the selected orientations
      for(unsigned i=0; i<objects.size(); i++)
        {
          std::string curobj = objects[i];
          std::string curdir = base + "/256_ObjectCategories/" + curobj + "/"; 

          int numfiles = vislab::util::dirfiles(curdir).size();
          if(numfiles > imagelimit) numfiles = imagelimit;

          // Get a list of views used for learning, or as example templates
          // The list has to be generated for each class separately, since objects have different number of images
          std::vector<int> learningViews = selectCaltechViews( numViews, numfiles, random );

          // Sort the learning views to make lookups easier
          std::sort(learningViews.begin(), learningViews.end());
        
          for(int view=1; view<=numfiles; view++)
            {
              ImgInfo inf;

              std::stringstream str;
              str << curdir << std::setfill('0') << std::setw(3) << i+1 << "_" << std::setw(4) << view << ".jpg";

              inf.filename = str.str();
              inf.label = curobj;
              inf.labelnum = i;

              // views reserved for learning
              if(std::binary_search(learningViews.begin(), learningViews.end(), view))
                files_train.push_back(inf);
              else     // the rest is for testing 
                files_test.push_back(inf);
            }
        }

      return 0;
    }

    //
    // PASCAL VOC 2011
    //

    // Internal: selects a set of PASCAL VOC labels, used for the VOC dataset
    std::vector<std::string> selectVOC2011Objects( int num )
    {
      std::vector<std::string> result;

      const char *objc[] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

      for(int i=0; i<20 && i<num; i++)
        {
          result.push_back(objc[i]);
        }

      return result;
    }

    // Internal: selects a set of numbers, determining which images are used; used for the VOC2011 dataset
    std::vector<int> selectVOC2011Views( int num, int max, bool random )
    {
      std::vector<int> result;
    
      if(!random)
        {
          for(int v = 0; v < num && v < max; v++)
            result.push_back(v);
        }
      else
        {
          for(int i=0; i<num ; i++)
            {
              bool problem = false;
              int randnum = -1;

              randnum = rand()%max+1;
              do 
                {
                  problem = false;
                  randnum = rand()%max+1;
                  if(std::find(result.begin(), result.end(), randnum) != result.end())
                    {
                      problem = true;
                    }
                }
              while(problem);

              result.push_back(randnum);
            }
        }
    
      if(result.size() != (unsigned) num)
        std::cout << "Warning: wanted " << num << " images, but got " << result.size() << "!\n";

      return result;
    }

    // Internal: parse a VOC xml file with annotations and return all objects of a given class as ImgInfo structs
    // N.B. This is not an elegant or correct XML parser, but it works with the VOC annotations
    std::vector<ImgInfo> VOC2011GetAnnotations( std::string imgfile, std::string annofile, std::string curobj, int labelnum )
    {
      std::vector<ImgInfo> result;

      std::ifstream xmlhandle(annofile.c_str());
      std::string line;

      if(xmlhandle.is_open())
        while(xmlhandle.good())
          {
            size_t pos = 0, pos2 = 0;
            do
              {
                if(!getline(xmlhandle,line)) return result;
                // std::cout << line << std::endl;
              }
            while((pos = line.find("<name>")) == std::string::npos);

            // If it is not the object we are looking for, skip
            if(line.find(curobj) == std::string::npos)
              continue;

            std::string xminstr, yminstr, xmaxstr, ymaxstr;

            // Next look for the bbox coordinates
            do
              {
                if(!getline(xmlhandle,line)) return result;
              }
            while((pos = line.find("<xmin>")) == std::string::npos);
            pos = line.find("<xmin>"); 
            pos2 = line.find("</xmin>");
            xminstr = line.substr(pos+6,pos2-pos-6);

            do
              {
                if(!getline(xmlhandle,line)) return result;
              }
            while((pos = line.find("<ymin>")) == std::string::npos);
            pos = line.find("<ymin>"); 
            pos2 = line.find("</ymin>");
            yminstr = line.substr(pos+6,pos2-pos-6);

            do
              {
                if(!getline(xmlhandle,line)) return result;
              }
            while((pos = line.find("<xmax>")) == std::string::npos);
            pos = line.find("<xmax>"); 
            pos2 = line.find("</xmax>");
            xmaxstr = line.substr(pos+6,pos2-pos-6);

            do
              {
                if(!getline(xmlhandle,line)) return result;
              }
            while((pos = line.find("<ymax>")) == std::string::npos);
            pos = line.find("<ymax>"); 
            pos2 = line.find("</ymax>");
            ymaxstr = line.substr(pos+6,pos2-pos-6);
            
            ImgInfo tempinfo;

            tempinfo.filename = imgfile;
            tempinfo.label = curobj;
            tempinfo.labelnum = labelnum;
            tempinfo.ROI.push_back(cv::Rect(atoi(xminstr.c_str()), atoi(yminstr.c_str()), 
                                            atoi(xmaxstr.c_str())-atoi(xminstr.c_str()), atoi(ymaxstr.c_str())-atoi(yminstr.c_str())));
            
            result.push_back(tempinfo);
          }

      return result;
    }

    /// \brief Creates a training and a testing set for the PASCAL VOC 2011 benchmark  
    ///
    /// \param base         Base directory of the PASCAL VOC dataset, where the "JPEGImages" dir is located
    /// \param numClasses   Number of classes to test
    /// \param numViews     Number of images (per class) to use for the training set.
    /// \param objects      Will contain a vector of strings listing the object classes being used
    /// \param files_train  Will contain a list of training objects containing filenames and corresponding labels 
    /// \param files_test   Will contain a list of testing objects containing filenames and corresponding labels 
    /// \param random       Whether the split should be randomised
    ///
    /// Returns 0 if all goes well
    int getVOC2011Datasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                            std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, bool random, int imagelimit )
    {
      // Get a list of objects
      objects = selectVOC2011Objects( numClasses );

      std::string curdir = base + "/JPEGImages/"; 
      std::string curannodir = base + "/Annotations/";
      std::string cursetsdir = base + "/ImageSets/Main/";
    
      // Now loop through all the objects and get the filenames with the selected orientations
      for(unsigned j=0; j<objects.size(); j++)
        {
          std::string curobj = objects[j];
        
          std::string curtrain = cursetsdir + objects[j] + "_train.txt";
          std::string curval   = cursetsdir + objects[j] + "_val.txt";

          std::vector<std::string> trainfilenames, valfilenames;

          ifstream myfiletrain(curtrain.c_str());
          ifstream myfileval(curval.c_str());

          std::string line, temp;
        
          // First get all the images showing the object, both the training and validation sets
          // (they are separate in PASCAL datasets)
          if(myfiletrain.is_open())
            while(getline(myfiletrain,line))
              {
                // Get a filename from the training image set description
                temp = line.substr(0,11) + ".jpg";

                // We only want the filenames not followed by -1
                if(line.substr(12) != "-1")     
                  trainfilenames.push_back(temp);
              }

          if(myfileval.is_open())
            while(getline(myfileval,line))
              {
                // Get a filename from the validation image set description
                temp = line.substr(0,11) + ".jpg";

                // We only want the filenames not followed by -1
                if(line.substr(12) != "-1")     
                  valfilenames.push_back(temp);
              }
    
          std::vector<ImgInfo> temptrain, temptest;
          // Now, for each image, go through the xml annotation
          for(unsigned i=0; i<trainfilenames.size() && i<(unsigned)imagelimit; i++)
            {
              std::string imgfile  = curdir + trainfilenames[i];
              std::string annofile = curannodir + trainfilenames[i];
              annofile.replace(annofile.find(".jpg"),4,".xml"); 

              std::vector<ImgInfo> annobjs = VOC2011GetAnnotations( imgfile, annofile, curobj, j );
              std::copy(annobjs.begin(), annobjs.end(), back_inserter(temptrain));
            }

          for(unsigned i=0; i<valfilenames.size() && i<(unsigned)imagelimit; i++)
            {
              std::string imgfile  = curdir + valfilenames[i];
              std::string annofile = curannodir + valfilenames[i];
              annofile.replace(annofile.find(".jpg"),4,".xml"); 
        
              std::vector<ImgInfo> annobjs = VOC2011GetAnnotations( imgfile, annofile, curobj, j );
              std::copy(annobjs.begin(), annobjs.end(), back_inserter(temptest));
            }
        
          // Now select the desired number of objects for both training and testing
          std::vector<int> seltrain, selval;

          seltrain = selectVOC2011Views( numViews, imagelimit, random );
          selval = selectVOC2011Views( numViews, imagelimit, random );

          for(unsigned i=0; i<seltrain.size(); i++)
            files_train.push_back(temptrain[seltrain[i]]);
        
          for(unsigned i=0; i< selval.size(); i++)
            files_test.push_back(temptest[selval[i]]);
        }

      return 0;
    }

    //
    // ETHZ
    //

    bool sortETHstrings( const string &first, const string &second)
    {
      string f = vislab::util::mass_replace(first, "_", "." );
      string s = vislab::util::mass_replace(second, "_", "." );
    
      return(f < s);
      // return(first < second);
    }

    // Internal: selects a set of ETHZ labels, used for the Caltech dataset
    std::vector<std::string> selectETHZObjects( int num )
    {
      std::vector<std::string> result;

      const char *objc[] = { "Applelogos", "Bottles", "Giraffes", "Mugs", "Swans" };

      for(int i=0; i<5 && i<num; i++)
        {
          result.push_back(objc[i]);
        }

      return result;
    }

    // Internal: selects a set of numbers, which determine the images used for training; used for the ETHZ dataset
    std::vector<int> selectETHZViews( int num, int max, bool random )
    {
      std::vector<int> result;
    
      if(!random)
        {
          for(int v = 1; v <= num && v <= max; v++)
            result.push_back(v);
        }
      else
        {
          // for(int i=1; i<=num && i<=max; i++)
          for(int i=1; i<=num ; i++)
            {
              bool problem = false;
              int randnum = -1;

              randnum = rand()%max+1;
              do 
                {
                  problem = false;
                  randnum = rand()%max+1;
                  if(std::find(result.begin(), result.end(), randnum) != result.end())
                    {
                      problem = true;
                    }
                }
              while(problem);

              result.push_back(randnum);
            }
        }
    
      if(result.size() != (unsigned) num)
        std::cout << "Warning: wanted " << num << " images, but got " << result.size() << "!\n";

      return result;
    }

    /// \brief Creates a training and a testing set for the ETHZ benchmark  
    ///
    /// \param base         Base directory of the ETHZ dataset, where the "README.txt" file is located
    /// \param numClasses   Number of classes to test
    /// \param numViews     Number of images (per class) to use for the training set. The remaining images will be used for testing
    /// \param objects      Will contain a vector of strings listing the object classes being used
    /// \param files_train  Will contain a list of training objects containing filenames and corresponding labels 
    /// \param files_test   Will contain a list of testing objects containing filenames and corresponding labels 
    /// \param random       Whether the split should be randomised
    ///
    /// Returns 0 if all goes well
    int getETHZDatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                         std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, bool random, int imagelimit )
    {
      // whether we should use the bbox annotations
      bool anno = true;

      // Get a list of objects
      objects = selectETHZObjects( numClasses );


      // Now loop through all the objects and get the filenames with the selected orientations
      for(unsigned i=0; i<objects.size(); i++)
        {
          std::string curobj = objects[i];
          std::string curdir = base + "/" + curobj + "/"; 
          std::string curannodir = base + "/" + curobj + "/";

          // Get all the image files and annotations from the current directory and sort them
          std::vector<std::string> filenames, annotations;
          std::vector<std::string> tempfilenames = vislab::util::dirfiles( curdir, "jpg" );
          std::vector<std::string> tempannotations = vislab::util::dirfiles( curdir, "groundtruth" );
          for(unsigned a=0; a<tempfilenames.size() && a<tempannotations.size(); a++)
            {
              if(tempfilenames[a].find("._") == std::string::npos) filenames.push_back(tempfilenames[a]);
              if(tempannotations[a].find("._") == std::string::npos) annotations.push_back(tempannotations[a]);
            }

          std::sort(filenames.begin(), filenames.end());
          std::sort(annotations.begin(), annotations.end(), sortETHstrings);

          int numfiles = filenames.size();
          if(numfiles > imagelimit) numfiles = imagelimit;

          // Get a list of views used for learning, or as example templates
          // The list has to be generated for each class separately, since objects have different number of images
          std::vector<int> learningViews = selectETHZViews( numViews, numfiles, random );
          // Sort the learning views to make lookups easier
          std::sort(learningViews.begin(), learningViews.end());

          for(int view=0; view<numfiles; view++)
            {
              ImgInfo inf;

              inf.filename = filenames[view].c_str();
              inf.label = curobj;
              inf.labelnum = i;

              // Extract the annotated bounding box, if needed
              if(anno)
                {
                  ifstream myfile (annotations[view].c_str());
                  while(!myfile.eof())
                    {
                      double ymin=-1,ymax=-1,xmin=-1,xmax=-1;
                      myfile >> xmin; //= temp;
                      myfile >> ymin; //= temp;
                      myfile >> xmax; //= temp;
                      myfile >> ymax; //= temp;
                      if(xmin == -1 || xmax == -1 || ymin == -1 || ymax == -1) break;
                      inf.ROI.push_back(cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin));
                      inf.objlabels.push_back(curobj);
                      inf.objlabelnums.push_back(i);
                    }
                }

              // views reserved for learning
              if(std::binary_search(learningViews.begin(), learningViews.end(), view))
                files_train.push_back(inf);
              else     // the rest is for testing 
                files_test.push_back(inf);
            }
        }

      return 0;
    }

    //
    // Extended ETHZ
    //

    // Internal: selects a set of ETHZ labels, used for the extended ETHZ dataset
    std::vector<std::string> selectETHZextObjects( int num )
    {
      std::vector<std::string> result;

      const char *objc[] = { "apple", "bottle", "giraffe", "hat", "mug", "starfish", "swan" };

      for(int i=0; i<7 && i<num; i++)
        {
          result.push_back(objc[i]);
        }

      return result;
    }

    /// \brief Creates a training and a testing set for the extended ETHZ benchmark  
    ///
    /// \param base         Base directory of the extended ETHZ dataset, where the "README" file is located
    /// \param numClasses   Number of classes to test
    /// \param numViews     Number of images (per class) to use for the training set. The remaining images will be used for testing
    /// \param objects      Will contain a vector of strings listing the object classes being used
    /// \param files_train  Will contain a list of training objects containing filenames and corresponding labels 
    /// \param files_test   Will contain a list of testing objects containing filenames and corresponding labels 
    /// \param random       Whether the split should be randomised
    ///
    /// Returns 0 if all goes well
    int getETHZextDatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                            std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, bool random, int imagelimit )
    {
      // whether we should use the bbox annotations
      bool anno = true;

      // Get a list of objects
      objects = selectETHZextObjects( numClasses );


      // Now loop through all the objects and get the filenames with the selected orientations
      for(unsigned i=0; i<objects.size(); i++)
        {
          std::string curobj = objects[i];
          std::string curdir = base + "/" + curobj + "/"; 
          std::string curannodir = base + "/" + curobj + "/";

          // Get all the image files and annotations from the current directory and sort them
          std::vector<std::string> filenames, annotations;
          std::vector<std::string> tempfilenames = vislab::util::dirfiles( curdir, "jpg" );
          std::vector<std::string> tempannotations = vislab::util::dirfiles( curdir, "groundtruth" );
          for(unsigned a=0; a<tempfilenames.size() && a<tempannotations.size(); a++)
            {
              if(tempfilenames[a].find("._") == std::string::npos) filenames.push_back(tempfilenames[a]);
              if(tempannotations[a].find("._") == std::string::npos) annotations.push_back(tempannotations[a]);
            }

          std::sort(filenames.begin(), filenames.end());
          std::sort(annotations.begin(), annotations.end(), sortETHstrings);

          int numfiles = filenames.size();
          if(numfiles > imagelimit) numfiles = imagelimit;

          // Get a list of views used for learning, or as example templates
          // The list has to be generated for each class separately, since objects have different number of images
          std::vector<int> learningViews = selectETHZViews( numViews, numfiles, random );
          // Sort the learning views to make lookups easier
          std::sort(learningViews.begin(), learningViews.end());

          for(int view=0; view<numfiles; view++)
            {
              ImgInfo inf;

              inf.filename = filenames[view].c_str();
              inf.label = curobj;
              inf.labelnum = i;

              // Extract the annotated bounding box, if needed
              if(anno)
                {
                  ifstream myfile (annotations[view].c_str());
                  while(!myfile.eof())
                    {
                      double ymin=-1,ymax=-1,xmin=-1,xmax=-1;
                      myfile >> xmin; //= temp;
                      myfile >> ymin; //= temp;
                      myfile >> xmax; //= temp;
                      myfile >> ymax; //= temp;
                      if(xmin == -1 || xmax == -1 || ymin == -1 || ymax == -1) break;
                      inf.ROI.push_back(cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin));
                      inf.objlabels.push_back(curobj);
                      inf.objlabelnums.push_back(i);
                    }
                }

              // views reserved for learning
              if(std::binary_search(learningViews.begin(), learningViews.end(), view))
                files_train.push_back(inf);
              else     // the rest is for testing 
                files_test.push_back(inf);
            }
        }

      return 0;
    }


    //
    //INRIA Horses 
    //

    /// \brief Creates a training and a testing set for the INRIA Horses benchmark  
    ///
    /// \param base         Base directory of the INRIA horses dataset, where the "README.txt" file is located
    /// \param numClasses   Number of classes to test
    /// \param numViews     Number of images (per class) to use for the training set. The remaining images will be used for testing
    /// \param objects      Will contain a vector of strings listing the object classes being used
    /// \param files_train  Will contain a list of training objects containing filenames and corresponding labels 
    /// \param files_test   Will contain a list of testing objects containing filenames and corresponding labels 
    /// \param random       Whether the split should be randomised
    ///
    /// Returns 0 if all goes well
    int getHorsesDatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                           std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, bool random, int imagelimit )
    {
      // whether we should use the bbox annotations
      bool anno = true;

      // Get a list of objects
      // objects = selectETHZObjects( numClasses );
      objects.clear();
      objects.push_back("pos");
      if(numClasses>1) objects.push_back("neg");

      // Now loop through all the objects and get the filenames with the selected orientations
      for(unsigned i=0; i<objects.size(); i++)
        {
          std::string curobj = objects[i];
          std::string curdir = base + "/" + curobj + "/"; 
          std::string curannodir = base + "/" + curobj + "/";

          // Get all the image files and annotations from the current directory and sort them
          std::vector<std::string> filenames, annotations;
          std::vector<std::string> tempfilenames = vislab::util::dirfiles( curdir, "jpg" );
          std::vector<std::string> tempannotations;
          if(curobj != "neg") tempannotations = vislab::util::dirfiles( curdir, "groundtruth" );

          for(unsigned a=0; a<tempfilenames.size(); a++)
            if(tempfilenames[a].find("._") == std::string::npos) filenames.push_back(tempfilenames[a]);

          for(unsigned a=0; a<tempannotations.size(); a++)
            if(tempannotations[a].find("._") == std::string::npos) annotations.push_back(tempannotations[a]);

          std::sort(filenames.begin(), filenames.end());
          std::sort(annotations.begin(), annotations.end(), sortETHstrings);

          int numfiles = filenames.size();
          if(numfiles > imagelimit) numfiles = imagelimit;

          // Get a list of views used for learning, or as example templates
          // The list has to be generated for each class separately, since objects have different number of images
          std::vector<int> learningViews = selectETHZViews( numViews, numfiles, random );
          // Sort the learning views to make lookups easier
          std::sort(learningViews.begin(), learningViews.end());

          for(int view=0; view<numfiles; view++)
            {
              ImgInfo inf;

              inf.filename = filenames[view].c_str();
              if(curobj == "pos") inf.label = "horse";
              if(curobj == "neg") inf.label = "negative";
              inf.labelnum = i;

              // Extract the annotated bounding box, if needed
              if(anno && view<annotations.size())
                {
                  ifstream myfile (annotations[view].c_str());
                  while(!myfile.eof())
                    {
                      double ymin=-1,ymax=-1,xmin=-1,xmax=-1;
                      myfile >> xmin; //= temp;
                      myfile >> ymin; //= temp;
                      myfile >> xmax; //= temp;
                      myfile >> ymax; //= temp;
                      if(xmin == -1 || xmax == -1 || ymin == -1 || ymax == -1) break;
                      inf.ROI.push_back(cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin));
                      inf.objlabels.push_back(curobj);
                      inf.objlabelnums.push_back(i);
                    }
                }

              // views reserved for learning
              if(std::binary_search(learningViews.begin(), learningViews.end(), view))
                files_train.push_back(inf);
              else     // the rest is for testing 
                files_test.push_back(inf);
            }
        }
      objects[0]="horse";

      return 0;
    }

    //
    //  IIIA 30
    //

    /// \brief Creates a training and a testing set for the IIIA30 dataset
    ///
    /// \param base         Base directory of the IIIA30 dataset, which contains the "iiia30_train_vt" directory
    /// \param numClasses   Number of classes to test
    /// \param numViews     Number of images (per class) to use for the training set. The remaining images will be used for testing
    /// \param objects      Will contain a vector of strings listing the object classes being used
    /// \param files_train  Will contain a list of training objects containing filenames and corresponding labels 
    /// \param files_test   Will contain a list of testing objects containing filenames and corresponding labels 
    /// \param random       Whether the split should be randomised
    /// \param number       Which video to load
    ///
    /// Returns 0 if all goes well
    int getIIIADatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                         std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, bool random, int imagelimit,
                         int number)
    {
      // whether we should use the bbox annotations
      bool anno = true;

      // The objects have to be in the correct order so they correspond to the ground truth annotation
      const char *objc[] = { "gray_battery", "red_battery", "bicycle", "ponce_book", "hartley_book", "calendar", "chair1", "chair2", "chair3", "charger", "cube1", "cube2", "cube3", "extinguisher", "monitor1", "monitor2", "monitor3", "orbit_box", "dentifrice", "poster_cmpi", "phone", "poster_mystrands", "poster_spices", "rack", "red_cup", "stapler", "umbrella", "window", "red_wine" };
    
      objects.clear();
      for(int i=0; i<29 && i<numClasses; i++) 
        objects.push_back(objc[i]);

      // Now loop through all the objects and get the filenames
      for(unsigned i=0; i<objects.size(); i++)
        {
          // std::cout << i << " " << objects[i] << std::endl;
          std::string curobj = objects[i];
          std::string curdir = base + "/iiia30_train_vt/" + curobj + "/"; 

          // Get all the image files and annotations from the current directory and sort them
          std::vector<std::string> filenames = vislab::util::dirfiles( curdir, "jpg" );

          int numfiles = filenames.size();
          if(numfiles > imagelimit) numfiles = imagelimit;
          // std::cout << "   " << filenames.size() << std::endl;

          for(int view=0; view<numViews && view<numfiles; view++)
            {
              ImgInfo inf;

              inf.filename = filenames[view].c_str();
              inf.label = curobj;
              inf.labelnum = i;

              // views reserved for learning
              files_train.push_back(inf);
            }
        }

      const char *vids[] = { "iiia30_seq1", "iiia30_seq2", "iiia30_seq3" };
      const char *anns[] = { "GT_iiia30_seq1.txt", "GT_iiia30_seq2.txt", "GT_iiia30_seq3.txt" };

      for(unsigned i=0; i<3; i++)
        {
          std::string curdir = base + "/" + vids[i] + "/";
          std::string curanno = base + "/" + anns[i];
          // std::cout << curdir << std::endl;

          std::vector<std::string> filenames = vislab::util::dirfiles( curdir, "png" );

          // Read the annotations
          std::vector<std::string> annotations;
          std::ifstream annofile(curanno.c_str());

          while(!annofile.eof())
            {
              std::string str;
              std::getline(annofile, str);
              annotations.push_back(str);
            }

          // loop through all the filenames
          for(unsigned j=0; j<filenames.size(); j++)
            {
              ImgInfo inf;
              std::string curfile = filenames[j];
              inf.filename = curfile;

              std::vector<std::string> tokens;
              std::string seps("/");
              vislab::util::split(curfile, seps, tokens);
              std::string shortfilename = tokens[tokens.size()-1];

              for(unsigned k=0; k<annotations.size(); k++)
                {
                  tokens.clear();
                  std::string seps2(" ");
                  vislab::util::split(annotations[k],seps2,tokens);

                  if(tokens[0] == shortfilename)
                    {
                      int xmin = atoi(tokens[1].c_str());
                      int ymin = atoi(tokens[2].c_str());
                      int xmax = atoi(tokens[3].c_str());
                      int ymax = atoi(tokens[4].c_str());
                      int objl = atoi(tokens[6].c_str());
                      inf.ROI.push_back(cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin));
                      inf.objlabels.push_back(objects[objl]);
                      inf.objlabelnums.push_back(objl);
                    }
                }
              files_test.push_back(inf);
            }

        }

      // Now loop through all the testing images objects and get the filenames

      return 0;
    }




    // Internal: selects a set of numbers, which determine the images used for training; used for generic datasets
    std::vector<int> selectGenericViews( int num, int max, bool random )
    {
      std::vector<int> result;
    
      num = std::min(num, max);

      if(!random)
        {
          for(int v = 0; v < num; v++)
            result.push_back(v);
        }
      else
        {
          // for(int i=1; i<=num && i<=max; i++)
          for(int i=1; i<=num ; i++)
            {
              bool problem = false;
              int randnum = -1;

              randnum = rand()%max+1;
              do 
                {
                  problem = false;
                  randnum = rand()%max+1;
                  if(std::find(result.begin(), result.end(), randnum) != result.end())
                    {
                      problem = true;
                    }
                }
              while(problem);

              result.push_back(randnum);
            }
        }
    
      if(result.size() != (unsigned) num)
        std::cout << "Warning: wanted " << num << " images, but got " << result.size() << "!\n";

      return result;
    }

    //
    // BOIL
    //

    // Internal: selects a set of BOIL labels, used for the BOIL dataset
    std::vector<std::string> selectBOILObjects( int num )
    {
      std::vector<std::string> result;

      const char *objc[] = { 
        "bit_box", "blue_black_screwdriver", "blue_boxcutter", "bluegreen_screwdriver", "blue_pliers", "blue_tape", "can", "casettes", "cookies", "deodorant", "fishes", "glue", "gravy", "green_screwdriver", "green_stapler", "hanuta", "honey", "multimeter", "pencil_sharpener", "razor", "red_boxcutter", "red_pliers", "red_screwdriver", "shampoo", "sunscreen", "tape_dispenser", "toothpaste", "yellow_boxcutter", "yellow_screwdriver", "yellow_stapler" };

      for(int i=0; i<30 && i<num; i++)
        {
          result.push_back(objc[i]);
        }

      return result;
    }

    /// \brief Creates a training and a testing set for the BOIL benchmark  
    ///
    /// \param base         Base directory of the BOIL dataset, where the images are located
    /// \param numClasses   Number of classes to test
    /// \param numViews     Number of images (per class) to use for the training set. The remaining images will be used for testing
    /// \param objects      Will contain a vector of strings listing the object classes being used
    /// \param files_train  Will contain a list of training objects containing filenames and corresponding labels 
    /// \param files_test   Will contain a list of testing objects containing filenames and corresponding labels 
    /// \param random       Whether the split should be randomised
    ///
    /// Returns 0 if all goes well
    int getBOILDatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                         std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, bool random, int imagelimit )
    {
      // whether we should use the bbox annotations
      bool anno = false;

      // Get a list of objects
      objects = selectBOILObjects( numClasses );

      // Now loop through all the objects and get the filenames with the selected orientations
      for(unsigned i=0; i<objects.size(); i++)
        {
          std::string curobj = objects[i];
          // std::cout << curobj << std::endl;
          std::string curdir = base + "/"; // + "/101_ObjectCategories/" + curobj + "/"; 
          std::string curannodir = base + "/"; // + "/Annotations_txt/" + curobj + "/";

          int numfiles = vislab::util::dirfiles(curdir).size();
          if(numfiles > imagelimit) numfiles = imagelimit;

          // Training view in canonical post
          {
            ImgInfo inf;

            std::stringstream str;
            str << curdir << "/" << curobj << ".train.png"; 

            inf.filename = str.str();
            inf.label = curobj;
            inf.labelnum = i;
            files_train.push_back(inf);
          }

          // Testing images
          for(unsigned j=7; j<=9; j++)
            {
              ImgInfo inf;

              std::stringstream str;
              str << curdir << "/" << curobj <<".test,view" << j << ".png"; 

              inf.filename = str.str();
              inf.label = curobj;
              inf.labelnum = i;
              files_test.push_back(inf);
            }
        }

      return 0;
    }

    /// \brief Creates a training and a testing set from a generic set of directories with images
    ///
    /// \param base         Base directory of the dataset
    /// \param numClasses   Number of classes to test
    /// \param numViews     Number of images (per class) to use for the training set. The remaining images will be used for testing
    /// \param objects      Will contain a vector of strings listing the object classes being used
    /// \param files_train  Will contain a list of training objects containing filenames and corresponding labels 
    /// \param files_test   Will contain a list of testing objects containing filenames and corresponding labels 
    /// \param ext          Extension used for the images (default is "jpg")
    /// \param random       Whether the split should be randomised
    ///
    /// The images should be in the "base" directory, separated into directories. Each directory should contain
    /// images of one class and be named after that class. A directory can have any number of images. The "base"
    /// directory should contain no other files, only the image directories!
    ///
    /// No annotations are supported at the moment
    ///
    /// Returns 0 if all goes well
    int getGenericDatasets( std::string base, int numClasses, int numViews, std::vector<std::string> &objects,
                            std::vector<ImgInfo> &files_train, std::vector<ImgInfo> &files_test, std::string ext,  
                            bool random, int imagelimit )
    {
      // Get a list of objects
      std::vector<std::string> files = vislab::util::dirfiles(base);
      for(unsigned a=0; a<files.size(); a++)
        {
          std::string curobj = files[a];
          if(curobj.find(".") != 0) 
            {
              int pos = curobj.find_last_of("/");
              string classname = curobj.substr( pos+1, curobj.length()-pos );
              objects.push_back(classname);
            }
        }
      std::sort(objects.begin(), objects.end());

      // Now loop through all the objects and get the filenames with the selected orientations
      for(unsigned i=0; i<objects.size() && i<numClasses; i++)
        {
          std::string curobj = objects[i];
          std::string curdir = base + "/" + curobj + "/"; 

          std::vector<std::string> filenames = vislab::util::dirfiles(curdir,ext);
          int numfiles = filenames.size();
          if(numfiles > imagelimit) numfiles = imagelimit;

          // Get a list of views used for learning, or as example templates
          // The list has to be generated for each class separately, since objects have different number of images
          std::vector<int> learningViews = selectGenericViews( numViews, numfiles, random );

          // Sort the learning views to make lookups easier
          std::sort(learningViews.begin(), learningViews.end());

          for(int view=0; view<numfiles; view++)
            {
              ImgInfo inf;

              inf.filename = filenames[view].c_str();
              inf.label = curobj;
              inf.labelnum = i;

              // views reserved for learning
              if(std::binary_search(learningViews.begin(), learningViews.end(), view))
                files_train.push_back(inf);
              else     // the rest is for testing 
                files_test.push_back(inf);
            }
        }

      return 0;
    }

    int getGenericDetectionTests( std::string base, std::vector<ImgInfo> &files_test, 
                                  std::string ext, int imagelimit )
    {
      std::string curdir = base + "/"; 

      std::vector<std::string> filenames = vislab::util::dirfiles(curdir,ext);
      int numfiles = filenames.size();
      // std::cout << numfiles << std::endl;
      if(numfiles > imagelimit) numfiles = imagelimit;

      for(int view=0; view<numfiles; view++)
        {
          ImgInfo inf;

          inf.filename = filenames[view].c_str();
          inf.label = "none";
          inf.labelnum = -1;

          // views reserved for learning
          files_test.push_back(inf);
        }
    }

    int getUIUCDetectionTests( std::string base, std::vector<ImgInfo> &files_test, 
                               std::string ext, int imagelimit )
    {
      std::string curdir = base + "/";  

      // std::vector<std::string> filenames = vislab::util::dirfiles(curdir,ext);
      // int numfiles = filenames.size();
      // if(numfiles > imagelimit) numfiles = imagelimit;
      int numfiles = imagelimit;

      for(int view=0; view<numfiles; view++)
        {
          ImgInfo inf;

          stringstream filename;
          filename << base << "test-" << view << "." << ext;
          inf.filename = filename.str();
          inf.label = "none";
          inf.labelnum = -1;

          // views reserved for learning
          files_test.push_back(inf);
        }
    }

    float getClassificationRate( ConfusionMatrix array )
    {
      float result = 0;

      for(unsigned i=0; i<array.size(); i++)
        {
          float correct = 0;
          float all = 0;

          for(unsigned j=0; j<array[i].size(); j++)
            {
              all += array[i][j];
              if(i==j) correct = array[i][j];
            }
          if(all != 0) result += correct/all;
        }
      result /= array.size();

      return result;
    }


    /// \brief Saves a confusion matrix to a file in CSV format 
    ///
    /// \param array         Confusion matrix to be saved
    /// \param filename      Output filename
    /// \param labels        List of object labels, as returned by the \a get*Datasets functions

    void printConfusionMatrix( ConfusionMatrix array, std::string filename, std::vector<std::string> labels )
    {
      std::cout << "Writing the confusion matrix to " << filename << "..." << std::endl;
      std::ofstream file;
      file.open(filename.c_str());

      int numClasses = labels.size();

      for(int i=0; i<numClasses; i++)
        {
          file << ", " << labels[i];
        }
      file << std::endl;

      for(int i=0; i<numClasses; i++)
        {
          file << labels[i];

          for(int j=0; j<numClasses; j++)
            {
              file << ", " << array[i][j];
            }

          file << std::endl;
        }
      file << std::endl;

      file.close();
    }

  } //namespace evaluation
}//namespace vislab
