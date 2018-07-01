#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <vector>
#include <iostream>
#include <dirent.h>
#include <cstdlib>
#include <cmath>

using namespace std;

namespace vislab
{
  namespace util
  {

/*! \file util.h
    \brief Very useful functions to have around */

/// Split a string into tokens, using a specified list of separators
inline void split(string & text, string & separators, vector<string> & words)
{
    int n = text.length();
    int start, stop;
    start = text.find_first_not_of(separators);
    while ((start >= 0) && (start < n))
    {
        stop = text.find_first_of(separators, start);
        if ((stop < 0) || (stop > n)) stop = n;
        words.push_back(text.substr(start, stop - start));
        start = text.find_first_not_of(separators, stop+1);
    }
}

/// Deliver a list of all filenames in a given directory
inline vector<string> dirfiles( string filename )
{
    DIR *inputdir;
    struct dirent *dirinfo;

    string directory;
    string extension;
    string tempstr;
    string filebase;

    directory = filename;
    directory += "/";

    vector<string> result;

    if(( inputdir = opendir( directory.c_str() ))==NULL) 
    {
        std::cout << "Error opening the directory" << directory << std::endl;
        // exit(1);
        return result;
    }

    // Start with the first image file in the directory...
    while(( dirinfo = readdir(inputdir)) != NULL ) {
        filename = dirinfo->d_name;
        if(filename == "." || filename == "..") continue;

        result.push_back(directory+filename);
    }

    return result;
}

/// Deliver a list of all filenames in a given directory, recursively
inline vector<string> dirfiles_recursive( string filename )
{
    vector<string> result;
    vector<string> allfiles;
    allfiles.push_back(filename);

    int counter = 0;
    int size = allfiles.size();

    vector<string> curdir;

    // Try to enter each file in the list and if it works, add the list of the files within that 
    // directory to the whole list
    do
    {
        curdir = dirfiles(allfiles[counter]);
        if(curdir.empty()) continue;

        std::copy( curdir.begin(), curdir.end(), std::back_inserter( allfiles ) );

        size = allfiles.size();
    }
    while(++counter<size);

    return allfiles;
}

inline vector<string> filterfiles( vector<string> files, string ext)
{
    vector<string> result;

    for(unsigned i=0; i<files.size(); i++)
    {
        string curfile = files[i];
        int pos = curfile.find_last_of(".");

        string extension = curfile.substr( pos+1, curfile.length()-pos );
//        if(extension.empty()) continue;
        
        for( string::iterator p = extension.begin(); p != extension.end(); ++p)
            *p = tolower(*p);
        
        if( extension == ext ) result.push_back(curfile);
    }
    
    return result;
}

inline vector<string> dirfiles( string filename, string ext ) 
{
    vector<string> files = dirfiles( filename );

    return filterfiles(files, ext);
}

inline vector<string> dirfiles_recursive( string filename, string ext )
{
    vector<string> files = dirfiles_recursive( filename );

    return filterfiles(files, ext);
}

/// Mass search and replace in a vector of strings
inline vector<string> replacestrings( vector<string> strings, std::string orig, std::string newstr)
{
    for(unsigned i=0; i<strings.size(); i++)
        strings[i].replace(strings[i].find(orig),orig.length(),newstr); 

    return strings;
}

/// Random number between 0 and 1
inline double randd()
{
    return (double) rand()/ (double) RAND_MAX;
}

/// Random number drawn from a gaussian distribution
inline double randomgauss()
{
    double x1, x2, w, y1; //, y2;

    do {
        x1 = 2.0 * randd() - 1.0;
        x2 = 2.0 * randd() - 1.0;
        w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 );

    w = sqrt( (-2.0 * log( w ) ) / w );
    y1 = x1 * w;
    // y2 = x2 * w;

    return y1;
}

/// replace all instances of victim with replacement
inline std::string mass_replace(const std::string &source, const std::string &victim, const std::string &replacement)
{
    std::string answer = source;
    std::string::size_type j = 0;
    while ((j = answer.find(victim, j)) != std::string::npos )
        answer.replace(j, victim.length(), replacement);
    return answer;
}

/// Indirect sorting of arrays, for use with std::sort
template <class T>
class gt_indirectsort {
    std::vector<T> _x;
    public:
    gt_indirectsort( std::vector<T> x ) : _x(x) {}
    bool operator()( int j, int k ) const { return _x[j] > _x[k]; }
};

/// Indirect sorting of arrays, for use with std::sort, only inverse
template <class T>
class lt_indirectsort {
    std::vector<T> _x;
    public:
    lt_indirectsort( std::vector<T> x ) : _x(x) {}
    bool operator()( int j, int k ) const { return _x[j] < _x[k]; }
};


      inline vector<string> readDir(const string& path)
      {
	  vector<string> res;
	  DIR *d;
	  struct dirent *dir;
	  d = opendir(path.c_str());
	  if (d) {
	      while ((dir = readdir(d)) != NULL) {
		  string name = dir->d_name;
		  if (name != "." && name != "..") {
		      res.push_back(name);
		  }
	      }
	      closedir(d);
	  }
	  return res;
      }
      

  }//namespace util
}//namespace vislab

#endif // UTIL_H
