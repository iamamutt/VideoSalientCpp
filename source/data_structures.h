#ifndef SALIENCY_DATA_STRUCTURES_H
#define SALIENCY_DATA_STRUCTURES_H

#include "command_line.h"
#include "timing.h"
#include <future>

using Strings       = std::vector<std::string>;
using MatVec        = std::vector<cv::Mat>;
using VecFutureMats = std::vector<std::shared_future<cv::Mat>>;
using FutureMatVec  = std::shared_future<MatVec>;

struct Images
{
  cv::Mat I8UC3;    // cv image, 8-bit uchar 3-channel, 0-255 scale
  cv::Mat I8UC1;    // cv image, 8-bit uchar 1-channel, 0-255 scale
  cv::Mat I32FC3U;  // cv image, 32-bit float 3-channel, 0-1 unit scale
  cv::Mat I32FC1U;  // cv image, 32-bit float 1-channel, 0-1 unit scale
};

struct ImageSet
{
  Images prev;    // previous image set
  Images curr;    // current image set
};

struct ImageDims
{
  cv::Size size;
  cv::Size resize;
};

struct FPSCounter
{
  timing::DefaultTimePoint clock;
  int frame;
  double fps;
  std::vector<double> fps_buff;
  size_t fps_buff_size;
  int print_after;
};

struct DisplayData
{
  cv::Size layout;      // number of rows and columns in final display grid
  cv::Size size;        // size of the display grid image
  double scale = 1;     // scaling factor for each image in display grid
  std::string winname;  // display window name
};

struct GridLayouts
{
  DisplayData main;      // window that displays input and saliency output images
  DisplayData color;     // window that displays three color images
  DisplayData lines;     // window that displays all line orientation images
  DisplayData flicker;   // window that displays flicker image
  DisplayData flow;      // window that displays all flow direction images
  DisplayData features;  // window that displays features after activation
};

struct Source
{
  cv::VideoCapture cap;
  cv::VideoWriter vid;
  ImageSet img;
  FPSCounter fps;
  ImageDims dim;
  CmdLineOpts opts;
  GridLayouts layouts;
};

struct ChannelImages
{
  MatVec color;    // stores luminance and color intensity images
  MatVec lines;    // stores several images based on rotated line orientations
  MatVec flicker;  // stores single image, temporal change in contrast
  MatVec flow;     // stores directional sparse optical flow images
};

// flat feature maps, one image per channel
struct FeatureMaps
{
  cv::Mat luminance;
  cv::Mat color;
  cv::Mat lines;
  cv::Mat flicker;
  cv::Mat flow;
};

#endif  // SALIENCY_DATA_STRUCTURES_H
