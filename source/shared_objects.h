#ifndef SALIENCY_SHARED_OBJECTS_H
#define SALIENCY_SHARED_OBJECTS_H

#include "command_line.h"
#include "timing.h"
#include <fstream>
#include <future>

using Strings       = std::vector<std::string>;
using MatVec        = std::vector<cv::Mat>;
using VecFutureMats = std::vector<std::shared_future<cv::Mat>>;
using FutureMatVec  = std::shared_future<MatVec>;

inline const double BASE_IMAGE_LENGTH = 500;

struct Images
{
  cv::Mat I8UC3;    // cv image, 8-bit uchar 3-channel, 0-255 scale
  cv::Mat I8UC1;    // cv image, 8-bit uchar 1-channel, 0-255 scale
  cv::Mat I32FC3U;  // cv image, 32-bit float 3-channel, 0-1 unit scale
  cv::Mat I32FC1U;  // cv image, 32-bit float 1-channel, 0-1 unit scale
};

struct ImageSet
{
  Images prev;  // previous image set
  Images curr;  // current image set
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

struct ProgramStatus
{
  bool start_detection      = true;
  bool stop_detection       = false;
  bool static_image         = false;
  bool frame_was_captured   = true;
  bool right_mouse_btn_down = false;
  bool end_program          = false;
  bool export_enabled       = false;
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
  ProgramStatus status;
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

  [[nodiscard]] bool
  is_empty() const
  {
    return (luminance.empty() && color.empty() && lines.empty() && flicker.empty() && flow.empty());
  }
};

struct SaliencyMap
{
  cv::Mat map;
  cv::Mat image;
  cv::Mat map_8bit;
  cv::Mat binary_img;
  cv::Mat prev_map;
  double threshold;
  std::ofstream file;
  std::vector<std::vector<cv::Point2i>> contours;  // contour maps for blobs greater than threshold
  std::vector<float> salient_values;               // max saliency value within each blob/contour
  std::vector<cv::Point> salient_coords;           // locations of max salient point for each blob/contour
  std::vector<double> contour_size;                // number of pixels inside each salient blob/contour
  cv::Scalar black{0, 0, 0};                       // color of salient point
  cv::Scalar magenta{255, 0, 255};                 // color of salient contour
  cv::Scalar cyan{255, 255, 0};                    // color of other salient regions above threshold
};

template<typename T>
T
yml_node_value(const cv::FileNode &node, T default_value)
{
  return node.empty() ? default_value : static_cast<T>(node);
}

template<typename T>
T
yml_node_value(const cv::FileNode &node, T default_value, T invalid_value)
{
  if (node.empty()) {
    return default_value;
  } else {
    auto node_value = static_cast<T>(node);
    return node_value == invalid_value ? default_value : node_value;
  }
}

#endif  // SALIENCY_SHARED_OBJECTS_H
