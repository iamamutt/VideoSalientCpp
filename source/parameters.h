#ifndef SALIENCY_PARAMETERS_H
#define SALIENCY_PARAMETERS_H

#include "channel_color.h"
#include "channel_flicker.h"
#include "channel_opticalflow.h"
#include "channel_orientation.h"

// collection of parameters for each feature channel
struct ChannelParameters
{
  color::Parameters color;
  lines::Parameters lines;
  flick::Parameters flicker;
  flow::Parameters flow;

  ChannelParameters() = default;
  explicit ChannelParameters(const cv::Size &dims) : color(), flicker()
  {
    // determine kernel sizes based on image dimensions
    if (!dims.empty()) {
      auto sz = std::min(dims.height, dims.width);

      auto lines_win = odd_int(static_cast<uint>(sz * .025));
      auto flow_win  = odd_int(static_cast<uint>(sz * .125));

      lines = lines::Parameters(lines_win);
      flow  = flow::Parameters(flow_win);
    }
  }
};

void
update_LoG_kernel_data(MatVec &kernels_LoG,
                       double &max_LoG_prop,
                       int &n_LoG_kern,
                       const double &length,
                       double min_k = 7,
                       double max_p = 2)
{
  double max_prop       = std::min(std::max((min_k / length), max_LoG_prop), max_p);
  uint max_LoG_win_size = odd_int(static_cast<int>(max_prop * length));
  kernels_LoG           = imtools::kernels_lap_of_gauss(max_LoG_win_size, n_LoG_kern);
  n_LoG_kern            = static_cast<int>(kernels_LoG.size());
  max_LoG_prop          = n_LoG_kern == 0 ? 0. : (kernels_LoG[0].rows / length);
}

// collection of general saliency parameters
struct ModelParameters
{
  double max_LoG_prop;
  int n_LoG_kern;
  int gauss_blur_win;
  double contrast_factor;
  double central_focus_prop;
  double saliency_thresh;
  double saliency_thresh_mult;

  // no constructor input
  double image_len              = 0;
  std::string debug_window_name = "FeatureMaps";
  bool toggle                   = true;

  // size dependent objects
  MatVec kernels_LoG;
  cv::Mat gauss_kern;
  cv::Mat central_mask;
  cv::Mat dilation_kernel;
  cv::Mat blank_image;

  // initialize parameters with defaults, some objects are left empty
  explicit ModelParameters(double _max_LoG_prop         = 0.5,
                           int _n_LoG_kern              = 3,
                           int _gauss_blur_win          = 5,
                           double _contrast_factor      = 2,
                           double _central_focus_prop   = .63,
                           double _saliency_thresh      = -1,
                           double _saliency_thresh_mult = 1.5)
    : max_LoG_prop(_max_LoG_prop),
      n_LoG_kern(_n_LoG_kern),
      gauss_blur_win(_gauss_blur_win),
      contrast_factor(_contrast_factor),
      central_focus_prop(_central_focus_prop),
      saliency_thresh(_saliency_thresh),
      saliency_thresh_mult(_saliency_thresh_mult){};

  // construct size-dependent objects and update default values
  explicit ModelParameters(const cv::Size &dims, ModelParameters pars = ModelParameters())
    : ModelParameters(std::move(pars))
  {
    if (dims.empty()) {
      std::cerr << "\n!!Image size is unspecified for ModelParameters constructor" << std::endl;
      exit(1);
    };

    blank_image = imtools::make_black(dims);
    image_len   = static_cast<double>(std::min(dims.height, dims.width));

    update_LoG_kernel_data(kernels_LoG, max_LoG_prop, n_LoG_kern, image_len);
    toggle = n_LoG_kern > 0;

    double _gauss_blur_win = gauss_blur_win <= 0 ? image_len * .01 : gauss_blur_win;
    gauss_kern             = imtools::kernel_gauss_2d(odd_int(_gauss_blur_win));
    gauss_blur_win         = gauss_kern.cols;

    central_mask = imtools::get_border_mask(dims.width, dims.height, central_focus_prop);

    dilation_kernel = cv::Mat::ones(3, 3, CV_8U);
  }
};

// container holding channel and model parameters
struct Parameters
{
  ModelParameters model;
  ChannelParameters chan;
  float toggle_adj = 1;

  Parameters() = default;
  explicit Parameters(const cv::Size &dims) : model(dims), chan(dims) {}
  explicit Parameters(ModelParameters _model, ChannelParameters _chan)
    : model(std::move(_model)), chan(std::move(_chan))
  {}
};

namespace params {

template<typename T>
T
get_node_default(const cv::FileNode &node, T default_value)
{
  return node.empty() ? default_value : static_cast<T>(node);
}

template<typename T>
T
get_node_default(const cv::FileNode &node, T default_value, T invalid_value)
{
  if (node.empty()) {
    return default_value;
  } else {
    auto node_value = static_cast<T>(node);
    return node_value == invalid_value ? default_value : node_value;
  }
}

ModelParameters
read_model_parameters(cv::FileStorage &fs, const cv::Size &dims)
{
  ModelParameters pars(dims);

  auto pars_node = fs["model"];
  if (pars_node.empty()) {
    std::cout << "no \"model\" node specified in parameters file" << std::endl;
    return pars;
  }

  ModelParameters user_pars(get_node_default<double>(pars_node["max_LoG_prop"], pars.max_LoG_prop),
                            get_node_default<int>(pars_node["n_LoG_kern"], -1),
                            get_node_default<int>(pars_node["gauss_blur_win"], -1),
                            get_node_default<int>(pars_node["contrast_factor"], pars.contrast_factor),
                            get_node_default<double>(pars_node["central_focus_prop"], pars.central_focus_prop),
                            get_node_default<double>(pars_node["saliency_thresh"], pars.saliency_thresh),
                            get_node_default<double>(pars_node["saliency_thresh_mult"], pars.saliency_thresh_mult));

  return ModelParameters(dims, user_pars);
}

ChannelParameters
read_channel_parameters(cv::FileStorage &fs, const cv::Size &dims)
{
  ChannelParameters pars(dims);

  auto pars_node = fs["feature_channels"];
  if (pars_node.empty()) {
    std::cout << "no \"feature_channels\" node specified in parameters file" << std::endl;
    return pars;
  }

  // read color channel parameters
  auto color_node = pars_node["color"];
  if (!color_node.empty()) {
    pars.color = color::Parameters(get_node_default<std::string>(color_node["colorspace"], "LAB"),
                                   get_node_default(color_node["rescale"], pars.color.scale),
                                   get_node_default(color_node["filter"], pars.color.shift),
                                   get_node_default(color_node["weight"], pars.color.weight));
  }

  // read lines channel parameters
  auto lines_node = pars_node["lines"];
  if (!lines_node.empty()) {
    pars.lines = lines::Parameters(
      get_node_default(lines_node["gabor_win_size"], pars.lines.gabor_pars.size.height, -1),
      get_node_default(lines_node["n_rotations"], (int)pars.lines.kernels.size()),
      get_node_default(lines_node["sigma"], pars.lines.gabor_pars.sigma),
      get_node_default(lines_node["lambda"], pars.lines.gabor_pars.lambda),
      get_node_default(lines_node["psi"], pars.lines.gabor_pars.psi),
      get_node_default(lines_node["gamma"], pars.lines.gabor_pars.gamma),
      get_node_default(lines_node["weight"], pars.lines.weight));
  }

  // read flick channel parameters
  auto flicker_node = pars_node["flicker"];
  if (!flicker_node.empty()) {
    pars.flicker = flick::Parameters(get_node_default(flicker_node["lower_limit"], pars.flicker.lower_lim),
                                     get_node_default(flicker_node["upper_limit"], pars.flicker.upper_lim),
                                     get_node_default(flicker_node["weight"], pars.flicker.weight));
  }

  // read flow channel parameters
  auto flow_node = pars_node["flow"];
  if (!flow_node.empty()) {
    auto flow_dilate_sz = (pars.flow.dilate_shape.rows - 1) / 2;
    pars.flow = flow::Parameters(get_node_default(flow_node["flow_window_size"], pars.flow.lk_win_size.height, -1),
                                 get_node_default(flow_node["max_num_points"], pars.flow.max_n_pts),
                                 get_node_default(flow_node["min_point_dist"], pars.flow.min_pt_dist),
                                 get_node_default(flow_node["morph_half_win"], flow_dilate_sz),
                                 get_node_default(flow_node["morph_iters"], pars.flow.dilate_iter),
                                 get_node_default(flow_node["weight"], pars.flow.weight));
  }

  return pars;
}

void
write_model_parameters(cv::FileStorage &fs, const ModelParameters &pars)
{
  fs.writeComment("--------------------------------------------------------------------------------------");
  fs.writeComment("General saliency model parameters");
  fs.writeComment("--------------------------------------------------------------------------------------");
  fs << "model"
     << "{";

  fs.writeComment(
    "Proportion of the image size used as the max LoG kernel size. "
    "Each kernel will be half the size of the previous.");
  fs << "max_LoG_prop" << pars.max_LoG_prop;

  fs.writeComment(
    "Number of LoG kernels. "
    "Set to -1 to get as many kernels as possible, i.e., until the smallest size is reached. "
    "Set to 0 to turn off all LoG convolutions.");
  fs << "n_LoG_kern" << pars.n_LoG_kern;

  fs.writeComment(
    "Window size for amount of blur applied to saliency map. "
    "Set to -1 to use ~min(rows, cols) * .01.");
  fs << "gauss_blur_win" << pars.gauss_blur_win;

  fs.writeComment("Increase global contrast between high/low saliency.");
  fs << "contrast_factor" << pars.contrast_factor;

  fs.writeComment(
    "Focal area proportion. "
    "Proportion of image size used to attenuate outer edges of the image area.");
  fs << "central_focus_prop" << pars.central_focus_prop;

  fs.writeComment(
    "Threshold value to generate salient contours. "
    "Should be between 0 and 255.");
  fs << "saliency_thresh" << pars.saliency_thresh;

  fs.writeComment(
    "Threshold multiplier. "
    "Only for automized Otsu saliency threshold (i.e., saliency_thresh=-1).");
  fs << "saliency_thresh_mult" << pars.saliency_thresh_mult;

  fs << "}";  // end model parameters
}

void
write_channel_parameters(cv::FileStorage &fs, const ChannelParameters &pars)
{
  fs.writeComment("--------------------------------------------------------------------------------------");
  fs.writeComment("List of parameters for each feature map channel");
  fs.writeComment("--------------------------------------------------------------------------------------");
  fs << "feature_channels"
     << "{";

  // chromatic feature maps -----------------------------------------------------------------------
  fs.writeComment("Luminance/Color parameters --------------------------------------------------------");
  fs << "color"
     << "{";

  fs.writeComment(
    "Color space to use as starting point for extracting luminance and color. "
    "Should be either \"DKL\", \"LAB\", or \"RGB\".");
  std::string cspace;
  switch (pars.color.cspace) {
    case color::ColorSpace::LAB: cspace = "LAB"; break;
    case color::ColorSpace::RGB: cspace = "RGB"; break;
    case color::ColorSpace::DKL: cspace = "DKL"; break;
    default: cspace = "DKL";
  }
  fs << "colorspace" << cspace;

  fs.writeComment(
    "Scale parameter (k) for logistic function. "
    "Sharpens boundary between high/low intensity as value increases.");
  fs << "rescale" << pars.color.scale;

  fs.writeComment(
    "Shift parameter (mu) for logistic function. "
    "This threshold cuts lower level intensity as this value increases.");
  fs << "filter" << pars.color.shift;

  fs.writeComment(
    "Weight applied to all pixels in each map/image. "
    "Set to 0 to toggle channel off.");
  fs << "weight" << pars.color.weight;

  fs << "}";  // end color

  // line orientations ----------------------------------------------------------------------------
  fs.writeComment("Line orientation parameters -------------------------------------------------------");
  fs << "lines"
     << "{";

  fs.writeComment(
    "Kernel size for square gabor patches. "
    "Set to -1 to use ~min(rows, cols) * .025");
  auto gabor_win_size = pars.lines.gabor_pars.size.empty() ? -1 : pars.lines.gabor_pars.size.height;
  fs << "gabor_win_size" << gabor_win_size;

  fs.writeComment(
    "Number of rotations used to create differently angled Gabor patches. "
    "N rotations are split evenly between 0 and 2pi.");
  fs << "n_rotations" << (int)pars.lines.kernels.size();

  fs.writeComment(
    "Sigma parameter for Gabor filter. "
    "Adjusts frequency.");
  fs << "sigma" << pars.lines.gabor_pars.sigma;

  fs.writeComment(
    "Lambda parameter for Gabor filter. "
    "Adjusts width.");
  fs << "lambda" << pars.lines.gabor_pars.lambda;

  fs.writeComment(
    "Psi parameter for Gabor filter. "
    "Adjusts angle.");
  fs << "psi" << pars.lines.gabor_pars.psi;

  fs.writeComment(
    "Gamma parameter for Gabor filter. "
    "Adjusts ratio.");
  fs << "gamma" << pars.lines.gabor_pars.gamma;

  fs.writeComment(
    "Weight applied to all pixels in each map/image. "
    "Set to 0 to toggle channel off.");
  fs << "weight" << pars.lines.weight;

  fs << "}";  // end lines

  // motion flicker -------------------------------------------------------------------------------
  fs.writeComment("Motion flicker parameters ---------------------------------------------------------");
  fs << "flicker"
     << "{";

  fs.writeComment(
    "Cutoff value for minimum change in image contrast. "
    "Value should be between 0 and 1.");
  fs << "lower_limit" << pars.flicker.lower_lim;

  fs.writeComment(
    "Cutoff value for maximum change in image contrast. "
    "Value should be between 0 and 1.");
  fs << "upper_limit" << pars.flicker.upper_lim;

  fs.writeComment(
    "Weight applied to all pixels in each map/image. "
    "Set to 0 to toggle channel off.");
  fs << "weight" << pars.flicker.weight;

  fs << "}";  // end flicker

  // optical flow ---------------------------------------------------------------------------------
  fs.writeComment("Optical flow parameters -----------------------------------------------------------");
  fs << "flow"
     << "{";

  fs.writeComment(
    "Size of square window for sparse flow estimation. "
    "Set to -1 to use ~min(rows, cols) * .125. "
    "Setting this to a smaller value generates higher flow intensity but at the cost of accuracy.");
  auto flow_window_size = pars.flow.lk_win_size.empty() ? -1 : pars.flow.lk_win_size.height;
  fs << "flow_window_size" << flow_window_size;

  fs.writeComment("Maximum number of allotted points used to estimate flow between frames. ");
  fs << "max_num_points" << pars.flow.max_n_pts;

  fs.writeComment("Minimum distance between new points used to estimate flow. ");
  fs << "min_point_dist" << pars.flow.min_pt_dist;

  fs.writeComment("Half size of the dilation/erosion kernel used to expand flow points. ");
  fs << "morph_half_win" << (pars.flow.dilate_shape.rows - 1) / 2;

  fs.writeComment(
    "Number of iterations for the morphology operations. "
    "This will perform N dilations and N/2 erosion steps.");
  fs << "morph_iters" << pars.flow.dilate_iter;

  fs.writeComment(
    "Weight applied to all pixels in each map/image. "
    "Set to 0 to toggle channel off.");
  fs << "weight" << pars.flow.weight;

  fs << "}";  // end flow

  // end feature_channels
  fs << "}";
}

cv::FileStorage
open_yaml_reader(const std::string &yaml_file)
{
  cv::FileStorage fs(yaml_file, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cerr << "\n!!Failed to open file: \"" << yaml_file << "\"" << std::endl;
    exit(1);
  }
  std::cout << "Reading YAML parameters from file: \"" << yaml_file << "\"" << std::endl;
  return fs;
}

void
update_pars_weights(Parameters &pars, bool static_image)
{
  float n_final_maps = 5;  // number of slots in FeatureMaps
  float n_present    = n_final_maps;

  if (static_image) {
    // turn off these channels for static images
    pars.chan.flicker.weight = 0.f;
    pars.chan.flow.weight    = 0.f;
  }

  if (pars.chan.color.weight == 0) {
    pars.chan.color.toggled = false;
    n_present -= 2.f;
  }
  if (pars.chan.lines.weight == 0) {
    pars.chan.lines.toggled = false;
    n_present--;
  }
  if (pars.chan.flicker.weight == 0) {
    pars.chan.flicker.toggled = false;
    n_present--;
  }
  if (pars.chan.flow.weight == 0) {
    pars.chan.flow.toggled = false;
    n_present--;
  }
  pars.toggle_adj = n_final_maps / n_present;
}

Parameters
initialize_parameters(const std::string &yaml_file, const cv::Size &dims, bool is_static)
{
  Parameters pars;

  if (yaml_file.empty()) {
    // return default parameters based on image size
    pars = Parameters(dims);
  } else {
    // return user specified parameters
    auto fs = open_yaml_reader(yaml_file);
    pars    = Parameters(read_model_parameters(fs, dims), read_channel_parameters(fs, dims));
    fs.release();
  }

  update_pars_weights(pars, is_static);
  return pars;
}

void
write_parameters(const std::string &yaml_file, const Parameters &pars)
{
  cv::FileStorage fs(yaml_file, cv::FileStorage::WRITE);
  write_model_parameters(fs, pars.model);
  write_channel_parameters(fs, pars.chan);
  fs.release();
}

void
parameter_defaults(const std::string &yaml_file)
{
  Parameters pars;
  // set to empty so that sizes can be determined by image
  pars.chan.flow.lk_win_size      = cv::Size();
  pars.chan.lines.gabor_pars.size = cv::Size();
  write_parameters(yaml_file, pars);
}
}  // namespace params

#endif  // SALIENCY_PARAMETERS_H
