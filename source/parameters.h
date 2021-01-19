#ifndef SALIENCY_PARAMETERS_H
#define SALIENCY_PARAMETERS_H

#include "channel_color.h"
#include "channel_flicker.h"
#include "channel_opticalflow.h"
#include "channel_orientation.h"

struct DefaultChanParams
{
  color::DefaultPars color;
  lines::DefaultPars lines;
  flick::DefaultPars flicker;
  flow::DefaultPars flow;

  DefaultChanParams() = default;

  explicit DefaultChanParams(const cv::Size &dims)
    : color(imtools::min_dim(dims)),
      lines(imtools::min_dim(dims)),
      flicker(imtools::min_dim(dims)),
      flow(imtools::min_dim(dims))
  {}

  explicit DefaultChanParams(const cv::FileNode &node, const cv::Size &dims)
    : color(node["color"], imtools::min_dim(dims)),
      lines(node["lines"], imtools::min_dim(dims)),
      flicker(node["flicker"], imtools::min_dim(dims)),
      flow(node["flow"], imtools::min_dim(dims))
  {
    if (node.empty()) std::cout << "\"feature_channels\" node not specified in parameters file" << std::endl;
  }
};

// collection of parameters for each feature channel
struct ChannelParameters
{
  color::Parameters color;
  lines::Parameters lines;
  flick::Parameters flicker;
  flow::Parameters flow;

  ChannelParameters() = default;

  // determine parameters and kernel sizes based on smallest image dimension
  explicit ChannelParameters(const DefaultChanParams &defaults)
    : color(defaults.color), flicker(defaults.flicker), flow(defaults.flow), lines(defaults.lines)
  {}
};

struct DefaultModelParams
{
  double max_LoG_prop         = 0.3334;
  int n_LoG_kern              = 3;
  double contrast_factor      = 2;
  double saliency_thresh      = -1;
  double saliency_thresh_mult = 2;
  double central_focus_prop   = .63;
  int gauss_blur_win          = 7;
  int morph_half_win          = 1;

  cv::Size_<int> dims = cv::Size(BASE_IMAGE_LENGTH, BASE_IMAGE_LENGTH);

  DefaultModelParams() = default;

  explicit DefaultModelParams(const cv::Size &_dims)
  {
    if (_dims.empty()) return;

    // adjust defaults based on user image size
    dims      = _dims;
    auto size = imtools::min_dim(dims);
    auto adj  = size / BASE_IMAGE_LENGTH;

    gauss_blur_win = static_cast<int>(round(gauss_blur_win * adj));
    morph_half_win = static_cast<int>(round(morph_half_win * adj));
  }

  explicit DefaultModelParams(const cv::FileNode &node, const cv::Size &dims) : DefaultModelParams(dims)
  {
    if (node.empty()) {
      std::cout << "\"model\" node not specified in parameters file" << std::endl;
      return;
    }

    max_LoG_prop         = yml_node_value<double>(node["max_LoG_prop"], max_LoG_prop);
    n_LoG_kern           = yml_node_value<int>(node["n_LoG_kern"], n_LoG_kern);
    gauss_blur_win       = yml_node_value<int>(node["gauss_blur_win"], gauss_blur_win, -1);
    contrast_factor      = yml_node_value<int>(node["contrast_factor"], contrast_factor);
    central_focus_prop   = yml_node_value<double>(node["central_focus_prop"], central_focus_prop);
    saliency_thresh      = yml_node_value<double>(node["saliency_thresh"], saliency_thresh);
    saliency_thresh_mult = yml_node_value<double>(node["saliency_thresh_mult"], saliency_thresh_mult);
  }
};

// collection of general saliency model parameters
struct ModelParameters
{
  // no constructor input
  std::string debug_window_name = "FeatureMaps";
  bool toggle                   = true;
  double image_len              = 0;

  double max_LoG_prop;
  int n_LoG_kern;
  int gauss_blur_win;
  double contrast_factor;
  double central_focus_prop;
  double saliency_thresh;
  double saliency_thresh_mult;

  // size dependent objects
  MatVec kernels_LoG;
  cv::Mat gauss_kern;
  cv::Mat central_mask;
  cv::Mat dilation_kernel;
  cv::Mat blank_image;

  // initialize parameters with defaults, some objects are left empty
  explicit ModelParameters(const DefaultModelParams &defaults = DefaultModelParams())
    : max_LoG_prop(defaults.max_LoG_prop),
      n_LoG_kern(defaults.n_LoG_kern),
      gauss_blur_win(defaults.gauss_blur_win),
      contrast_factor(defaults.contrast_factor),
      central_focus_prop(defaults.central_focus_prop),
      saliency_thresh(defaults.saliency_thresh),
      saliency_thresh_mult(defaults.saliency_thresh_mult)
  {
    image_len = static_cast<double>(imtools::min_dim(defaults.dims));
    if (image_len < 5) {
      std::cerr << "\n!!Image size: " << defaults.dims << " is too small" << std::endl;
      exit(1);
    };

    dilation_kernel = imtools::kernel_morph(defaults.morph_half_win);
    blank_image     = imtools::make_black(defaults.dims);
    central_mask    = imtools::get_border_mask(defaults.dims.width, defaults.dims.height, central_focus_prop);
    gauss_kern      = imtools::kernel_gauss_2d(odd_int(gauss_blur_win));

    imtools::update_LoG_kernel_data(kernels_LoG, max_LoG_prop, n_LoG_kern, image_len);

    toggle         = n_LoG_kern > 0;
    gauss_blur_win = gauss_kern.rows;
  }
};

// container holding channel and model parameters
struct Parameters
{
  ModelParameters model;
  ChannelParameters chan;
  float toggle_adj = 1;

  Parameters() : model(DefaultModelParams()), chan(DefaultChanParams()){};
  explicit Parameters(const cv::Size &dims) : model(DefaultModelParams(dims)), chan(DefaultChanParams(dims)) {}
  explicit Parameters(cv::FileStorage &fs, const cv::Size &dims)
    : model(DefaultModelParams(fs["model"], dims)), chan(DefaultChanParams(fs["feature_channels"], dims))
  {}
};

namespace params {

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
    "Set to -1 to calculate window size from image size.");
  fs << "gauss_blur_win" << pars.gauss_blur_win;

  fs.writeComment("Increase global contrast between high/low saliency.");
  fs << "contrast_factor" << pars.contrast_factor;

  fs.writeComment(
    "Focal area proportion. "
    "Proportion of image size used to attenuate outer edges of the image area.");
  fs << "central_focus_prop" << pars.central_focus_prop;

  fs.writeComment(
    "Threshold value to generate salient contours. "
    "Should be between 0 and 255. "
    "Set to -1 to use Otsu automatic thresholding.");
  fs << "saliency_thresh" << pars.saliency_thresh;

  fs.writeComment(
    "Threshold multiplier. "
    "Only applied to automatic threshold (i.e., saliency_thresh=-1).");
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
  std::string colorspace;
  switch (pars.color.cspace) {
    case color::ColorSpace::LAB: colorspace = "LAB"; break;
    case color::ColorSpace::RGB: colorspace = "RGB"; break;
    case color::ColorSpace::DKL: colorspace = "DKL"; break;
    default: colorspace = "DKL";
  }
  fs << "colorspace" << colorspace;

  fs.writeComment(
    "Scale parameter (k) for logistic function. "
    "Sharpens boundary between high/low intensity as value increases.");
  fs << "scale" << pars.color.scale;

  fs.writeComment(
    "Shift parameter (mu) for logistic function. "
    "This threshold cuts lower level intensity as this value increases.");
  fs << "shift" << pars.color.shift;

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
    "Set to -1 to calculate window size from image size.");
  auto kern_size = pars.lines.kern_size;
  fs << "kern_size" << kern_size;

  fs.writeComment(
    "Number of rotations used to create differently angled Gabor patches. "
    "N rotations are split evenly between 0 and 2pi.");
  fs << "n_rotations" << pars.lines.n_rotations;

  fs.writeComment(
    "Sigma parameter for Gabor filter. "
    "Adjusts frequency.");
  fs << "sigma" << pars.lines.sigma;

  fs.writeComment(
    "Lambda parameter for Gabor filter. "
    "Adjusts width.");
  fs << "lambda" << pars.lines.lambda;

  fs.writeComment(
    "Psi parameter for Gabor filter. "
    "Adjusts angle.");
  fs << "psi" << pars.lines.psi;

  fs.writeComment(
    "Gamma parameter for Gabor filter. "
    "Adjusts ratio.");
  fs << "gamma" << pars.lines.gamma;

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
    "Set to -1 to calculate window size from image size. "
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
  float n_final_maps = 5;             // number of channels in FeatureMaps
  float n_present    = n_final_maps;  // number of channels toggled on

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
initialize_parameters(const std::string &yaml_file, const cv::Size &dims)
{
  if (yaml_file.empty()) {
    // use default parameters adjusted for size of input image
    return Parameters(dims);
  } else {
    // return user specified parameters
    auto file_store = open_yaml_reader(yaml_file);
    Parameters pars(file_store, dims);
    file_store.release();
    return pars;
  }
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
  pars.chan.flow.lk_win_size = cv::Size();
  pars.chan.lines.kern_size  = -1;
  pars.model.gauss_blur_win  = -1;
  pars.model.saliency_thresh = -1;
  write_parameters(yaml_file, pars);
}
}  // namespace params

#endif  // SALIENCY_PARAMETERS_H
