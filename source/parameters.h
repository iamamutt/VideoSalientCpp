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

      auto lines_win = odd_int(static_cast<uint>(sz * .02));
      auto flow_win  = odd_int(static_cast<uint>(sz * .125));

      lines = lines::Parameters(lines_win);
      flow  = flow::Parameters(flow_win);
    }
  }
};

// collection of general saliency parameters
struct ModelParameters
{
  int max_LoG_size;
  int n_LoG_kern;
  int gauss_blur_win;
  double contrast_factor;
  double central_focus_prop;
  double saliency_thresh;
  double saliency_thresh_mult;

  std::string debug_window_name = "FeatureMaps";
  bool toggle                   = true;

  // size dependent objects
  MatVec kernels_LoG;
  cv::Mat gauss_kern;
  cv::Mat central_mask;
  cv::Mat dilation_kernel;
  cv::Mat blank_image;

  // initialize parameters with defaults, some objects are left empty
  explicit ModelParameters(int _max_LoG_size            = -1,
                           int _n_LoG_kern              = -1,
                           int _gauss_blur_win          = -1,
                           double _contrast_factor      = 4,
                           double _central_focus_prop   = .67,
                           double _saliency_thresh      = -1,
                           double _saliency_thresh_mult = 2)
    : max_LoG_size(_max_LoG_size),
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
      std::cerr << "\n!!Image size is unspecified for model parameters" << std::endl;
      exit(1);
    };

    blank_image = imtools::make_black(dims);
    auto sz     = std::min(dims.height, dims.width);

    max_LoG_size = max_LoG_size < 1 ? sz / 4 : max_LoG_size;
    kernels_LoG  = imtools::LoG_kernels(max_LoG_size, n_LoG_kern);
    n_LoG_kern   = static_cast<int>(kernels_LoG.size());
    max_LoG_size = n_LoG_kern == 0 ? 0 : kernels_LoG[0].rows;
    toggle       = n_LoG_kern > 0;

    double _gauss_blur_win = gauss_blur_win < 1 ? sz * .015 : gauss_blur_win;
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

  ModelParameters user_pars(get_node_default<int>(pars_node["max_LoG_size"], -1),
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
    pars.flicker = flick::Parameters(get_node_default(flicker_node["lower_limit"], pars.flicker.lower_lim * 255),
                                     get_node_default(flicker_node["upper_limit"], pars.flicker.upper_lim * 255),
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
  // TODO: write descriptions of parameters for output file
  fs.writeComment("General saliency model parameters");
  fs << "model"
     << "{";
  fs.writeComment("Max LoG kernel window size. Set to -1 to use ~min(rows, cols)/4");
  fs << "max_LoG_size" << pars.max_LoG_size;
  fs << "n_LoG_kern" << pars.n_LoG_kern;
  fs << "gauss_blur_win" << pars.gauss_blur_win;
  fs << "contrast_factor" << pars.contrast_factor;
  fs << "central_focus_prop" << pars.central_focus_prop;
  fs << "saliency_thresh" << pars.saliency_thresh;
  fs << "saliency_thresh_mult" << pars.saliency_thresh_mult;
  fs << "}";
}

void
write_channel_parameters(cv::FileStorage &fs, const ChannelParameters &pars)
{
  // TODO: write descriptions of parameters for output file
  fs.writeComment("List of feature channel parameters");
  fs << "feature_channels"
     << "{";

  fs.writeComment("Luminance/Color parameters");
  std::string cspace;
  switch (pars.color.cspace) {
    case color::ColorSpace::LAB: cspace = "LAB"; break;
    case color::ColorSpace::RGB: cspace = "RGB"; break;
    default: cspace = "LAB";
  }
  fs << "color"
     << "{"
     << "colorspace" << cspace << "rescale" << pars.color.scale << "filter" << pars.color.shift << "weight"
     << pars.color.weight << "}";

  fs.writeComment("Line orientation parameters");
  auto gabor_win_size = pars.lines.gabor_pars.size.empty() ? -1 : pars.lines.gabor_pars.size.height;
  fs << "lines"
     << "{";
  fs.writeComment("Kernel size for square gabor patches. Set to -1 to use ~min(rows, cols) * .02");
  fs << "gabor_win_size" << gabor_win_size << "n_rotations" << (int)pars.lines.kernels.size() << "sigma"
     << pars.lines.gabor_pars.sigma << "lambda" << pars.lines.gabor_pars.lambda << "psi" << pars.lines.gabor_pars.psi
     << "gamma" << pars.lines.gabor_pars.gamma << "weight" << pars.lines.weight << "}";

  fs.writeComment("Motion flicker parameters");
  fs << "flicker"
     << "{"
     << "lower_limit" << pars.flicker.lower_lim * 255 << "upper_limit" << pars.flicker.upper_lim * 255 << "weight"
     << pars.flicker.weight << "}";

  fs.writeComment("Optical flow parameters");
  auto flow_window_size = pars.flow.lk_win_size.empty() ? -1 : pars.flow.lk_win_size.height;

  fs << "flow"
     << "{";
  fs.writeComment("Size of square flow estimation window. Set to -1 to use ~min(rows, cols) * .125");
  fs << "flow_window_size" << flow_window_size << "max_num_points" << pars.flow.max_n_pts << "min_point_dist"
     << pars.flow.min_pt_dist << "morph_half_win" << (pars.flow.dilate_shape.rows - 1) / 2 << "morph_iters"
     << pars.flow.dilate_iter << "weight" << pars.flow.weight << "}";

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
  pars.model.max_LoG_size         = -1;
  write_parameters(yaml_file, pars);
}
}  // namespace params

#endif  // SALIENCY_PARAMETERS_H
