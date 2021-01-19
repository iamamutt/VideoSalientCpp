#ifndef SALIENCY_CHANNEL_COLOR_H
#define SALIENCY_CHANNEL_COLOR_H

#include "cv_tools.h"

namespace color {

enum class ColorSpace
{
  DKL = 0,
  LAB = 1,
  RGB = 2
};

MatVec
luma_chroma_compress(MatVec &images, const float &k, const float &m, const float &scale)
{
  for (auto &&col : images) {
    // convert to logit
    imtools::truncated_logit(col);

    // saturate
    col = imtools::logistic(col, 1, k, m);

    // additional scaling based on white point
    if (scale == 1) continue;
    col *= scale;
  }

  return images;
}

MatVec
cspace_lab(const cv::Mat &I32FC3U)
{
  cv::Mat lab_img;
  // CIELAB requires 3-channel float [0,1] scaled image
  cv::cvtColor(I32FC3U, lab_img, cv::COLOR_BGR2Lab);

  // returned scales: 0:100, -127:127, -127:127
  auto [lightness, red_v_grn, blu_v_ylw] = imtools::split_bgr(lab_img);

  // rescale to be in range [0,1]
  lightness /= 100.f;

  // fold absolute intensity for competing color channels, rescale in range [0,1]
  red_v_grn = cv::abs(red_v_grn) / 127.f;
  blu_v_ylw = cv::abs(blu_v_ylw) / 127.f;

  return {lightness, red_v_grn, blu_v_ylw};
}

MatVec
cspace_rgb(const cv::Mat &I32FC3U)
{
  auto [blue, green, red] = imtools::split_bgr(I32FC3U);

  // all in range [0,1]
  cv::Mat lightness = (blue * 0.0722f + red * 0.2126f + green * 0.7152f);
  cv::Mat red_v_grn = cv::abs(red - green);
  cv::Mat blu_v_ylw = cv::abs(blue - ((red + green) * 0.5f));

  return {lightness, red_v_grn, blu_v_ylw};
}

MatVec
cspace_dkl(const cv::Mat &I32FC3U)
{
  auto [lightness, red_v_grn, blu_v_ylw] = imtools::bgr32FC3U_to_DKL(I32FC3U);

  // adjust to -1,1
  red_v_grn *= 1.27003572f;
  blu_v_ylw *= 1.10502244f;

  // fold absolute intensity for competing color channels
  red_v_grn = cv::abs(red_v_grn);
  blu_v_ylw = cv::abs(blu_v_ylw);

  return {lightness, red_v_grn, blu_v_ylw};
}

float
white_pt_norm(const ColorSpace &cspace, const float &scale, const float &shift)
{
  cv::Mat white_box(5, 5, CV_32FC3);
  white_box = cv::Scalar_<float>::all(1);

  MatVec cspace_imgs;
  switch (cspace) {
    case ColorSpace::LAB: cspace_imgs = cspace_lab(white_box); break;
    case ColorSpace::RGB: cspace_imgs = cspace_rgb(white_box); break;
    case ColorSpace::DKL: cspace_imgs = cspace_dkl(white_box); break;
  }

  cspace_imgs = luma_chroma_compress(cspace_imgs, scale, shift, 1);

  // use luma channel only for white norm value
  auto max_v = imtools::global_max(cspace_imgs[0]);
  max_v      = 1 / (round(max_v * 1e6) / 1e6);

  return static_cast<float>(max_v);
}

struct DefaultPars
{
  std::string colorspace = "dkl";
  float scale            = 1;
  float shift            = 0.5;
  float weight           = 1;

  DefaultPars() = default;

  explicit DefaultPars(int) {}

  explicit DefaultPars(const cv::FileNode &node, int size = 0) : DefaultPars(size)
  {
    if (node.empty()) return;

    colorspace = yml_node_value<std::string>(node["colorspace"], "LAB");
    scale      = yml_node_value(node["scale"], scale);
    shift      = yml_node_value(node["shift"], shift);
    weight     = yml_node_value(node["weight"], weight);
  }
};

struct Parameters
{
  std::string debug_window_name = "ColorChannel";
  bool toggled                  = true;

  float weight;
  float scale;
  float shift;

  ColorSpace cspace;
  float norm_value;

  explicit Parameters(const DefaultPars &defaults = DefaultPars())
    : scale(std::max(0.5f, defaults.scale)), shift(defaults.shift), weight(defaults.weight)
  {

    if (to_lower(defaults.colorspace) == "lab") {
      cspace = ColorSpace::LAB;
    } else if (to_lower(defaults.colorspace) == "rgb") {
      cspace = ColorSpace::RGB;
    } else if (to_lower(defaults.colorspace) == "dkl") {
      cspace = ColorSpace::DKL;
    } else {
      std::cerr << "Invalid colorspace value, "
                   "defaulting to \"DKL\""
                << std::endl;
      cspace = ColorSpace::DKL;
    }

    norm_value = white_pt_norm(cspace, scale, shift);
  };
};

MatVec
detect(const cv::Mat &I32FC3U, const color::Parameters &pars)
{
  MatVec cspace_imgs;
  if (!pars.toggled) return cspace_imgs;

  switch (pars.cspace) {
    case ColorSpace::DKL: cspace_imgs = cspace_dkl(I32FC3U); break;
    case ColorSpace::LAB: cspace_imgs = cspace_lab(I32FC3U); break;
    case ColorSpace::RGB: cspace_imgs = cspace_rgb(I32FC3U); break;
  }

  cspace_imgs = luma_chroma_compress(cspace_imgs, pars.scale, pars.shift, pars.norm_value);

  if (pars.weight == 1) return cspace_imgs;
  for (auto &&img : cspace_imgs) img *= pars.weight;

  return cspace_imgs;
}

// *******************************************************
// Interactively select parameters and display adjustments
// *******************************************************
namespace debug {
  void
  callback_scale(int pos, void *user_data)
  {
    auto *pars       = (Parameters *)user_data;
    pars->scale      = static_cast<float>(pos);
    pars->norm_value = white_pt_norm(pars->cspace, pars->scale, pars->shift);
    cv::setTrackbarPos("scale", pars->debug_window_name, pos);
  }

  void
  callback_shift(int pos, void *user_data)
  {
    auto *pars       = (Parameters *)user_data;
    pars->shift      = static_cast<float>(pos) * .01f;
    pars->norm_value = white_pt_norm(pars->cspace, pars->scale, pars->shift);
    cv::setTrackbarPos("shift", pars->debug_window_name, pos);
  }

  void
  callback_cspace(int pos, void *user_data)
  {
    auto *pars       = (Parameters *)user_data;
    pars->cspace     = static_cast<color::ColorSpace>(pos);
    pars->norm_value = white_pt_norm(pars->cspace, pars->scale, pars->shift);
    cv::setTrackbarPos("cspace", pars->debug_window_name, pos);
  }

  struct TrackbarPositions
  {
    int scale;
    int shift;
    int cspace;

    explicit TrackbarPositions(const color::Parameters &defaults = color::Parameters())
    {
      scale  = static_cast<int>(defaults.scale);
      shift  = static_cast<int>(defaults.shift * 100.);
      cspace = static_cast<int>(defaults.cspace);
    }
  };

  void
  create_trackbar(color::debug::TrackbarPositions *notches, color::Parameters *pars)
  {
    if (!pars->toggled) return;
    cv::namedWindow(pars->debug_window_name, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("scale", pars->debug_window_name, &notches->scale, 50, &callback_scale, pars);
    cv::setTrackbarMin("scale", pars->debug_window_name, 1);
    cv::createTrackbar("shift", pars->debug_window_name, &notches->shift, 100, &callback_shift, pars);
    cv::createTrackbar("cspace", pars->debug_window_name, &notches->cspace, 2, &callback_cspace, pars);
  }

  std::vector<Strings>
  texify_pars(const color::Parameters &pars)
  {
    Strings chan_type = {"Luminance", "Red vs. Green", "Blue vs. Yellow"};

    std::vector<Strings> each_image{3};
    for (int n = 0; n < 3; ++n) {
      std::stringstream par1;
      par1.precision(4);
      par1 << "scale: " << pars.scale;

      std::stringstream par2;
      par2.precision(4);
      par2 << "shift: " << pars.shift;

      each_image[n] = {chan_type[n], par1.str(), par2.str()};
    };

    return each_image;
  }

  void
  visualize(const MatVec &img_channels, const color::Parameters &pars, const cv::Size &resize, const DisplayData &disp)
  {
    if (img_channels.empty() || !pars.toggled) return;
    MatVec colorized_lab;
    auto labels = texify_pars(pars);

    cv::Scalar green(0, 255, 0);
    for (int c = 0; c < 3; ++c) {
      auto img = img_channels[c].clone();
      imtools::to_color(img, 255, 0, cv::COLORMAP_BONE);
      img       = imtools::imresize(img, resize, true);
      auto text = labels[c];
      imtools::add_text(img, text, 1, 2, 0.5, 1, green);
      colorized_lab.emplace_back(img);
    }

    imtools::show_layout_imgs(colorized_lab, disp);
  }
}  // namespace debug
}  // namespace color

#endif  // SALIENCY_CHANNEL_COLOR_H
