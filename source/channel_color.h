#ifndef SALIENCY_CHANNEL_COLOR_H
#define SALIENCY_CHANNEL_COLOR_H

#include "image_tools.h"

namespace color {

enum class ColorSpace
{
  LAB = 0,
  RGB = 1
};

MatVec
luma_chroma_compress(MatVec &images, const float &scale, const float &shift)
{
  // absolute intensity for competing colors
  images[1] = cv::abs(images[1]);
  images[2] = cv::abs(images[2]);

  // compress and shift values
  for (auto &&col : images) col = imtools::logistic(col, 1, scale, shift);

  return images;
}

MatVec
cspace_lab(const cv::Mat &I32FC3U, const float &scale, const float &shift)
{
  cv::Mat lab_img;
  // CIELAB requires float [0,1] scale image
  cv::cvtColor(I32FC3U, lab_img, cv::COLOR_BGR2Lab);

  // scales: 0:100, -127:127, -127:127
  auto [lightness, red_v_grn, blu_v_ylw] = imtools::split_bgr(lab_img);

  // rescale on [-0.5, 0.5] scale
  lightness /= 100.f;
  lightness -= 0.5f;
  lightness *= 2.f;

  // rescale on [-1,1] scale
  red_v_grn /= 127.f;
  blu_v_ylw /= 127.f;

  MatVec lab_images = {lightness, red_v_grn, blu_v_ylw};

  return luma_chroma_compress(lab_images, scale, shift);
}

MatVec
cspace_rgb(const cv::Mat &I32FC3U, const float &scale, const float &shift)
{
  auto [blue, green, red] = imtools::split_bgr(I32FC3U);
  cv::Mat yellow          = (red + green) / 2.f;
  cv::Mat red_v_grn       = red - green;
  cv::Mat blu_v_ylw       = blue - yellow;
  cv::Mat lightness       = (blue + red + green) / 3.f;
  lightness -= 0.5f;
  lightness *= 2.f;

  MatVec lrgby_images = {lightness, red_v_grn, blu_v_ylw};
  return luma_chroma_compress(lrgby_images, scale, shift);
}

float
get_norm_value(const ColorSpace &cspace, const float &scale, const float &shift)
{
  auto white_box = imtools::get_test_img(5, 5, 1, 1);
  white_box      = imtools::gray_to_bgr(white_box, CV_32FC3);
  MatVec cspace_imgs;
  switch (cspace) {
    case ColorSpace::LAB: cspace_imgs = cspace_lab(white_box, scale, shift); break;
    case ColorSpace::RGB: cspace_imgs = cspace_rgb(white_box, scale, shift); break;
  }
  double max_v;
  cv::minMaxLoc(cspace_imgs[0], nullptr, &max_v);
  max_v = 1. / max_v;
  return static_cast<float>(max_v);
}

struct Parameters
{
  std::string debug_window_name = "ColorChannel";
  float scale;
  float shift;
  float norm_value;
  float weight;
  bool toggled = true;
  ColorSpace cspace;

  explicit Parameters(const std::string &colorspace = "lab", float rescale = 6, float filter = .5, float _weight = 1)
    : shift(filter), weight(_weight), norm_value(1)
  {
    if (rescale < 1) {
      scale = shift * (1.f - (1.f / shift));
    } else {
      scale = rescale;
    }
    if (to_lower(colorspace) == "lab") {
      cspace = ColorSpace::LAB;
    } else if (to_lower(colorspace) == "rgb") {
      cspace = ColorSpace::RGB;
    } else {
      std::cerr << "Invalid colorspace value, "
                   "defaulting to \"LAB\""
                << std::endl;
      cspace = ColorSpace::LAB;
    }
    norm_value = get_norm_value(cspace, scale, shift);
  };
};

MatVec
detect(const cv::Mat &I32FC3U, const color::Parameters &pars)
{
  MatVec cspace_imgs;
  if (!pars.toggled) return cspace_imgs;

  switch (pars.cspace) {
    case ColorSpace::LAB: cspace_imgs = cspace_lab(I32FC3U, pars.scale, pars.shift); break;
    case ColorSpace::RGB: cspace_imgs = cspace_rgb(I32FC3U, pars.scale, pars.shift); break;
  }

  for (auto &&img : cspace_imgs) img *= pars.norm_value;
  if (pars.weight == 1) return cspace_imgs;
  for (auto &&img : cspace_imgs) img *= pars.weight;
  return cspace_imgs;
}

// *******************************************************
// Interactively select parameters and display adjustments
// *******************************************************
namespace debug {
  void
  callback_brighten(int pos, void *user_data)
  {
    auto *pars       = (Parameters *)user_data;
    pars->scale      = static_cast<float>(pos);
    pars->norm_value = get_norm_value(pars->cspace, pars->scale, pars->shift);
    cv::setTrackbarPos("Brighten", pars->debug_window_name, pos);
  }

  void
  callback_filter(int pos, void *user_data)
  {
    auto *pars       = (Parameters *)user_data;
    pars->shift      = static_cast<float>(pos) * .01f;
    pars->norm_value = get_norm_value(pars->cspace, pars->scale, pars->shift);
    cv::setTrackbarPos("Shift", pars->debug_window_name, pos);
  }

  void
  callback_colorspace(int pos, void *user_data)
  {
    auto *pars   = (Parameters *)user_data;
    pars->cspace = static_cast<color::ColorSpace>(pos);
    cv::setTrackbarPos("Colorspace", pars->debug_window_name, pos);
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
    cv::namedWindow(pars->debug_window_name);
    cv::createTrackbar("Brighten", pars->debug_window_name, &notches->scale, 50, &callback_brighten, pars);
    cv::setTrackbarMin("Brighten", pars->debug_window_name, 1);
    cv::createTrackbar("Shift", pars->debug_window_name, &notches->shift, 50, &callback_filter, pars);
    cv::createTrackbar("Colorspace", pars->debug_window_name, &notches->cspace, 1, &callback_colorspace, pars);
  }

  std::vector<Strings>
  texify_pars(const color::Parameters &pars)
  {
    Strings chan_type = {"Luminance", "Red vs. Green", "Blue vs. Yellow"};

    std::vector<Strings> each_image{3};
    for (int n; n < 3; ++n) {
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

    for (int c = 0; c < 3; ++c) {
      auto img  = imtools::colorize_32FC1U(img_channels[c]);
      img       = imtools::imresize(img, resize, true);
      auto text = labels[c];
      imtools::add_text(img, text);
      colorized_lab.emplace_back(img);
    }

    imtools::show_layout_imgs(colorized_lab, disp);
  }
}  // namespace debug
}  // namespace color

#endif  // SALIENCY_CHANNEL_COLOR_H
