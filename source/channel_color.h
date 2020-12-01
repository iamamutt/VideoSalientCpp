#ifndef SALIENCY_CHANNEL_COLOR_H
#define SALIENCY_CHANNEL_COLOR_H

#include "image_tools.h"

namespace color {

enum class ColorSpace
{
  LAB = 0,
  RGB = 1
};

struct Parameters
{
  bool toggle                   = true;
  std::string debug_window_name = "ColorChannel";
  float scale;
  float shift;
  float weight;
  ColorSpace cspace;

  explicit Parameters(const std::string &colorspace = "lab", float rescale = 33, float filter = .17, float _weight = 1)
    : shift(filter), weight(_weight)
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
  };
};

MatVec
luma_chroma_compress(MatVec &images, const float &scale, const float &shift)
{
  // absolute intensity for competing colors
  images[1] = cv::abs(images[1]);
  images[2] = cv::abs(images[2]);

  // compress and shift values
  images[0] = imtools::logistic(images[0], 1, scale, shift * 2.f);
  images[1] = imtools::logistic(images[1], 1, scale, shift);
  images[2] = imtools::logistic(images[2], 1, scale, shift);

  return images;
}

MatVec
cspace_lab(const cv::Mat &I32FC3U, const color::Parameters &pars)
{
  cv::Mat lab_img;
  // CIELAB requires float [0,1] scale image
  cv::cvtColor(I32FC3U, lab_img, cv::COLOR_BGR2Lab);

  // scales: 0:100, -127:127, -127:127
  auto [lightness, red_v_grn, blu_v_ylw] = imtools::split_bgr(lab_img);

  // rescale on [-0.5, 0.5] scale
  lightness /= 100.f;
  lightness -= 0.5f;

  // rescale on [-1,1] scale
  red_v_grn /= 127.f;
  blu_v_ylw /= 127.f;

  MatVec lab_images = {lightness, red_v_grn, blu_v_ylw};

  return luma_chroma_compress(lab_images, pars.scale, pars.shift);
}

MatVec
cspace_rgb(const cv::Mat &I32FC3U, const color::Parameters &pars)
{
  auto [blue, green, red] = imtools::split_bgr(I32FC3U);
  cv::Mat yellow          = (red + green) / 2.f;
  cv::Mat red_v_grn       = red - green;
  cv::Mat blu_v_ylw       = blue - yellow;
  cv::Mat lightness       = (blue + red + green) / 3.f;
  lightness -= 0.5f;

  MatVec lrgby_images = {lightness, red_v_grn, blu_v_ylw};
  return luma_chroma_compress(lrgby_images, pars.scale, pars.shift);
}

MatVec
detect(const cv::Mat &I32FC3U, const color::Parameters &pars)
{
  MatVec cspace_imgs;
  if (!pars.toggle) return cspace_imgs;

  switch (pars.cspace) {
    case ColorSpace::LAB: cspace_imgs = cspace_lab(I32FC3U, pars); break;
    case ColorSpace::RGB: cspace_imgs = cspace_rgb(I32FC3U, pars); break;
  }

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
    auto *pars  = (Parameters *)user_data;
    pars->scale = static_cast<float>(pos);
    cv::setTrackbarPos("Brighten", pars->debug_window_name, pos);
  }

  void
  callback_filter(int pos, void *user_data)
  {
    auto *pars  = (Parameters *)user_data;
    pars->shift = static_cast<float>(pos) * .01f;
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
    if (!pars->toggle) return;
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
    if (img_channels.empty() || !pars.toggle) return;
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
