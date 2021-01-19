#ifndef SALIENCY_CHANNEL_FLICKER_H
#define SALIENCY_CHANNEL_FLICKER_H

#include "cv_tools.h"

namespace flick {

struct DefaultPars
{
  float lower_limit  = .20;
  float upper_limit  = 1;
  int morph_half_win = 5;
  float weight       = 1;

  DefaultPars() = default;

  explicit DefaultPars(int size)
  {
    if (size < 1 || size == BASE_IMAGE_LENGTH) return;

    // adjust defaults based on user image size
    auto adj = size / BASE_IMAGE_LENGTH;

    morph_half_win = static_cast<int>(round(morph_half_win * adj));
  }

  explicit DefaultPars(const cv::FileNode &node, int size = 0) : DefaultPars(size)
  {
    if (node.empty()) return;

    lower_limit    = yml_node_value(node["lower_limit"], lower_limit);
    upper_limit    = yml_node_value(node["upper_limit"], upper_limit);
    morph_half_win = yml_node_value(node["morph_half_win"], morph_half_win);
    weight         = yml_node_value(node["weight"], weight);
  }
};

struct Parameters
{
  std::string debug_window_name = "FlickerChannel";
  bool toggled                  = true;

  float weight;
  float lower_lim;
  float upper_lim;

  cv::Point dilate_ctr;
  cv::Mat morph_shape;

  explicit Parameters(const DefaultPars &defaults = DefaultPars())
    : lower_lim(defaults.lower_limit),
      upper_lim(defaults.upper_limit),
      weight(defaults.weight),
      dilate_ctr(-1, -1),
      morph_shape(imtools::kernel_morph(defaults.morph_half_win)){};
};

MatVec
detect(const cv::Mat &prev_32FC1_unit, const cv::Mat &curr_32FC1_unit, const Parameters &pars)
{
  MatVec flicker;
  if (!pars.toggled) return flicker;
  cv::Mat flicker_image;
  cv::absdiff(curr_32FC1_unit, prev_32FC1_unit, flicker_image);
  imtools::clip(flicker_image, pars.lower_lim, pars.upper_lim);
  flicker_image *= (1.f / pars.upper_lim);
  cv::morphologyEx(flicker_image, flicker_image, cv::MORPH_DILATE, pars.morph_shape, pars.dilate_ctr, 4);
  cv::morphologyEx(flicker_image, flicker_image, cv::MORPH_ERODE, pars.morph_shape, pars.dilate_ctr, 4);
  flicker_image *= 3.f;
  imtools::tanh<float>(flicker_image);
  flicker = {flicker_image};
  if (pars.weight == 1) return flicker;
  flicker[0] *= pars.weight;
  return flicker;
}

// *******************************************************
// Interactively select parameters and display adjustments
// *******************************************************
namespace debug {
  void
  callback_lower_lim(int pos, void *user_data)
  {
    auto *pars      = (flick::Parameters *)user_data;
    pos             = std::min(pos, static_cast<int>(pars->upper_lim * 255) - 1);
    pars->lower_lim = static_cast<float>(pos) / 255.f;
    cv::setTrackbarPos("lower_lim", pars->debug_window_name, pos);
  }

  void
  callback_upper_lim(int pos, void *user_data)
  {
    auto *pars      = (flick::Parameters *)user_data;
    pos             = std::max(pos, static_cast<int>(pars->lower_lim * 255) + 1);
    pars->upper_lim = static_cast<float>(pos) / 255.f;
    cv::setTrackbarPos("upper_lim", pars->debug_window_name, pos);
  }

  struct TrackbarPositions
  {
    int lower_lim;
    int upper_lim;

    explicit TrackbarPositions(const flick::Parameters &defaults = flick::Parameters())
    {
      lower_lim = static_cast<int>(defaults.lower_lim * 255.);
      upper_lim = static_cast<int>(defaults.upper_lim * 255.);
    }
  };

  void
  create_trackbar(flick::debug::TrackbarPositions *notches, flick::Parameters *pars)
  {
    if (!pars->toggled) return;
    cv::namedWindow(pars->debug_window_name, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("lower_lim", pars->debug_window_name, &notches->lower_lim, 255, &callback_lower_lim, pars);
    cv::setTrackbarMin("lower_lim", pars->debug_window_name, 1);
    cv::createTrackbar("upper_lim", pars->debug_window_name, &notches->upper_lim, 255, &callback_upper_lim, pars);
    cv::setTrackbarMin("upper_lim", pars->debug_window_name, 2);
  }

  Strings
  texify_pars(const flick::Parameters &pars)
  {
    std::stringstream min_t;
    std::stringstream max_t;

    min_t << "lower_lim: " << pars.lower_lim;
    max_t << "upper_lim: " << pars.upper_lim;

    Strings text_pars = {min_t.str(), max_t.str()};

    return text_pars;
  }

  void
  visualize(const MatVec &flick_img, const flick::Parameters &pars, const cv::Size &resize, const DisplayData &disp)
  {
    if (flick_img.empty() || !pars.toggled) return;
    MatVec colorized_flick = {imtools::colorize_32FC1U(flick_img[0])};
    colorized_flick[0]     = imtools::imresize(colorized_flick[0], resize);
    imtools::add_text(colorized_flick[0], texify_pars(pars));
    imtools::show_layout_imgs(colorized_flick, disp);
  }
}  // namespace debug
}  // namespace flick

#endif  // SALIENCY_CHANNEL_FLICKER_H
