#ifndef SALIENCY_CHANNEL_FLICKER_H
#define SALIENCY_CHANNEL_FLICKER_H

#include "image_tools.h"

namespace flick {

struct Parameters
{
  float lower_lim;
  float upper_lim;
  float weight;
  bool toggle = true;
  cv::Mat morph_shape;
  std::string debug_window_name = "FlickerChannel";

  explicit Parameters(float lower_limit = 32, float upper_limit = 255, float _weight = 1)
    : lower_lim(lower_limit / 255.f), upper_lim(upper_limit / 255.f), weight(_weight)
  {
    morph_shape = imtools::get_morph_shape(4);
  };
};

MatVec
detect(const cv::Mat &prev_32FC1_unit, const cv::Mat &curr_32FC1_unit, const Parameters &pars)
{
  MatVec flicker;
  if (!pars.toggle) return flicker;
  cv::Mat flicker_image;
  cv::absdiff(curr_32FC1_unit, prev_32FC1_unit, flicker_image);
  imtools::clip(flicker_image, pars.lower_lim, pars.upper_lim);
  flicker_image *= (1.f / pars.upper_lim);
  cv::morphologyEx(flicker_image, flicker_image, cv::MORPH_DILATE, pars.morph_shape, cv::Point(-1, -1), 4);
  cv::morphologyEx(flicker_image, flicker_image, cv::MORPH_ERODE, pars.morph_shape, cv::Point(-1, -1), 4);
  flicker_image = imtools::tanh(flicker_image * 3.);
  flicker       = {flicker_image};
  if (pars.weight == 1) return flicker;
  flicker[0] *= pars.weight;
  return flicker;
}

// *******************************************************
// Interactively select parameters and display adjustments
// *******************************************************
namespace debug {
  void
  callback_min_thresh(int pos, void *user_data)
  {
    auto *pars      = (flick::Parameters *)user_data;
    pos             = std::min(pos, static_cast<int>(pars->upper_lim * 255) - 1);
    pars->lower_lim = static_cast<float>(pos) / 255.f;
    cv::setTrackbarPos("Min Thresh", pars->debug_window_name, pos);
  }

  void
  callback_max_thresh(int pos, void *user_data)
  {
    auto *pars      = (flick::Parameters *)user_data;
    pos             = std::max(pos, static_cast<int>(pars->lower_lim * 255) + 1);
    pars->upper_lim = static_cast<float>(pos) / 255.f;
    cv::setTrackbarPos("Max Thresh", pars->debug_window_name, pos);
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
    if (!pars->toggle) return;
    cv::namedWindow(pars->debug_window_name);
    cv::createTrackbar("Min Thresh", pars->debug_window_name, &notches->lower_lim, 255, &callback_min_thresh, pars);
    cv::createTrackbar("Max Thresh", pars->debug_window_name, &notches->upper_lim, 255, &callback_max_thresh, pars);
  }

  Strings
  texify_pars(const flick::Parameters &pars)
  {
    std::stringstream min_t;
    std::stringstream max_t;

    min_t << "minthresh: " << pars.lower_lim;
    max_t << "maxthresh: " << pars.upper_lim;

    Strings text_pars = {min_t.str(), max_t.str()};

    return text_pars;
  }

  void
  visualize(const MatVec &flick_img, const flick::Parameters &pars, const cv::Size &resize, const DisplayData &disp)
  {
    if (flick_img.empty() || !pars.toggle) return;
    MatVec colorized_flick = {imtools::colorize_32FC1U(flick_img[0])};
    colorized_flick[0]     = imtools::imresize(colorized_flick[0], resize);
    imtools::add_text(colorized_flick[0], texify_pars(pars));
    imtools::show_layout_imgs(colorized_flick, disp);
  }
}  // namespace debug
}  // namespace flick

#endif  // SALIENCY_CHANNEL_FLICKER_H
