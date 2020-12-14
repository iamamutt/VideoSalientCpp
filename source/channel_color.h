#ifndef SALIENCY_CHANNEL_COLOR_H
#define SALIENCY_CHANNEL_COLOR_H

#include "image_tools.h"

namespace color {

enum class ColorSpace
{
  DKL = 0,
  LAB = 1,
  RGB = 2
};

cv::Mat
lms_inv(const cv::Mat &row_mat = cv::Mat(0, 0, CV_32FC3))
{
  // row-wise allocation, matrix is for BGR ordered images
  cv::Mat lms = (cv::Mat_<float>(
    {0.09920825, 0.64933633, 0.25145542, -0.23151325, -0.55586618, 0.78737943, -0.90495899, 0.63933074, 0.26562825}));

  lms = lms.reshape(1, 3);

  if (row_mat.empty()) return lms;

  return lms * row_mat;
}

auto
bgr32FC3U_to_DKL(const cv::Mat &I32FC3U)
{
  // converts MxNx3 image to 3x(M*N)
  cv::Mat row_mat = I32FC3U.reshape(3, I32FC3U.total()).reshape(1).t();

  // transformation
  auto dkl = lms_inv(row_mat);

  // luminance, long (~red) + medium (~green) cone response
  cv::Mat l_plus_m = dkl.row(0);

  // red vs. green, difference of long and medium cones
  cv::Mat l_minus_m = dkl.row(1);

  // blue vs. yellow, short (~blue) cone response minus L+M
  cv::Mat l_plus_m_minus_s = dkl.row(2);

  // convert back to matrix
  l_plus_m         = l_plus_m.reshape(1, I32FC3U.rows);
  l_minus_m        = l_minus_m.reshape(1, I32FC3U.rows);
  l_plus_m_minus_s = l_plus_m_minus_s.reshape(1, I32FC3U.rows);

  // rescale chromatic competition channels
  l_minus_m += 1.f;
  l_minus_m /= 2.f;
  l_plus_m_minus_s += 1.f;
  l_plus_m_minus_s /= 2.f;

  return std::make_tuple(l_plus_m, l_minus_m, l_plus_m_minus_s);
}

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
  // returned scales: 0:1, 0:1, 0:1
  auto [lightness, red_v_grn, blu_v_ylw] = bgr32FC3U_to_DKL(I32FC3U);

  // fold absolute intensity for competing color channels, rescale in range [0,1]
  imtools::unit_flt_to_zero_center(red_v_grn);
  imtools::unit_flt_to_zero_center(blu_v_ylw);
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

struct Parameters
{
  std::string debug_window_name = "ColorChannel";
  float scale;
  float shift;
  float norm_value;
  float weight;
  bool toggled = true;
  ColorSpace cspace;

  explicit Parameters(const std::string &colorspace = "dkl", float rescale = 1, float filter = 0, float _weight = 1)
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
    } else if (to_lower(colorspace) == "dkl") {
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
  callback_brighten(int pos, void *user_data)
  {
    auto *pars       = (Parameters *)user_data;
    pars->scale      = static_cast<float>(pos);
    pars->norm_value = white_pt_norm(pars->cspace, pars->scale, pars->shift);
    cv::setTrackbarPos("scale", pars->debug_window_name, pos);
  }

  void
  callback_filter(int pos, void *user_data)
  {
    auto *pars       = (Parameters *)user_data;
    pars->shift      = static_cast<float>(pos) * .01f;
    pars->norm_value = white_pt_norm(pars->cspace, pars->scale, pars->shift);
    cv::setTrackbarPos("shift", pars->debug_window_name, pos);
  }

  void
  callback_colorspace(int pos, void *user_data)
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
    cv::namedWindow(pars->debug_window_name, cv::WINDOW_NORMAL);
    cv::createTrackbar("scale", pars->debug_window_name, &notches->scale, 50, &callback_brighten, pars);
    cv::setTrackbarMin("scale", pars->debug_window_name, 1);
    cv::createTrackbar("shift", pars->debug_window_name, &notches->shift, 100, &callback_filter, pars);
    cv::createTrackbar("cspace", pars->debug_window_name, &notches->cspace, 2, &callback_colorspace, pars);
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
