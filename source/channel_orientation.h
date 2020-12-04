#ifndef SALIENCY_CHANNEL_ORIENTATION_H
#define SALIENCY_CHANNEL_ORIENTATION_H

#include "image_tools.h"

namespace lines {

// stores parameters to make gabor patches
struct GaborParams
{
  cv::Size size;
  double sigma;
  std::vector<double> theta;
  double lambda;
  double gamma;
  double psi;

  GaborParams(int n_rotations, int _size, double _sigma, double _lambda, double _psi, double _gamma)
    : sigma(_sigma), lambda(_lambda), psi(_psi), gamma(_gamma)
  {
    _size = std::max(_size, 5);
    size  = cv::Size(_size, _size);
    // n equally spaced rotations between 0 and 2pi
    theta = linspace<double>(0., 2. * pi(), n_rotations + 1);
    // remove redundant ending
    theta.pop_back();
  }
};

// make several rotated gabor patches
MatVec
gabor_kernels(const GaborParams &gabor_pars)
{
  MatVec kernels;
  for (auto &theta : gabor_pars.theta) {
    kernels.emplace_back(cv::getGaborKernel(
      gabor_pars.size, gabor_pars.sigma, theta, gabor_pars.lambda, gabor_pars.gamma, gabor_pars.psi, CV_32FC1));
  };
  return kernels;
}

struct Parameters
{
  bool toggled = true;
  float weight;
  GaborParams gabor_pars;
  MatVec kernels;
  std::string debug_window_name = "LinesChannel";

  explicit Parameters(int gabor_win_size = 15,
                      int n_rotations    = 8,
                      double sigma       = 3.5,
                      double lambda      = 10,
                      double psi         = 1.9635,
                      double gamma       = 0.625,
                      float _weight      = 1)
    : gabor_pars(n_rotations, gabor_win_size, sigma, lambda, psi, gamma), weight(_weight)
  {
    kernels = gabor_kernels(gabor_pars);
  }
};

MatVec
detect(const cv::Mat &curr_32FC1_unit, const Parameters &pars)
{
  MatVec line_images;
  if (!pars.toggled) return line_images;
  line_images = imtools::convolve_all(curr_32FC1_unit, pars.kernels);
  for (auto &&img : line_images) img = imtools::logistic(img, 1, 10, .2);
  if (pars.weight == 1) return line_images;
  for (auto &&img : line_images) img *= pars.weight;
  return line_images;
}

// *******************************************************
// Interactively select parameters and display adjustments
// *******************************************************
namespace debug {

  void
  callback_size(int pos, void *user_data)
  {
    auto *pars            = (lines::Parameters *)user_data;
    pars->gabor_pars.size = cv::Size(pos, pos);
    pars->kernels         = lines::gabor_kernels(pars->gabor_pars);
    cv::setTrackbarPos("Win Size", pars->debug_window_name, pos);
  }

  void
  callback_sigma(int pos, void *user_data)
  {
    auto *pars             = (lines::Parameters *)user_data;
    pars->gabor_pars.sigma = static_cast<double>(pos) * 0.125;
    pars->kernels          = lines::gabor_kernels(pars->gabor_pars);
    cv::setTrackbarPos("Sigma", pars->debug_window_name, pos);
  }

  void
  callback_lambda(int pos, void *user_data)
  {
    auto *pars              = (lines::Parameters *)user_data;
    pars->gabor_pars.lambda = static_cast<double>(pos) * 0.125;
    pars->kernels           = lines::gabor_kernels(pars->gabor_pars);
    cv::setTrackbarPos("Lambda", pars->debug_window_name, pos);
  }

  void
  callback_psi(int pos, void *user_data)
  {
    auto *pars           = (lines::Parameters *)user_data;
    pars->gabor_pars.psi = static_cast<double>(pos) * 0.125 * pi();
    pars->kernels        = lines::gabor_kernels(pars->gabor_pars);
    cv::setTrackbarPos("Psi", pars->debug_window_name, pos);
  }

  void
  callback_gamma(int pos, void *user_data)
  {
    auto *pars             = (lines::Parameters *)user_data;
    pars->gabor_pars.gamma = static_cast<double>(pos) * 0.125;
    pars->kernels          = lines::gabor_kernels(pars->gabor_pars);
    cv::setTrackbarPos("Gamma", pars->debug_window_name, pos);
  }

  struct TrackbarPositions
  {
    int size;
    int sigma;
    int lambda;
    int psi;
    int gamma;

    explicit TrackbarPositions(const lines::Parameters &defaults = lines::Parameters())
    {
      size   = defaults.gabor_pars.size.height;
      sigma  = static_cast<int>(defaults.gabor_pars.sigma * 8.);
      lambda = static_cast<int>(defaults.gabor_pars.lambda * 8.);
      psi    = static_cast<int>((defaults.gabor_pars.psi / pi()) * 8.);
      gamma  = static_cast<int>(defaults.gabor_pars.gamma * 8.);
    }
  };

  void
  create_trackbar(lines::debug::TrackbarPositions *notches, lines::Parameters *pars)
  {
    if (!pars->toggled) return;
    cv::namedWindow(pars->debug_window_name);

    cv::createTrackbar("Win Size", pars->debug_window_name, &notches->size, 50, &callback_size, pars);
    cv::setTrackbarMin("Win Size", pars->debug_window_name, 5);

    cv::createTrackbar("Sigma", pars->debug_window_name, &notches->sigma, 56, &callback_sigma, pars);
    cv::setTrackbarMin("Sigma", pars->debug_window_name, 1);

    cv::createTrackbar("Lambda", pars->debug_window_name, &notches->lambda, 120, &callback_lambda, pars);
    cv::setTrackbarMin("Lambda", pars->debug_window_name, 1);

    cv::createTrackbar("Psi", pars->debug_window_name, &notches->psi, 16, &callback_psi, pars);

    cv::createTrackbar("Gamma", pars->debug_window_name, &notches->gamma, 16, &callback_gamma, pars);
  }

  std::vector<Strings>
  texify_pars(const lines::GaborParams &gabor_pars)
  {
    std::vector<Strings> per_img;

    for (int n = 0; n < gabor_pars.theta.size(); ++n) {
      std::stringstream size;
      std::stringstream sigma;
      std::stringstream theta;
      std::stringstream lambda;
      std::stringstream psi;
      std::stringstream gamma;

      sigma.precision(5);
      theta.precision(5);
      lambda.precision(5);
      gamma.precision(5);
      psi.precision(5);

      size << "winsize: " << gabor_pars.size;
      sigma << "sigma: " << gabor_pars.sigma;
      theta << "theta: " << gabor_pars.theta[n];
      lambda << "lambda: " << gabor_pars.lambda;
      psi << "psi: " << gabor_pars.psi;
      gamma << "gamma: " << gabor_pars.gamma;

      Strings text_pars = {size.str(), sigma.str(), theta.str(), lambda.str(), psi.str(), gamma.str()};

      per_img.emplace_back(text_pars);
    };

    return per_img;
  }

  MatVec
  kernels_to_images(const MatVec &kernels)
  {
    MatVec out_images;
    for (auto &kern : kernels) {
      cv::Mat k_alt = kern.clone().mul(128.) + 127.;
      imtools::convert(k_alt, k_alt, CV_8UC1);
      imtools::colorize_grey(k_alt, k_alt);
      out_images.emplace_back(k_alt);
    }
    return out_images;
  }

  MatVec
  orientations_to_images(const MatVec &orientations)
  {
    MatVec out_images;
    for (auto &&i : orientations) out_images.emplace_back(imtools::colorize_32FC1U(i));
    return out_images;
  }

  void
  visualize(const MatVec &filtered_images,
            const lines::Parameters &pars,
            const cv::Size &resize,
            const DisplayData &disp)
  {
    if (filtered_images.empty() || !pars.toggled) return;
    auto o_imgs     = orientations_to_images(filtered_images);
    auto k_imgs     = kernels_to_images(pars.kernels);
    auto param_text = texify_pars(pars.gabor_pars);

    std::vector<cv::Mat> out_images;
    for (int i = 0; i < filtered_images.size(); ++i) {
      auto image = imtools::blend_images_topleft(imtools::imresize(o_imgs[i], resize), k_imgs[i], 0.1);
      imtools::add_text(image, param_text[i], k_imgs[i].rows);
      out_images.emplace_back(image);
    }

    imtools::show_layout_imgs(out_images, disp);
  }
}  // namespace debug
}  // namespace lines

#endif  // SALIENCY_CHANNEL_ORIENTATION_H
