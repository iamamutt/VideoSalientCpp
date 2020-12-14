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
gabor_patches(const GaborParams &gabor_pars)
{
  return imtools::kernels_gabor(
    gabor_pars.theta, gabor_pars.size, gabor_pars.sigma, gabor_pars.lambda, gabor_pars.gamma, gabor_pars.psi);
}

struct Parameters
{
  bool toggled = true;
  float weight;
  GaborParams gabor_pars;
  MatVec kernels;
  std::string debug_window_name = "LinesChannel";

  explicit Parameters(int gabor_win_size = 11,
                      int n_rotations    = 8,
                      double sigma       = 1.625,
                      double lambda      = 6,
                      double psi         = 1.963495,
                      double gamma       = 0.375,
                      float _weight      = 1)
    : gabor_pars(n_rotations, gabor_win_size, sigma, lambda, psi, gamma), weight(_weight)
  {
    kernels = gabor_patches(gabor_pars);
  }
};

MatVec
detect(const cv::Mat &curr_32FC1_unit, const Parameters &pars)
{
  MatVec line_images;
  if (!pars.toggled) return line_images;
  auto line_imgs_futr = imtools::convolve_all_async(curr_32FC1_unit, pars.kernels);
  line_images         = imtools::capture_image_futures(line_imgs_futr);
  for (auto &&img : line_images) {
    imtools::gelu_approx<float>(img);
    imtools::tanh<float>(img);
  }
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
    pars->kernels         = lines::gabor_patches(pars->gabor_pars);
    cv::setTrackbarPos("Win Size", pars->debug_window_name, pos);
  }

  void
  callback_sigma(int pos, void *user_data)
  {
    auto *pars             = (lines::Parameters *)user_data;
    pars->gabor_pars.sigma = static_cast<double>(pos) * 0.125;
    pars->kernels          = lines::gabor_patches(pars->gabor_pars);
    cv::setTrackbarPos("Sigma", pars->debug_window_name, pos);
  }

  void
  callback_lambda(int pos, void *user_data)
  {
    auto *pars              = (lines::Parameters *)user_data;
    pars->gabor_pars.lambda = static_cast<double>(pos) * 0.125;
    pars->kernels           = lines::gabor_patches(pars->gabor_pars);
    cv::setTrackbarPos("Lambda", pars->debug_window_name, pos);
  }

  void
  callback_psi(int pos, void *user_data)
  {
    auto *pars           = (lines::Parameters *)user_data;
    pars->gabor_pars.psi = static_cast<double>(pos) * 0.125 * pi();
    pars->kernels        = lines::gabor_patches(pars->gabor_pars);
    cv::setTrackbarPos("Psi", pars->debug_window_name, pos);
  }

  void
  callback_gamma(int pos, void *user_data)
  {
    auto *pars             = (lines::Parameters *)user_data;
    pars->gabor_pars.gamma = static_cast<double>(pos) * 0.125;
    pars->kernels          = lines::gabor_patches(pars->gabor_pars);
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
    cv::namedWindow(pars->debug_window_name, cv::WINDOW_NORMAL);

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
    for (auto &kernel : kernels) {
      cv::Mat kern_copy = kernel.clone();
      imtools::unit_norm(kern_copy);
      imtools::to_color(kern_copy, 255, 0, cv::COLORMAP_BONE);
      out_images.emplace_back(kern_copy);
    }
    return out_images;
  }

  MatVec
  orientations_to_images(const MatVec &orientations)
  {
    MatVec out_images;
    for (auto &&line_img : orientations) {
      cv::Mat img = line_img.clone();
      double min  = imtools::global_min(img);
      img += min;
      imtools::to_color(img, 255, 0, cv::COLORMAP_CIVIDIS);
      out_images.emplace_back(img);
    }
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
    cv::Scalar green(0, 255, 0);
    for (int i = 0; i < filtered_images.size(); ++i) {
      auto image = imtools::blend_images_topleft(imtools::imresize(o_imgs[i], resize), k_imgs[i], 0);
      imtools::add_text(image, param_text[i], k_imgs[i].rows, 2, 0.5, 1, green);
      out_images.emplace_back(image);
    }

    imtools::show_layout_imgs(out_images, disp);
  }
}  // namespace debug
}  // namespace lines

#endif  // SALIENCY_CHANNEL_ORIENTATION_H
