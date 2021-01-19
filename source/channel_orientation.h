#ifndef SALIENCY_CHANNEL_ORIENTATION_H
#define SALIENCY_CHANNEL_ORIENTATION_H

#include "cv_tools.h"

namespace lines {

struct DefaultPars
{
  int kern_size   = 33;
  int n_rotations = 8;
  double sigma    = 3;
  double lambda   = 9;
  double psi      = 1.963495;
  double gamma    = 0.25;
  float weight    = 1;

  DefaultPars() = default;

  explicit DefaultPars(int size)
  {
    if (size < 1 || size == BASE_IMAGE_LENGTH) return;

    // adjust defaults based on user image size
    auto adj  = size / BASE_IMAGE_LENGTH;
    kern_size = static_cast<int>(round(kern_size * adj));
    sigma     = sigma * adj;
    lambda    = lambda * adj;
  }

  explicit DefaultPars(const cv::FileNode &node, int size = 0) : DefaultPars(size)
  {
    if (node.empty()) return;

    kern_size   = yml_node_value(node["kern_size"], kern_size, -1);
    n_rotations = yml_node_value(node["n_rotations"], n_rotations);
    sigma       = yml_node_value(node["sigma"], sigma);
    lambda      = yml_node_value(node["lambda"], lambda);
    psi         = yml_node_value(node["psi"], psi);
    gamma       = yml_node_value(node["gamma"], gamma);
    weight      = yml_node_value(node["weight"], weight);
  }
};

struct Parameters
{
  std::string debug_window_name = "LinesChannel";
  bool toggled                  = true;

  int kern_size;
  int n_rotations;
  double sigma;
  double lambda;
  double gamma;
  double psi;
  float weight;

  std::vector<double> theta;
  MatVec kernels;

  explicit Parameters(const DefaultPars &defaults = DefaultPars())
    : kern_size(defaults.kern_size),
      n_rotations(defaults.n_rotations),
      sigma(defaults.sigma),
      lambda(defaults.lambda),
      psi(defaults.psi),
      gamma(defaults.gamma),
      weight(defaults.weight)
  {
    if (n_rotations < 1) {
      weight = 0;
      return;
    }
    update_gabor_patches();
  }

  void
  update_gabor_patches()
  {
    // n equally spaced rotations between 0 and 2pi
    theta = linspace<double>(0., 2. * pi(), n_rotations + 1);

    // remove redundant ending
    theta.pop_back();

    // min size of kernel is 5
    auto k = odd_int(std::max(kern_size, 5));

    // make gabor kernels
    kernels = imtools::kernels_gabor(theta, cv::Size(k, k), sigma, lambda, gamma, psi);

    // update filter size info
    kern_size = kernels.empty() ? 0 : kernels[0].rows;
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
  callback_k_size(int pos, void *user_data)
  {
    auto *pars      = (lines::Parameters *)user_data;
    pars->kern_size = pos;
    pars->update_gabor_patches();
    cv::setTrackbarPos("k_size", pars->debug_window_name, pos);
  }

  void
  callback_sigma(int pos, void *user_data)
  {
    auto *pars  = (lines::Parameters *)user_data;
    pars->sigma = static_cast<double>(pos) * 0.125;
    pars->update_gabor_patches();
    cv::setTrackbarPos("sigma", pars->debug_window_name, pos);
  }

  void
  callback_lambda(int pos, void *user_data)
  {
    auto *pars   = (lines::Parameters *)user_data;
    pars->lambda = static_cast<double>(pos) * 0.125;
    pars->update_gabor_patches();
    cv::setTrackbarPos("lambda", pars->debug_window_name, pos);
  }

  void
  callback_psi(int pos, void *user_data)
  {
    auto *pars = (lines::Parameters *)user_data;
    pars->psi  = static_cast<double>(pos) * 0.125 * pi();
    pars->update_gabor_patches();
    cv::setTrackbarPos("psi", pars->debug_window_name, pos);
  }

  void
  callback_gamma(int pos, void *user_data)
  {
    auto *pars  = (lines::Parameters *)user_data;
    pars->gamma = static_cast<double>(pos) * 0.125;
    pars->update_gabor_patches();
    cv::setTrackbarPos("gamma", pars->debug_window_name, pos);
  }

  struct TrackbarPositions
  {
    int kern_size;
    int sigma;
    int lambda;
    int psi;
    int gamma;

    explicit TrackbarPositions(const lines::Parameters &defaults = lines::Parameters())
    {
      kern_size = defaults.kern_size;
      sigma     = static_cast<int>(defaults.sigma * 8.);
      lambda    = static_cast<int>(defaults.lambda * 8.);
      psi       = static_cast<int>((defaults.psi / pi()) * 8.);
      gamma     = static_cast<int>(defaults.gamma * 8.);
    }
  };

  void
  create_trackbar(lines::debug::TrackbarPositions *notches, lines::Parameters *pars)
  {
    if (!pars->toggled) return;
    cv::namedWindow(pars->debug_window_name, cv::WINDOW_AUTOSIZE);

    cv::createTrackbar("k_size", pars->debug_window_name, &notches->kern_size, 50, &callback_k_size, pars);
    cv::setTrackbarMin("k_size", pars->debug_window_name, 5);

    cv::createTrackbar("sigma", pars->debug_window_name, &notches->sigma, 56, &callback_sigma, pars);
    cv::setTrackbarMin("sigma", pars->debug_window_name, 1);

    cv::createTrackbar("lambda", pars->debug_window_name, &notches->lambda, 120, &callback_lambda, pars);
    cv::setTrackbarMin("lambda", pars->debug_window_name, 1);

    cv::createTrackbar("psi", pars->debug_window_name, &notches->psi, 16, &callback_psi, pars);

    cv::createTrackbar("gamma", pars->debug_window_name, &notches->gamma, 16, &callback_gamma, pars);
  }

  std::vector<Strings>
  texify_pars(const lines::Parameters &gabor_pars)
  {
    std::vector<Strings> per_img;

    for (int n = 0; n < gabor_pars.n_rotations; ++n) {
      std::stringstream k_size;
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

      k_size << "k_size: " << gabor_pars.kern_size;
      sigma << "sigma: " << gabor_pars.sigma;
      theta << "theta: " << gabor_pars.theta[n];
      lambda << "lambda: " << gabor_pars.lambda;
      psi << "psi: " << gabor_pars.psi;
      gamma << "gamma: " << gabor_pars.gamma;

      Strings text_pars = {k_size.str(), sigma.str(), theta.str(), lambda.str(), psi.str(), gamma.str()};
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
    auto param_text = texify_pars(pars);

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
