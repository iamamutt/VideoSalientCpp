#ifndef SALIENCY_IMAGE_TOOLS_H
#define SALIENCY_IMAGE_TOOLS_H

#include "data_structures.h"
#include "tools.h"
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <utility>

namespace imtools {

uchar
mat_depth(const cv::Mat &mat)
{
  return mat.type() & CV_MAT_DEPTH_MASK;
}

uchar
n_channels(const cv::Mat &mat)
{
  return 1 + (mat.type() >> CV_CN_SHIFT);
}

std::string
mat_type(const cv::Mat &mat)
{
  std::string t, a;
  uchar depth    = mat_depth(mat);
  uchar channels = n_channels(mat);

  switch (depth) {
    case CV_8U:
      t = "8U";
      a = "Mat.at<uchar>(r,c)";
      break;
    case CV_8S:
      t = "8S";
      a = "Mat.at<schar>(r,c)";
      break;
    case CV_16U:
      t = "16U";
      a = "Mat.at<ushort>(r,c)";
      break;
    case CV_16S:
      t = "16S";
      a = "Mat.at<short>(r,c)";
      break;
    case CV_32S:
      t = "32S";
      a = "Mat.at<int>(r,c)";
      break;
    case CV_32F:
      t = "32F";
      a = "Mat.at<float>(r,c)";
      break;
    case CV_64F:
      t = "64F";
      a = "Mat.at<double>(r,c)";
      break;
    default:
      t = "User";
      a = "Mat.at<UKNOWN>(r,c)";
      break;
  };

  t += "C";
  t += (channels + '0');

  std::cout << "Mat is of type " << t << " and should be accessed with " << a << std::endl;

  return t;
}

cv::Mat
make_black(const cv::Mat &img, int type = CV_32FC1)
{
  return cv::Mat::zeros(img.rows, img.cols, type);
}

cv::Mat
make_black(const cv::Size &dims, int type = CV_32FC1)
{
  return cv::Mat::zeros(dims.height, dims.width, type);
}

// convert grayscale image to bgr
cv::Mat
mat_to_bgr(const cv::Mat &IC1, int type = CV_8UC3)
{
  cv::Mat mat_copy  = IC1.clone();
  cv::Mat layers[3] = {mat_copy, mat_copy, mat_copy};
  cv::Mat bgr;
  cv::merge(layers, 3, bgr);
  bgr.convertTo(bgr, type);
  return bgr;
}

std::tuple<cv::Mat, cv::Mat, cv::Mat>
split_bgr(const cv::Mat &bgr)
{
  std::vector<cv::Mat> vec(3);
  cv::split(bgr.clone(), vec);
  return std::make_tuple(vec[0], vec[1], vec[2]);
}

cv::Mat
merge_bgr(const cv::Mat &blue, const cv::Mat &green, const cv::Mat &red)
{
  cv::Mat layers[3] = {blue, green, red};
  cv::Mat bgr;
  cv::merge(layers, 3, bgr);
  return bgr;
}

void
flatten(const cv::Mat &bgr, cv::Mat &dst)
{
  cv::cvtColor(bgr, dst, cv::COLOR_BGR2GRAY);
}

cv::Mat
mat_to_row_arr(const cv::Mat &image)
{
  uint n_elem = image.total() * image.channels();
  cv::Mat arr = image.reshape(1, n_elem);
  if (!image.isContinuous()) arr = arr.clone();
  return arr;
}

void
neg_trunc(cv::Mat &image)
{
  cv::threshold(image, image, 0., -1, cv::THRESH_TOZERO);
}

void
clip(cv::Mat &image, double lower = NAN, double upper = NAN)
{
  auto lower_nan = isnan(lower);
  auto upper_nan = isnan(upper);
  if (lower_nan && upper_nan) return;

  if (!lower_nan) cv::threshold(image, image, lower, -1, cv::THRESH_TOZERO);
  if (!upper_nan) cv::threshold(image, image, upper, -1, cv::THRESH_TOZERO_INV);
}

void
clamp(cv::Mat &image, double lower = NAN, double upper = NAN)
{
  auto lower_nan = isnan(lower);
  auto upper_nan = isnan(upper);
  if (lower_nan && upper_nan) return;

  if (!lower_nan) {
    cv::Mat mask = image < lower;
    image.setTo(lower, mask);
  }

  if (!upper_nan) {
    cv::Mat mask = image > upper;
    image.setTo(upper, mask);
  }
}

void
unit_clamp(cv::Mat &image)
{
  clamp(image, 0., 1.);
}

void
unit_norm(cv::Mat &image, cv::InputArray &mask = cv::noArray())
{
  cv::normalize(image, image, 0., 1., cv::NORM_MINMAX, -1, mask);
}

void
l0_norm(cv::Mat &image, cv::InputArray &mask = cv::noArray())
{
  // image / max(abs(i) for each i)
  cv::normalize(image, image, 1., 0., cv::NORM_INF, -1, mask);
}

void
l1_norm(cv::Mat &image, cv::InputArray &mask = cv::noArray())
{
  // image / sum(abs(i) for each i)
  cv::normalize(image, image, 1., 0., cv::NORM_L1, -1, mask);
}

void
l2_norm(cv::Mat &image, cv::InputArray &mask = cv::noArray())
{
  // image / sum(i^2 for each i)
  cv::normalize(image, image, 1., 0., cv::NORM_L2, -1, mask);
}

void
std_norm(cv::Mat image)
{
  cv::Scalar mu, sigma;
  cv::meanStdDev(image, mu, sigma);
  image -= mu;
  image /= sigma;
}

void
unit_flt_to_zero_center(cv::Mat &float_img)
{
  cv::subtract(float_img, 0.5f, float_img);
  cv::multiply(float_img, 2.f, float_img);
}

cv::Scalar
sum_non_zero(cv::Mat &mat)
{
  cv::Mat mat_copy = mat.clone();
  cv::Mat mask     = mat_copy < 0.;
  mat_copy.setTo(0., mask);
  return cv::sum(mat_copy);
}

cv::Mat
sum_images(const MatVec &I32FC1U)
{
  // all images must be the same size
  if (I32FC1U.empty()) return cv::Mat(0, 0, CV_32FC1);
  cv::Mat dst = make_black(I32FC1U[0], CV_32FC1);
  for (auto &img : I32FC1U) dst += img;
  return dst;
}

cv::Mat
sum_and_scale(const MatVec &I32FC1U)
{
  if (I32FC1U.empty()) return cv::Mat(0, 0, CV_32FC1);
  auto dst   = sum_images(I32FC1U);
  auto scale = static_cast<float>(I32FC1U.size());
  dst /= scale;
  return dst;
}

cv::Mat
image_product(const MatVec &I32FC1U)
{
  assert(!I32FC1U.empty());
  cv::Mat dst = I32FC1U[0].clone();
  if (I32FC1U.size() == 1) return dst;
  for (int i = 1; i < I32FC1U.size(); ++i) cv::multiply(dst, I32FC1U[i], dst);
  return dst;
}

cv::Mat
imresize(const cv::Mat &src_image, const cv::Size &dims, bool shrink = true)
{
  auto interp = shrink ? cv::INTER_AREA : cv::INTER_CUBIC;
  cv::Mat img;
  cv::resize(src_image, img, dims, 0, 0, interp);
  return img;
}

cv::Mat
imresize(const cv::Mat &src_image, const ImageDims &dims)
{
  return imresize(src_image, dims.resize, true);
}

MatVec
abs_diff(const MatVec &imgs)
{
  assert(!imgs.empty());
  MatVec img_vec;
  if (imgs.size() == 1) {
    img_vec.emplace_back(imgs[0]);
  } else {
    for (int i = 1; i < imgs.size(); ++i) {
      cv::Mat mat;
      cv::absdiff(imgs[i - 1], imgs[i], mat);
      img_vec.emplace_back(mat);
    }
  }
  return img_vec;
}

cv::Mat
abs_diff_reduce(const MatVec &imgs)
{
  MatVec img_vec = abs_diff(imgs);
  auto n_iter    = img_vec.size();
  for (int i = 0; i < n_iter - 1; ++i) {
    img_vec = abs_diff(img_vec);
  }
  return img_vec[0];
}

cv::Mat
convert(const cv::Mat &src, cv::Mat &dst, int type = -1, double scale = 1, double shift = 0)
{
  src.convertTo(dst, type, scale, shift);
  return dst;
}

cv::Mat
convert(const cv::Mat &image, int type = -1, double scale = 1, double shift = 0)
{
  cv::Mat out_img;
  return convert(image, out_img, type, scale, shift);
}

void
convert_32FC1U_to_8UC1(const cv::Mat &I32FC1U, cv::Mat &dst)
{
  I32FC1U.convertTo(dst, CV_8UC1, 255);
}

void
convert_8UC3_to_32FC3U(const cv::Mat &I8UC3, cv::Mat &dst)
{
  I8UC3.convertTo(dst, CV_32FC3, 1. / 255.);
}

void
to_color(cv::Mat &mat, double scale = 1, double shift = 0, cv::ColormapTypes cmap = cv::COLORMAP_VIRIDIS)
{
  convert(mat, mat, CV_8UC1, scale, shift);
  cv::applyColorMap(mat, mat, cmap);
};

cv::Mat
colorize_grey(const cv::Mat &I8UC1, cv::Mat &dst)
{
  cv::applyColorMap(I8UC1, dst, cv::COLORMAP_VIRIDIS);
  return dst;
}

cv::Mat
colorize_32FC1U(const cv::Mat &I32FC1U)
{
  cv::Mat dst;
  convert(I32FC1U, dst, CV_8UC1, 255);
  cv::applyColorMap(dst, dst, cv::COLORMAP_VIRIDIS);
  return dst;
}

template<typename T>
void
tanh(cv::Mat &mat)
{
  for (int r = 0; r < mat.rows; ++r) {
    for (int c = 0; c < mat.cols; ++c) {
      mat.at<T>(r, c) = static_cast<T>(std::tanh(static_cast<double>(mat.at<T>(r, c))));
    }
  }
}

template<typename T>
void
gelu_approx(cv::Mat &mat)
{
  for (int r = 0; r < mat.rows; ++r) {
    for (int c = 0; c < mat.cols; ++c) {
      double val = static_cast<double>(mat.at<T>(r, c));
      double out = val * .5 * (1 + std::tanh(0.7978846 * (val + 0.044715 * std::pow(val, 3.))));

      mat.at<T>(r, c) = static_cast<T>(out);
    }
  }
}

void
truncated_logit(cv::Mat &img_unit, float min = 1e-6)
{
  img_unit.setTo(min, img_unit < min);
  img_unit.setTo(1.f - min, img_unit > (1.f - min));
  cv::log(img_unit / (1.f - img_unit), img_unit);
}

cv::Mat
logistic(const cv::Mat &img_32F, float l = 1.f, float k = 2.f, float m = 0.f)
{
  // l * exp(-logaddexp(0, -(x - m) * k))
  // l * exp(-log(exp(0) + exp(-(x - m) * k)))
  cv::Mat mat = img_32F.clone();
  mat -= m;
  mat *= (-1.f * k);
  cv::exp(mat, mat);
  mat += exp(0.f);
  cv::log(mat, mat);
  mat *= -1.f;
  cv::exp(mat, mat);
  mat *= l;
  return mat;
}

cv::Mat
sigmoid(const cv::Mat &img_32F)
{
  // input image should be floating type unit scale, no checking is done
  // exp(-log(exp(0) + exp(-x)))
  // logistic(x, 1, 1, 0);
  cv::Mat mat = img_32F.clone().mul(-1.f);
  cv::exp(mat, mat);
  mat += exp(0.f);
  cv::log(mat, mat);
  mat *= -1.f;
  cv::exp(mat, mat);
  return mat;
}

double
l2_dist(const cv::Point2f &p0, const cv::Point2f &p1, double scale = 1)
{
  return euclidean_dist(p0.x, p0.y, p1.x, p1.y, scale);
}

void
convolve(cv::Mat &I32FC, const cv::Mat &float_kernel)
{
  if (float_kernel.empty()) return;
  cv::filter2D(I32FC, I32FC, -1, float_kernel);
}

MatVec
convolve_all(const cv::Mat &I32FC, const MatVec &float_kernels)
{
  MatVec images;

  if (float_kernels.empty()) {
    images.emplace_back(I32FC.clone());
    return images;
  }

  for (auto &kern : float_kernels) {
    cv::Mat img = I32FC.clone();
    imtools::convolve(img, kern);
    images.emplace_back(img);
  }
  return images;
}

std::vector<MatVec>
cross_convolve(const MatVec &I32FC_vec, const MatVec &kernels)
{
  std::vector<MatVec> vec_of_matvecs;
  for (auto &&image : I32FC_vec) vec_of_matvecs.emplace_back(convolve_all(image, kernels));
  return vec_of_matvecs;
}

VecFutureMats
convolve_all_async(const cv::Mat &I32FC, const MatVec &float_kernels)
{
  VecFutureMats future_images;

  if (float_kernels.empty()) {
    // return image as is
    future_images.emplace_back(std::async(
      std::launch::async,
      [](const cv::Mat &flt_img) -> cv::Mat {
        cv::Mat convolved = flt_img.clone();
        return convolved;
      },
      std::ref(I32FC)));
    return future_images;
  }

  for (auto &kernel : float_kernels) {
    future_images.emplace_back(std::async(
      std::launch::async,
      [](const cv::Mat &flt_img, const cv::Mat &_kernel) -> cv::Mat {
        cv::Mat convolved = flt_img.clone();
        convolve(convolved, _kernel);
        return convolved;
      },
      std::ref(I32FC), std::ref(kernel)));
  }
  return future_images;
}

std::vector<VecFutureMats>
cross_convolve_async(const MatVec &I32FC_vec, const MatVec &kernels)
{
  std::vector<VecFutureMats> vec_of_fut_mat_vecs;
  for (auto &&image : I32FC_vec) vec_of_fut_mat_vecs.emplace_back(convolve_all_async(image, kernels));
  return vec_of_fut_mat_vecs;
}

cv::Mat
capture_image_futures(std::shared_future<cv::Mat> &future)
{
  return future.get();
}

MatVec
capture_image_futures(VecFutureMats &futures)
{
  MatVec images;
  for (auto &fimg : futures) {
    images.emplace_back(capture_image_futures(fimg));
  }
  return images;
}

std::vector<MatVec>
capture_image_futures(std::vector<VecFutureMats> &futures)
{
  std::vector<MatVec> vec_of_img_vecs;
  for (auto &vec : futures) vec_of_img_vecs.emplace_back(capture_image_futures(vec));
  return vec_of_img_vecs;
}

double
global_min(const cv::Mat &img, cv::InputArray &mask = cv::noArray())
{
  double min_val;
  cv::minMaxLoc(img, &min_val, nullptr, nullptr, nullptr, mask);
  return min_val;
}

double
global_max(const cv::Mat &img, cv::InputArray &mask = cv::noArray())
{
  double max_val;
  cv::minMaxLoc(img, nullptr, &max_val, nullptr, nullptr, mask);
  return max_val;
}

cv::Vec<double, 2>
global_range(const cv::Mat &img, cv::InputArray &mask = cv::noArray())
{
  cv::Vec<double, 2> range;
  cv::minMaxLoc(img, &range[0], &range[1], nullptr, nullptr, mask);
  return range;
}

double
global_median(const cv::Mat &image)
{
  auto row_mat = mat_to_row_arr(image);
  std::vector<double> vec_data;
  row_mat.copyTo(vec_data);
  std::nth_element(vec_data.begin(), vec_data.begin() + vec_data.size() / 2, vec_data.end());
  return vec_data[vec_data.size() / 2];
}

// find min absolute value from corner pixels of an image
template<typename T>
T
find_min_corner(const cv::Mat &mat)
{
  auto size = mat.size();

  std::vector<T> corner_values = {mat.at<T>(0, 0),                              // TL
                                  mat.at<T>(0, size.width - 1),                 // TR
                                  mat.at<T>(size.height - 1, 0),                // BL
                                  mat.at<T>(size.height - 1, size.width - 1)};  // BR

  // corner_values assumed to be near 0, find which is closest to center and shift
  T min_corner = corner_values[0];
  for (int i = 1; i < 4; ++i) {
    if (abs(corner_values[i]) < abs(min_corner)) min_corner = corner_values[i];
  }

  return min_corner;
}

cv::Mat
kernel_gauss_1d(int k, double sigma = -1)
{
  auto kernel = cv::getGaussianKernel(k, sigma, CV_32F);
  kernel /= cv::sum(kernel);
  return kernel;
}

cv::Mat
kernel_gauss_2d(int x, double sigma_x = -1, int y = -1, double sigma_y = -1)
{
  y               = y < 1 ? x : y;
  auto kernel_x   = kernel_gauss_1d(x, sigma_x);
  auto kernel_y   = kernel_gauss_1d(y, sigma_y);
  cv::Mat kern_2d = kernel_y * kernel_x.t();
  return kern_2d;
}

cv::Mat
kernel_mexican_hat(int k, double sigma = 1.4, double qr = 5)
{
  k              = std::max(k, 5);
  sigma          = std::max(sigma, .01);
  cv::Mat kern   = cv::Mat::zeros(k, k, CV_32FC1);
  auto quantiles = linspace(-1. * qr, qr, k);
  double dens;
  for (int x = 0; x < k; ++x) {
    for (int y = 0; y < k; ++y) {
      dens                 = laplace_gauss_pdf(quantiles[x], quantiles[y], sigma);
      kern.at<float>(y, x) = static_cast<float>(dens);
    }
  }
  return kern;
}

MatVec
kernels_gabor(const std::vector<double> &radians,
              const cv::Size &size,
              const double &sigma,
              const double &lambda,
              const double &gamma,
              const double &psi)
{
  MatVec kernels;
  if (radians.empty()) return kernels;

  for (auto &theta : radians) {
    cv::Mat kernel = cv::getGaborKernel(size, sigma, theta, lambda, gamma, psi, CV_32FC1);
    kernel -= find_min_corner<float>(kernel);
    kernels.push_back(kernel);
  };

  return kernels;
}

// make pyramid stack of laplacian of gaussian kernels
MatVec
kernels_lap_of_gauss(int max_size, int n = -1)
{
  MatVec kernels;
  if (max_size <= 0 || n == 0) return kernels;

  // minimum kernel size is 7x7
  int min_size  = 7;
  max_size      = std::max(max_size, min_size);
  n             = n == -1 ? max_size : n;
  int next_size = max_size;

  // get smaller kernels by a factor of 2 each iter
  for (int i = 0; i < n; ++i) {
    odd_int(next_size);
    if (next_size < min_size) break;
    kernels.emplace_back(kernel_mexican_hat(next_size));
    next_size /= 2;
  }

  // adjust kernel density
  for (auto &&kern : kernels) {
    auto adj = find_min_corner<float>(kern);
    kern -= adj;
    imtools::l2_norm(kern);
    kern *= (1. / sqrt(static_cast<float>(kern.rows)));
  }

  return kernels;
}

// get kernel for dilation/erosion
cv::Mat
kernel_morph(uint half_size, const cv::MorphShapes &shape = cv::MORPH_ELLIPSE)
{
  auto k = 2 * half_size + 1;
  return cv::getStructuringElement(shape, cv::Size(k, k), cv::Point(half_size, half_size));
}

cv::Mat
get_border_mask(int width, int height, double p = 1, float scale = 5)
{
  p = std::max(p, .01);

  auto sigma_x = sigma_prop_k(width, p);
  auto sigma_y = sigma_prop_k(height, p);
  auto mask    = kernel_gauss_2d(width, sigma_x, height, sigma_y);

  l0_norm(mask);
  mask *= scale;
  tanh<float>(mask);
  unit_norm(mask);

  return mask;
}

double
get_otsu_thresh_value(const cv::Mat &mat)
{
  cv::Size size = mat.size();
  if (mat.isContinuous()) {
    size.width *= size.height;
    size.height = 1;
  }
  const int N = 256;
  int i, j, h[N] = {0};
  for (i = 0; i < size.height; i++) {
    const uchar *src = mat.data + mat.step * i;
    for (j = 0; j <= size.width - 4; j += 4) {
      int v0 = src[j], v1 = src[j + 1];
      h[v0]++;
      h[v1]++;
      v0 = src[j + 2];
      v1 = src[j + 3];
      h[v0]++;
      h[v1]++;
    }
    for (; j < size.width; j++) h[src[j]]++;
  }

  double mu = 0, scale = 1. / (size.width * size.height);
  for (i = 0; i < N; i++) mu += i * h[i];

  mu *= scale;
  double mu1 = 0, q1 = 0;
  double max_sigma = 0, max_val = 0;

  for (i = 0; i < N; i++) {
    double p_i, q2, mu2, sigma;

    p_i = h[i] * scale;
    mu1 *= q1;
    q1 += p_i;
    q2 = 1. - q1;

    if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON) continue;

    mu1   = (mu1 + i * p_i) / q1;
    mu2   = (mu - q1 * mu1) / q2;
    sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
    if (sigma > max_sigma) {
      max_sigma = sigma;
      max_val   = i;
    }
  }

  return max_val;
}

ImageDims
get_image_dims(const cv::Mat &image, double base_size = 240., bool print = false)
{
  ImageDims dims;
  int width      = image.cols;
  int height     = image.rows;
  double ratio   = base_size / height;
  auto width_sm  = static_cast<int>(ratio * width);
  auto height_sm = static_cast<int>(ratio * height);
  dims.size      = cv::Size2i(width, height);
  dims.resize    = cv::Size2i(width_sm, height_sm);

  if (!print) return dims;

  std::cout << std::setprecision(3) << "Source image: " << dims.size.width << " x " << dims.size.height << " -> "
            << dims.resize.width << " x " << dims.resize.height << " | ratio = " << 1. * height_sm / height
            << std::endl;

  return dims;
}

template<typename T = float>
cv::Mat
get_test_img(int rows = 5, int cols = 5, double start = 0., double stop = 255.)
{
  int size       = rows * cols;
  auto data      = linspace<T>(start, stop, size);
  cv::Mat matrix = cv::Mat_<T>(rows, cols);
  int k          = 0;
  for (int y = 0; y < matrix.rows; y++) {
    for (int x = 0; x < matrix.cols; x++) {
      matrix.at<T>(y, x) = data[k];
      k++;
    }
  }
  // returns single channel image
  return matrix;
}

template<typename T>
void
print_elem_data(const cv::Mat &mat)
{
  if (mat.channels() != 1) {
    std::cout << "!!Warning: elem printing only for 1 channel mat" << std::endl;
    return;
  }

  mat_type(mat);
  for (int y = 0; y < mat.rows; y++) {
    for (int x = 0; x < mat.cols; x++) {
      auto elem = static_cast<double>(mat.at<T>(y, x));
      std::cout << elem << " | ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

void
print_image_value_info(const cv::Mat &image)
{
  cv::Scalar mu, sigma;
  cv::meanStdDev(image, mu, sigma);
  auto range = global_range(image);
  auto med   = global_median(image);
  std::cout << "size=" << image.size() << ", range=" << range << ",  median=" << med << ",  mean=" << mu
            << ", sd=" << sigma << std::endl;
}

void
debug_image_flt_unit(const cv::Mat &img, std::string win_name = "IMAGE_DEBUG", int ms_wait = 1)
{
  cv::Mat mat = img.clone();
  unit_norm(mat);
  mat = colorize_32FC1U(mat);
  cv::imshow(win_name, mat);
  if (ms_wait != -1) cv::waitKey(ms_wait);
}

cv::Mat
get_noise_image(int rows = 5, int cols = 5)
{
  cv::Size sz(cols, rows);
  auto img = make_black(sz, CV_32FC3);
  for (int y = 0; y < img.rows; y++) {
    for (int x = 0; x < img.cols; x++) {
      float b = cv::theRNG().uniform(0.f, 1.f);
      float g = cv::theRNG().uniform(0.f, 1.f);
      float r = cv::theRNG().uniform(0.f, 1.f);
      cv::Vec3f color(b, g, r);
      img.at<cv::Vec3f>(y, x) = color;
    }
  }
  img.convertTo(img, CV_8UC3, 255);
  return img;
}

DisplayData
setup_image_layout(const MatVec &mat_vec,
                   int cols,
                   int rows                = -1,
                   double scale            = 1,
                   std::string window_name = "Unnamed Window")
{
  DisplayData display;
  if (mat_vec.empty()) return display;

  display.scale   = scale;
  display.winname = std::move(window_name);

  // find number of images for placement
  auto n_images = mat_vec.size();
  auto n_show   = static_cast<double>(n_images);
  int n_cols;
  int m_rows;

  // calculate m rows and n cols based on num images
  if (cols == -1) {
    n_cols = static_cast<int>(ceil(sqrt(n_show)));
  } else {
    n_cols = cols;
  }

  if (rows == -1) {
    m_rows = static_cast<int>(ceil(n_show / static_cast<double>(n_cols)));
  } else {
    m_rows = rows;
  }

  if (n_cols * m_rows < n_show) {
    n_cols = static_cast<int>(ceil(sqrt(n_show)));
    m_rows = static_cast<int>(ceil(n_show / static_cast<double>(n_cols)));
  }

  // small matrix which holds scaled dims of all images, used later for max function
  cv::Mat tmp_heights = cv::Mat_<double>(m_rows, n_cols);
  cv::Mat tmp_widths  = cv::Mat_<double>(m_rows, n_cols);
  int img_index       = 0;
  for (int r = 0; r < m_rows; ++r) {
    for (int c = 0; c < n_cols; ++c) {
      if (img_index < n_images) {
        // place x,y size of current image in matrix
        tmp_heights.at<double>(r, c) = mat_vec[img_index].rows * display.scale;
        tmp_widths.at<double>(r, c)  = mat_vec[img_index].cols * display.scale;
      } else {
        tmp_heights.at<double>(r, c) = 0.;
        tmp_widths.at<double>(r, c)  = 0.;
      }
      ++img_index;
    }
  }

  // find mat size of each image to size width and height of display
  double img_height = 0;
  double img_width;

  // max height for each row in grid
  std::vector<double> max_row(static_cast<unsigned>(m_rows), 0);
  for (int m = 0; m < m_rows; ++m) {
    cv::minMaxIdx(tmp_heights.row(m), nullptr, &max_row[m]);
    img_height += max_row[m];
  }

  // sum of widths for each row in grid, find max
  std::vector<double> row_sums(static_cast<unsigned>(m_rows), 0);
  for (int m = 0; m < m_rows; ++m) {
    for (int n = 0; n < n_cols; ++n) {
      row_sums[m] += tmp_widths.at<double>(m, n);
    }
  }
  auto max_col_it = std::max_element(std::begin(row_sums), std::end(row_sums));
  img_width       = *max_col_it;
  display.size    = cv::Size(static_cast<int>(ceil(img_width + n_cols)), static_cast<int>(ceil(img_height + m_rows)));

  display.layout = cv::Size(n_cols, m_rows);

  return display;
}

cv::Mat
show_layout_imgs(const MatVec &mat_vec, const DisplayData &disp)
{
  cv::Mat out_img = cv::Mat::zeros(disp.size.height, disp.size.width, CV_8UC3);

  auto n_images     = static_cast<int>(mat_vec.size());
  int img_idx       = 0;
  int out_h_start   = 1;
  int out_w_start   = 1;
  int out_h_end     = 0;
  int out_w_end     = 0;
  bool no_more_imgs = false;

  for (auto m = 0; m < disp.layout.height; ++m) {
    out_w_start = 1;
    for (auto n = 0; n < disp.layout.width; ++n, ++img_idx) {
      if (img_idx < n_images) {
        cv::Mat in_img;
        if (disp.scale == 1) {
          in_img = mat_vec[img_idx];
        } else {
          cv::resize(mat_vec[img_idx], in_img, cv::Size(), disp.scale, disp.scale, cv::INTER_NEAREST);
        }

        cv::Size in_img_sz  = in_img.size();
        out_h_end           = out_h_start + (in_img_sz.height - 1);
        out_w_end           = out_w_start + (in_img_sz.width - 1);
        cv::Range in_roi_h  = cv::Range(0, in_img_sz.height - 1);
        cv::Range in_roi_w  = cv::Range(0, in_img_sz.width - 1);
        cv::Range out_roi_w = cv::Range(out_w_start, out_w_end);
        cv::Range out_roi_h = cv::Range(out_h_start, out_h_end);

        // paste scaled image onto larger image grid
        in_img(in_roi_h, in_roi_w).copyTo(out_img(out_roi_h, out_roi_w));
      } else {
        no_more_imgs = true;
        break;
      }
      out_w_start = out_w_end + 2;
    }
    if (no_more_imgs) break;
    out_h_start = out_h_end + 2;
  }

  cv::imshow(disp.winname, out_img);
  return out_img;
}

DisplayData
setup_window_layout(const cv::Size &size,
                    int rows,
                    int cols,
                    const std::string &window_name = "Unnamed Window",
                    bool show                      = false,
                    const cv::Point &pos           = cv::Point(-1, -1))
{
  auto img = imtools::make_black(size, CV_8UC3);
  MatVec imgs;
  for (int i = 0; i < abs(rows * cols); ++i) imgs.emplace_back(img);
  auto layout = imtools::setup_image_layout(imgs, cols, rows, 1, window_name);
  if (show) {
    imtools::show_layout_imgs(imgs, layout);
    cv::waitKey(1);
    if (pos.x == -1 && pos.y == -1) return layout;
    cv::moveWindow(window_name, pos.x, pos.y);
  }
  return layout;
}

DisplayData
setup_window_layout(int cols,
                    int rows                       = -1,
                    const cv::Rect &rect           = cv::Rect(-1, -1, 0, 0),
                    const std::string &window_name = "Unnamed Window",
                    bool show                      = true)
{
  return setup_window_layout(rect.size(), rows, cols, window_name, show, rect.tl());
}

cv::Mat
blend_images_topleft(const cv::Mat &main_img, const cv::Mat &sub_img, double alpha = 0.333)
{
  auto width            = sub_img.cols;
  auto height           = sub_img.rows;
  cv::Rect roi          = cv::Rect(0, 0, width, height);
  cv::Mat main_img_copy = main_img.clone();
  cv::addWeighted(main_img(roi), alpha, sub_img, 1 - alpha, 0.0, main_img_copy(roi));

  return main_img_copy;
}

void
add_text(cv::Mat &image,
         const Strings &text,
         int y_px                = 1,
         int x_px                = 2,
         double scale            = 0.5,
         int thickness           = 1,
         const cv::Scalar &color = cv::Scalar::all(255))
{
  if (text.empty()) return;

  int baseline;
  std::string txt = text[0];
  auto sz         = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
  baseline += thickness;
  y_px += sz.height;

  for (auto &line : text) {
    cv::Point txt_ps(x_px, y_px);
    cv::putText(image, line, txt_ps, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv::LINE_AA, false);
    y_px += sz.height + thickness * 2;
  }
}

void
add_text(cv::Mat &image,
         std::string text,
         int y_px                = 1,
         int x_px                = 2,
         double scale            = 0.5,
         int thickness           = 1,
         const cv::Scalar &color = cv::Scalar::all(255))
{
  Strings vec = {std::move(text)};
  add_text(image, vec, y_px, x_px, scale, thickness, color);
}

int
four_cc_str_to_int(std::string fourcc)
{
  if (fourcc.empty() || fourcc.size() != 4) {
    std::cerr << "FOUR_CC codec string was empty or incorrect: " << fourcc << std::endl;
    exit(1);
  }

  return cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
}

bool
win_opened(const int &key_pressed,
           const int &esc_key,
           const bool *button_pressed    = nullptr,
           bool check_button             = false,
           const std::string &check_name = "")
{
  bool non_stopping_key = key_pressed != esc_key;

  if (check_button) {
    return non_stopping_key && !*button_pressed;
  }

  if (!check_name.empty()) {
    auto visible = static_cast<int>(cv::getWindowProperty(check_name, cv::WND_PROP_VISIBLE));
    return non_stopping_key && visible != 0;
  }

  return non_stopping_key;
}
}  // namespace imtools

#endif  // SALIENCY_IMAGE_TOOLS_H
