#ifndef SALIENCY_SALIENCY_H
#define SALIENCY_SALIENCY_H

#include "image_tools.h"
#include "parameters.h"
#include "tools.h"

namespace sal {

cv::Mat
summation_saturation(MatVec &filt_chan_imgs)
{
  // sum a single image processed by several filters
  auto dst = imtools::sum_and_scale(filt_chan_imgs);

  // remove negative activations
  imtools::neg_trunc(dst);

  // apply saturation and return in unit scale
  return imtools::tanh(dst);
}

MatVec
center_surround_activation_serial(const MatVec &channel_images, const MatVec &LoG_kernels)
{
  // each channel has sep images (e.g., luma, chromas), and each of those are filtered by kernels
  auto filtered_image_sets = imtools::cross_convolve(channel_images, LoG_kernels);
  MatVec combined_images;
  for (auto &imgs : filtered_image_sets) combined_images.emplace_back(summation_saturation(imgs));
  return combined_images;
}

cv::Mat
summation_saturation(VecFutureMats &fut_filt_chan_imgs)
{
  auto filtered_images = imtools::capture_image_futures(fut_filt_chan_imgs);
  return summation_saturation(filtered_images);
}

MatVec
summation_saturation(std::vector<VecFutureMats> &channel)
{
  MatVec channel_images;
  for (auto &futures : channel) channel_images.emplace_back(summation_saturation(futures));
  return channel_images;
}

FutureMatVec
center_surround_activation_async(const MatVec &channel_images, const MatVec &LoG_kernels)
{
  FutureMatVec fut = std::async(
    std::launch::async,
    [](const MatVec &_channel_images, const MatVec &_LoG_kernels) -> MatVec {
      auto futures = imtools::cross_convolve_async(_channel_images, _LoG_kernels);
      return summation_saturation(futures);
    },
    std::cref(channel_images), std::cref(LoG_kernels));
  return fut;
}

// convolve all feature images with mexican hat kernels
MatVec
center_surround_activation(const MatVec &channel_images, const MatVec &LoG_kernels, bool no_async)
{
  if (no_async) return center_surround_activation_serial(channel_images, LoG_kernels);
  return center_surround_activation_async(channel_images, LoG_kernels).get();
}

void
extract_color(const ImageSet &images, Parameters &pars, ChannelImages &sep_chan_imgs, FeatureMaps &maps, bool no_async)
{
  sep_chan_imgs.color = color::detect(images.curr.I32FC3U, pars.chan.color);
  if (sep_chan_imgs.color.empty()) return;
  auto color     = center_surround_activation(sep_chan_imgs.color, pars.model.kernels_LoG, no_async);
  maps.luminance = color[0].clone();
  maps.color     = imtools::sum_and_scale(slice(color, 1, 2));

  // adjust scale for missing feature maps
  if (pars.toggle_adj != 1.f) {
    maps.luminance *= pars.toggle_adj;
    maps.color *= pars.toggle_adj;
  }
}

void
extract_lines(const ImageSet &images, Parameters &pars, ChannelImages &sep_chan_imgs, FeatureMaps &maps, bool no_async)
{
  sep_chan_imgs.lines = lines::detect(images.curr.I32FC1U, pars.chan.lines);
  if (sep_chan_imgs.lines.empty()) return;
  auto lines = center_surround_activation(sep_chan_imgs.lines, pars.model.kernels_LoG, no_async);
  maps.lines = imtools::sum_images(lines);
  maps.lines /= (.5 * lines.size());  // half of the images overlap
  if (pars.toggle_adj != 1.f) maps.lines *= pars.toggle_adj;
}

void
extract_flick(const ImageSet &images, Parameters &pars, ChannelImages &sep_chan_imgs, FeatureMaps &maps, bool no_async)
{
  sep_chan_imgs.flicker = flick::detect(images.prev.I32FC1U, images.curr.I32FC1U, pars.chan.flicker);
  if (sep_chan_imgs.flicker.empty()) return;
  auto flicker = center_surround_activation(sep_chan_imgs.flicker, pars.model.kernels_LoG, no_async);
  maps.flicker = imtools::sum_images(flicker);
  if (pars.toggle_adj != 1.f) maps.flicker *= pars.toggle_adj;
}

void
extract_flow(const ImageSet &images, Parameters &pars, ChannelImages &sep_chan_imgs, FeatureMaps &maps, bool no_async)
{
  sep_chan_imgs.flow = flow::detect(images.prev.I8UC1, images.curr.I8UC1, pars.chan.flow);
  if (sep_chan_imgs.flow.empty()) return;
  auto flow = center_surround_activation(sep_chan_imgs.flow, pars.model.kernels_LoG, no_async);
  maps.flow = imtools::sum_images(flow);  // mostly non-overlapping, don't rescale
  if (pars.toggle_adj != 1.f) maps.flow *= pars.toggle_adj;
}

void
channel_extraction_activation(const ImageSet &images,
                              Parameters &pars,
                              ChannelImages &sep_chan_imgs,
                              FeatureMaps &feature_maps,
                              bool no_async = false)
{
  if (no_async) {
    extract_flow(images, pars, sep_chan_imgs, feature_maps, no_async);
    extract_lines(images, pars, sep_chan_imgs, feature_maps, no_async);
    extract_color(images, pars, sep_chan_imgs, feature_maps, no_async);
    extract_flick(images, pars, sep_chan_imgs, feature_maps, no_async);
    return;
  }

  // use source images to extract feature maps for each channel and do activation
  auto flow_future = std::async(std::launch::async, extract_flow, std::cref(images), std::ref(pars),
                                std::ref(sep_chan_imgs), std::ref(feature_maps), no_async);

  //  extract_flow(images, pars, sep_chan_imgs, maps);
  auto lines_future = std::async(std::launch::async, extract_lines, std::cref(images), std::ref(pars),
                                 std::ref(sep_chan_imgs), std::ref(feature_maps), no_async);

  auto color_future = std::async(std::launch::async, extract_color, std::cref(images), std::ref(pars),
                                 std::ref(sep_chan_imgs), std::ref(feature_maps), no_async);

  auto flick_future = std::async(std::launch::async, extract_flick, std::cref(images), std::ref(pars),
                                 std::ref(sep_chan_imgs), std::ref(feature_maps), no_async);

  flick_future.get();
  color_future.get();
  lines_future.get();
  flow_future.get();
}

MatVec
feature_maps_struct_to_vec(const FeatureMaps &feature_maps, const cv::Mat &fallback)
{
  MatVec maps_vec;
  maps_vec.emplace_back(feature_maps.luminance.empty() ? fallback : feature_maps.luminance);
  maps_vec.emplace_back(feature_maps.color.empty() ? fallback : feature_maps.color);
  maps_vec.emplace_back(feature_maps.lines.empty() ? fallback : feature_maps.lines);
  maps_vec.emplace_back(feature_maps.flicker.empty() ? fallback : feature_maps.flicker);
  maps_vec.emplace_back(feature_maps.flow.empty() ? fallback : feature_maps.flow);
  return maps_vec;
}

cv::Mat
extract_saliency(const FeatureMaps &feature_maps, const cv::Mat &fallback)
{
  // separate feature channel images into one vec
  MatVec maps_vec = feature_maps_struct_to_vec(feature_maps, fallback);

  // sum all feature maps together into a single image
  auto saliency_image = imtools::sum_images(maps_vec);

  // compress and saturate back to unit scale
  saliency_image = imtools::tanh(saliency_image);

  return saliency_image;
}

cv::Mat
detect(const Source &source, Parameters &pars, ChannelImages &sep_chan_imgs, FeatureMaps &feature_maps)
{
  // center-surround activation on all feature channel vectors with images
  channel_extraction_activation(source.img, pars, sep_chan_imgs, feature_maps, source.opts.no_async);

  // extract a saliency image from set of feature maps
  auto saliency_image = extract_saliency(feature_maps, pars.model.blank_image);

  // blur together salient parts
  imtools::convolve(saliency_image, pars.model.gauss_kern);

  // apply central bias mask
  saliency_image = saliency_image.mul(pars.model.central_mask);

  // attenuate weaker saliency areas
  cv::pow(saliency_image, pars.model.contrast_factor, saliency_image);

  return saliency_image;
}

cv::Point
find_salient_point(const cv::Mat &saliency_image, double threshold)
{
  cv::Mat img_copy = saliency_image.clone();
  imtools::clip(img_copy, threshold / 255.);
  cv::boxFilter(img_copy, img_copy, -1, cv::Size(15, 15));

  double salient_value;
  cv::Point salient_point;
  cv::minMaxLoc(img_copy, nullptr, &salient_value, nullptr, &salient_point);

  return salient_point;
}

std::vector<std::vector<cv::Point>>
find_salient_contours(const cv::Mat &I8UC1, double threshold, const cv::Mat &kern)
{
  cv::Mat binary_img = I8UC1.clone();
  cv::threshold(binary_img, binary_img, threshold, 255, cv::THRESH_BINARY);
  cv::dilate(binary_img, binary_img, kern, cv::Point(-1, -1), 3);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  return contours;
}

// TODO: save contours, point, point value to YAML file
cv::Mat
saliency_output_image(const cv::Mat &saliency_image, const ModelParameters &pars)
{
  cv::Mat gray_img;
  imtools::convert_32FC1U_to_8UC1(saliency_image, gray_img);

  // find lower level saliency cutoff value
  auto saliency_thresh = pars.saliency_thresh < 0 ? imtools::get_otsu_thresh_value(gray_img) * 2 : pars.saliency_thresh;

  // find salient point and contours using threshold
  auto salient_contours = find_salient_contours(gray_img, saliency_thresh, pars.dilation_kernel);
  auto salient_point    = find_salient_point(saliency_image, saliency_thresh);
  auto salient_value    = saliency_image.at<float>(salient_point) * 255.f;

  std::cout << std::setprecision(5) << "- salient: point=" << salient_point << ", threshold=" << saliency_thresh
            << ", value=" << salient_value << std::endl;

  // start making final saliency output image
  auto salient_colorized = imtools::colorize_32FC1U(saliency_image);
  if (salient_point.x == 0 && salient_point.y == 0) return salient_colorized;

  cv::Scalar white(255, 255, 255);  // salient point
  cv::Scalar magenta(255, 0, 255);  // salient contour
  cv::Scalar cyan(255, 255, 0);     // other salient regions above threshold

  // draw salient contours
  for (size_t i = 0; i < salient_contours.size(); i++) {
    auto contains_point = cv::pointPolygonTest(salient_contours[i], salient_point, false);
    if (contains_point >= 0) {
      cv::drawContours(salient_colorized, salient_contours, static_cast<int>(i), magenta, 2);
    } else {
      cv::drawContours(salient_colorized, salient_contours, static_cast<int>(i), cyan, 2);
    }
  }
  // draw salient point
  cv::circle(salient_colorized, salient_point, 5, white, -1);

  return salient_colorized;
}

// *******************************************************
// Interactively select parameters and display adjustments
// *******************************************************
namespace debug {

  void
  callback_log_max_win(int pos, void *user_data)
  {
    auto *pars         = (ModelParameters *)user_data;
    auto k             = std::max(pos, 1) * 7;
    auto n             = pars->n_LoG_kern == 0 ? -1 : pars->n_LoG_kern;
    pars->kernels_LoG  = imtools::LoG_kernels(k, n);
    pars->n_LoG_kern   = static_cast<int>(pars->kernels_LoG.size());
    pars->max_LoG_size = pars->n_LoG_kern == 0 ? 0 : pars->kernels_LoG[0].rows;

    cv::setTrackbarPos("N LoG kerns", pars->debug_window_name, pars->n_LoG_kern);
    cv::setTrackbarPos("Max LoG size", pars->debug_window_name, pars->max_LoG_size / 7);
  }

  void
  callback_log_n_kern(int pos, void *user_data)
  {
    auto *pars         = (ModelParameters *)user_data;
    auto k             = pars->max_LoG_size == 0 ? 105 : pars->max_LoG_size;
    pars->kernels_LoG  = imtools::LoG_kernels(k, pos);
    pars->n_LoG_kern   = static_cast<int>(pars->kernels_LoG.size());
    pars->max_LoG_size = pars->n_LoG_kern == 0 ? 0 : pars->kernels_LoG[0].rows;

    cv::setTrackbarPos("N LoG kerns", pars->debug_window_name, pars->n_LoG_kern);
    cv::setTrackbarPos("Max LoG size", pars->debug_window_name, pars->max_LoG_size / 7);
  }

  void
  callback_gauss_blur_win(int pos, void *user_data)
  {
    auto *pars           = (ModelParameters *)user_data;
    pars->gauss_kern     = imtools::kernel_gauss_2d(odd_int(pos));
    pars->gauss_blur_win = pars->gauss_kern.cols;

    cv::setTrackbarPos("Blur size", pars->debug_window_name, pars->gauss_blur_win);
  }

  void
  callback_contrast_factor(int pos, void *user_data)
  {
    auto *pars            = (ModelParameters *)user_data;
    pars->contrast_factor = static_cast<double>(pos);

    cv::setTrackbarPos("Contrast fac", pars->debug_window_name, pos);
  }

  void
  callback_central_focus_prop(int pos, void *user_data)
  {
    auto *pars               = (ModelParameters *)user_data;
    pars->central_focus_prop = static_cast<double>(pos) * .01;
    pars->central_mask       = imtools::get_border_mask(
      pars->central_mask.cols, pars->central_mask.rows, pars->central_focus_prop);
    cv::setTrackbarPos("Central foc", pars->debug_window_name, pos);
  }

  void
  callback_saliency_thresh(int pos, void *user_data)
  {
    auto *pars            = (ModelParameters *)user_data;
    pars->saliency_thresh = static_cast<double>(pos - 1);
    cv::setTrackbarPos("Saliency thresh", pars->debug_window_name, pos);
  }

  struct TrackbarPositions
  {
    int max_LoG_size;
    int n_LoG_kern;
    int gauss_blur_win;
    int contrast_factor;
    int central_focus_prop;
    int saliency_thresh;

    explicit TrackbarPositions(const ModelParameters &defaults = ModelParameters(105, 3, 3))
    {
      max_LoG_size       = static_cast<int>(defaults.max_LoG_size / 7);
      n_LoG_kern         = static_cast<int>(defaults.n_LoG_kern);
      gauss_blur_win     = static_cast<int>(defaults.gauss_blur_win);
      contrast_factor    = static_cast<int>(defaults.contrast_factor);
      central_focus_prop = static_cast<int>(100. * defaults.central_focus_prop);
      saliency_thresh    = static_cast<int>(defaults.saliency_thresh + 1);
    }
  };

  void
  create_trackbar(TrackbarPositions *notches, ModelParameters *pars)
  {
    if (!pars->toggle) return;
    cv::namedWindow(pars->debug_window_name);
    cv::createTrackbar(
      "Max LoG size", pars->debug_window_name, &notches->max_LoG_size, 100, &callback_log_max_win, pars);
    cv::createTrackbar("N LoG kerns", pars->debug_window_name, &notches->n_LoG_kern, 10, &callback_log_n_kern, pars);

    cv::createTrackbar(
      "Blur size", pars->debug_window_name, &notches->gauss_blur_win, 50, &callback_gauss_blur_win, pars);
    cv::setTrackbarMin("Blur size", pars->debug_window_name, 3);

    cv::createTrackbar(
      "Contrast fac", pars->debug_window_name, &notches->contrast_factor, 10, &callback_contrast_factor, pars);
    cv::setTrackbarMin("Contrast fac", pars->debug_window_name, 1);

    cv::createTrackbar(
      "Central foc", pars->debug_window_name, &notches->central_focus_prop, 100, &callback_central_focus_prop, pars);
    cv::setTrackbarMin("Central foc", pars->debug_window_name, 1);

    cv::createTrackbar(
      "Saliency thresh", pars->debug_window_name, &notches->saliency_thresh, 256, &callback_saliency_thresh, pars);
  }

  std::vector<Strings>
  texify_pars(const ModelParameters &pars)
  {
    Strings map_names = {"Luminance", "Color", "Lines", "Flicker", "Flow", "Params:"};
    std::vector<Strings> all_text;

    for (auto &name : map_names) {
      Strings image_txt = {name};
      if (name == "Params:") {
        std::stringstream max_LoG_size;
        std::stringstream n_LoG_kern;
        std::stringstream gauss_blur_win;
        std::stringstream contrast_factor;
        std::stringstream saliency_thresh;
        std::stringstream central_focus_prop;

        max_LoG_size << "max_LoG_size: " << pars.max_LoG_size;
        n_LoG_kern << "n_LoG_kern: " << pars.n_LoG_kern;
        gauss_blur_win << "gauss_blur_win: " << pars.gauss_blur_win;
        contrast_factor << "contrast_factor: " << pars.contrast_factor;
        saliency_thresh << "saliency_thresh: " << pars.saliency_thresh;
        central_focus_prop << "central_focus_prop: " << pars.central_focus_prop;

        image_txt.emplace_back(max_LoG_size.str());
        image_txt.emplace_back(n_LoG_kern.str());
        image_txt.emplace_back(gauss_blur_win.str());
        image_txt.emplace_back(contrast_factor.str());
        image_txt.emplace_back(saliency_thresh.str());
        image_txt.emplace_back(central_focus_prop.str());
      }
      all_text.emplace_back(image_txt);
    }

    return all_text;
  }

  void
  visualize(const FeatureMaps &feature_maps,
            const ModelParameters &pars,
            const cv::Size &resize,
            const DisplayData &disp)
  {
    if (!pars.toggle) return;
    MatVec maps = sal::feature_maps_struct_to_vec(feature_maps, pars.blank_image);
    maps.emplace_back(pars.blank_image);
    auto par_text = texify_pars(pars);
    MatVec colorized_maps;
    for (int i = 0; i < maps.size(); i++) {
      auto cmap = imtools::imresize(maps[i], resize);
      cmap      = imtools::colorize_32FC1U(cmap);
      imtools::add_text(cmap, par_text[i]);
      colorized_maps.emplace_back(cmap);
    }

    imtools::show_layout_imgs(colorized_maps, disp);
  }
}  // namespace debug
}  // namespace sal

#endif  // SALIENCY_SALIENCY_H
