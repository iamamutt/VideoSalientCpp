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
  maps.color     = imtools::sum_images(slice(color, 1, 2));

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

void
detect(const Source &source,
       Parameters &pars,
       ChannelImages &sep_chan_imgs,
       FeatureMaps &feature_maps,
       SaliencyMap &saliency_map)
{
  // center-surround activation on all feature channel vectors with images
  channel_extraction_activation(source.img, pars, sep_chan_imgs, feature_maps, source.opts.no_async);

  // extract a saliency image from set of feature maps
  auto sal_map = extract_saliency(feature_maps, pars.model.blank_image);

  // blur together salient parts
  imtools::convolve(sal_map, pars.model.gauss_kern);

  // apply central bias mask
  sal_map = sal_map.mul(pars.model.central_mask);

  // attenuate weaker saliency areas
  cv::pow(sal_map, pars.model.contrast_factor, sal_map);

  saliency_map.map = sal_map;
}

void
find_salient_contours(SaliencyMap &map_data, const cv::Mat &dilate_kernel)
{
  cv::Mat binary_img;
  cv::threshold(map_data.map_8bit, binary_img, map_data.threshold, 255, cv::THRESH_BINARY);
  cv::dilate(binary_img, binary_img, dilate_kernel, cv::Point(-1, -1), 2);
  map_data.contours.clear();
  cv::findContours(binary_img, map_data.contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
}

void
find_salient_points(SaliencyMap &map_data)
{
  cv::Mat peaks_smoothed;
  cv::boxFilter(map_data.map, peaks_smoothed, -1, cv::Size(15, 15));

  map_data.salient_coords.clear();
  map_data.salient_values.clear();

  double total_max = 0;
  int which_max    = -1;
  for (int i = 0; i < map_data.contours.size(); i++) {
    auto mask = imtools::make_black(map_data.map, CV_8UC1);
    cv::drawContours(mask, map_data.contours, i, cv::Scalar_<uchar>(1), -1);

    double max_val;
    cv::Point max_loc;
    cv::minMaxLoc(peaks_smoothed, nullptr, &max_val, nullptr, &max_loc, mask);

    if (max_val > total_max) {
      total_max = max_val;
      which_max = i;
    }
    map_data.salient_values.push_back(max_val);
    map_data.salient_coords.push_back(max_loc);
  }

  // move max saliency to front
  if (which_max > 0) {
    std::swap(map_data.contours[0], map_data.contours[which_max]);
    std::swap(map_data.salient_values[0], map_data.salient_values[which_max]);
    std::swap(map_data.salient_coords[0], map_data.salient_coords[which_max]);
  }
}

// TODO: save contours, point, point value to YAML file
void
analyze(SaliencyMap &map_data, const ModelParameters &pars)
{
  imtools::convert_32FC1U_to_8UC1(map_data.map, map_data.map_8bit);
  map_data.image = imtools::colorize_32FC1U(map_data.map);

  // find lower level saliency cutoff value
  map_data.threshold = pars.saliency_thresh <= 0 ?
                         imtools::get_otsu_thresh_value(map_data.map_8bit) * pars.saliency_thresh_mult :
                         pars.saliency_thresh;

  // find salient contours using threshold
  find_salient_contours(map_data, pars.dilation_kernel);
  if (map_data.contours.empty()) return;

  // find salient points from contours
  find_salient_points(map_data);

  // draw salient contours on final output image
  cv::drawContours(map_data.image, map_data.contours, 0, map_data.magenta, 2, cv::LINE_AA);
  for (int i = 1; i < map_data.contours.size(); i++) {
    cv::drawContours(map_data.image, map_data.contours, i, map_data.cyan, 2, cv::LINE_AA);
  }

  // draw salient point as circle
  cv::circle(map_data.image, map_data.salient_coords[0], 6, map_data.black, -1, cv::LINE_AA);
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

  void
  callback_saliency_thresh_mult(int pos, void *user_data)
  {
    auto *pars                 = (ModelParameters *)user_data;
    pars->saliency_thresh_mult = static_cast<double>(pos) / 10.;
    cv::setTrackbarPos("Saliency thresh mult", pars->debug_window_name, pos);
  }

  struct TrackbarPositions
  {
    int max_LoG_size;
    int n_LoG_kern;
    int gauss_blur_win;
    int contrast_factor;
    int central_focus_prop;
    int saliency_thresh;
    int saliency_thresh_mult;

    explicit TrackbarPositions(const ModelParameters &defaults = ModelParameters(105, 3, 3))
    {
      max_LoG_size         = static_cast<int>(defaults.max_LoG_size / 7);
      n_LoG_kern           = static_cast<int>(defaults.n_LoG_kern);
      gauss_blur_win       = static_cast<int>(defaults.gauss_blur_win);
      contrast_factor      = static_cast<int>(defaults.contrast_factor);
      central_focus_prop   = static_cast<int>(100. * defaults.central_focus_prop);
      saliency_thresh      = static_cast<int>(defaults.saliency_thresh + 1);
      saliency_thresh_mult = static_cast<int>(defaults.saliency_thresh_mult * 10);
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

    cv::createTrackbar("Saliency thresh mult", pars->debug_window_name, &notches->saliency_thresh_mult, 50,
                       &callback_saliency_thresh_mult, pars);
    cv::setTrackbarMin("Saliency thresh mult", pars->debug_window_name, 1);
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
        std::stringstream saliency_thresh_mult;
        std::stringstream central_focus_prop;

        max_LoG_size << "max_LoG_size: " << pars.max_LoG_size;
        n_LoG_kern << "n_LoG_kern: " << pars.n_LoG_kern;
        gauss_blur_win << "gauss_blur_win: " << pars.gauss_blur_win;
        contrast_factor << "contrast_factor: " << pars.contrast_factor;
        central_focus_prop << "central_focus_prop: " << pars.central_focus_prop;
        saliency_thresh << "saliency_thresh: " << pars.saliency_thresh;
        saliency_thresh_mult << "saliency_thresh_mult: " << pars.saliency_thresh_mult;

        image_txt.emplace_back(max_LoG_size.str());
        image_txt.emplace_back(n_LoG_kern.str());
        image_txt.emplace_back(gauss_blur_win.str());
        image_txt.emplace_back(contrast_factor.str());
        image_txt.emplace_back(central_focus_prop.str());
        image_txt.emplace_back(saliency_thresh.str());
        image_txt.emplace_back(saliency_thresh_mult.str());
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
