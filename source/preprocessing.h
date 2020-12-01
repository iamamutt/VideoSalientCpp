#ifndef SALIENCY_PREPROCESSING_H
#define SALIENCY_PREPROCESSING_H

#include "capture.h"
#include "parameters.h"
#include "saliency.h"

struct AllTrackbarPositions
{
  color::debug::TrackbarPositions color;
  lines::debug::TrackbarPositions lines;
  flick::debug::TrackbarPositions flicker;
  flow::debug::TrackbarPositions flow;
  sal::debug::TrackbarPositions features;
};

namespace preprocess {

void
create_trackbars(bool debug, Parameters *pars, AllTrackbarPositions *debug_bar_positions)
{
  if (!debug) return;

  debug_bar_positions->color    = color::debug::TrackbarPositions(pars->chan.color);
  debug_bar_positions->lines    = lines::debug::TrackbarPositions(pars->chan.lines);
  debug_bar_positions->flicker  = flick::debug::TrackbarPositions(pars->chan.flicker);
  debug_bar_positions->flow     = flow::debug::TrackbarPositions(pars->chan.flow);
  debug_bar_positions->features = sal::debug::TrackbarPositions(pars->model);

  color::debug::create_trackbar(&debug_bar_positions->color, &pars->chan.color);
  lines::debug::create_trackbar(&debug_bar_positions->lines, &pars->chan.lines);
  flick::debug::create_trackbar(&debug_bar_positions->flicker, &pars->chan.flicker);
  flow::debug::create_trackbar(&debug_bar_positions->flow, &pars->chan.flow);
  sal::debug::create_trackbar(&debug_bar_positions->features, &pars->model);
}

void
make_output_folder(CmdLineOpts &opts)
{
  if (opts.debug || opts.out_dir.empty()) {
    opts.out_dir = "";
    return;
  };
  std::string stem = "/saliency_output_" + timing::date_time_string();
  opts.out_dir     = make_dir(opts.out_dir + stem);
}

cv::VideoWriter
setup_video_writer(const std::string &folder,
                   const std::string &stem,
                   const cv::Size &dims,
                   const std::string &codec = "",
                   double fps               = 0)
{
  int four_cc          = codec.empty() ? 0 : imtools::four_cc_str_to_int(codec);
  std::string file     = four_cc == 0 ? stem + "_%05d.jpg" : stem + ".avi";
  std::string fullpath = normalize_path(folder, file);

  cv::VideoWriter writer;
  writer.open(fullpath, four_cc, fps, dims);

  if (!writer.isOpened()) {
    std::cerr << "Failed to open file \"" + fullpath + "\" for writing. " + "Check codec settings." << std::endl;
    exit(1);
  }

  return writer;
}

void
setup_video_writer(Source &src)
{
  make_output_folder(src.opts);
  if (src.opts.out_dir.empty()) return;

  std::string codec;
  double fps = 0;
  if (!src.opts.split_output) {
    codec = "MJPG";
    fps   = cap::get_cap_fps(src.cap, 5);
  }

  src.vid = setup_video_writer(src.opts.out_dir, "saliency", src.dim.size, codec, fps);
}

static void
mouse_exit_callback_fn(int event, int x, int y, int flags, void *user_data)
{
  auto *status = (bool *)user_data;
  if (event == cv::EVENT_RBUTTONDOWN) {
    std::cout << "Right mouse button down. Exiting program." << std::endl;
    *status = true;
    return;
  }
}

void
setup_windows(const cv::Size &size, GridLayouts &layouts, Parameters &pars, bool debug, bool set_mouse_exit)
{
  cv::startWindowThread();

  // main display, always shown
  cv::Point start_pos(20, 20);
  cv::Rect main_rect = cv::Rect(start_pos, size);
  layouts.main       = imtools::setup_window_layout(2, 1, main_rect, "Saliency", true);
  main_rect          = cv::getWindowImageRect("Saliency");
  auto menu_bar_ht   = main_rect.y - start_pos.y;

  if (set_mouse_exit) {
    std::cout << "Escape by right-clicking Saliency window!" << std::endl;
    cv::setMouseCallback("Saliency", mouse_exit_callback_fn, &pars.rbutton_pressed);
  }

  // flicker display, right of main
  auto flick_rect = cv::Rect(cv::Point(main_rect.br().x + 1, start_pos.y), size);
  if (pars.chan.flicker.toggle) {
    layouts.flicker = imtools::setup_window_layout(1, 1, flick_rect, pars.chan.flicker.debug_window_name, debug);
    if (debug) flick_rect = cv::getWindowImageRect(pars.chan.flicker.debug_window_name);
  } else {
    flick_rect.width  = 0;
    flick_rect.height = 0;
  }
  std::cout << "Main image" << std::endl;
  // color display, below main and flicker
  auto color_rect = cv::Rect(cv::Point(main_rect.x, std::max(main_rect.br().y + 1, flick_rect.br().y + 1)), size);
  if (pars.chan.color.toggle) {
    layouts.color = imtools::setup_window_layout(3, 1, color_rect, pars.chan.color.debug_window_name, debug);
    if (debug) color_rect = cv::getWindowImageRect(pars.chan.color.debug_window_name);
  } else {
    color_rect.width  = 0;
    color_rect.height = 0;
  }

  // lines, right of flicker
  auto lines_rect = cv::Rect(cv::Point(std::max(color_rect.br().x + 1, flick_rect.br().x + 1), start_pos.y), size);
  cv::Rect(cv::Point(main_rect.width + main_rect.x + 1, main_rect.y), size);
  if (pars.chan.lines.toggle) {
    layouts.lines = imtools::setup_window_layout(4, 2, lines_rect, pars.chan.lines.debug_window_name, debug);
    if (debug) lines_rect = cv::getWindowImageRect(pars.chan.lines.debug_window_name);
  } else {
    lines_rect.width  = 0;
    lines_rect.height = 0;
  }

  // flow, below lines
  auto flow_rect = cv::Rect(cv::Point(lines_rect.x - 1, std::max(lines_rect.br().y + 1, start_pos.y)), size);
  if (pars.chan.flow.toggle) {
    layouts.flow = imtools::setup_window_layout(4, 1, flow_rect, pars.chan.flow.debug_window_name, debug);
    if (debug) flow_rect = cv::getWindowImageRect(pars.chan.flow.debug_window_name);
  } else {
    flow_rect.width  = 0;
    flow_rect.height = 0;
  }

  // feature maps bottom left
  auto maps_rect = cv::Rect(cv::Point(main_rect.x, std::max(main_rect.br().y + 1, color_rect.br().y + 1)), size);
  if (pars.model.toggle) {
    layouts.features = imtools::setup_window_layout(3, 2, maps_rect, pars.model.debug_window_name, debug);
    if (debug) maps_rect = cv::getWindowImageRect(pars.model.debug_window_name);
  } else {
    maps_rect.width  = 0;
    maps_rect.height = 0;
  }
}

auto
initialize(int argc, const char *const *argv)
{
  auto opts = parse_command_line_args(argc, argv);

  auto [capture_device, first_image] = cap::get_device(opts.device_switch, opts.device_values);

  Source src;
  src.cap  = capture_device;
  src.img  = cap::init_source_images(first_image);
  src.fps  = cap::get_fps_counter(5, 1, 30);
  src.dim  = imtools::get_image_dims(first_image, 240, true);
  src.opts = opts;

  Parameters pars = params::initialize_parameters(src.opts.pars_file, src.dim.size);

  setup_video_writer(src);

  return std::make_tuple(src, pars);
}

}  // namespace preprocess

#endif  // SALIENCY_PREPROCESSING_H
