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
  saliency::debug::TrackbarPositions features;
};

namespace preprocess {

void
create_trackbars(const CmdLineOpts &opts, Parameters *pars, AllTrackbarPositions *debug_bar_positions)
{
  // trackbars only for --debug or when windows are open
  if (!opts.debug || opts.no_gui) return;

  debug_bar_positions->color    = color::debug::TrackbarPositions(pars->chan.color);
  debug_bar_positions->lines    = lines::debug::TrackbarPositions(pars->chan.lines);
  debug_bar_positions->flicker  = flick::debug::TrackbarPositions(pars->chan.flicker);
  debug_bar_positions->flow     = flow::debug::TrackbarPositions(pars->chan.flow);
  debug_bar_positions->features = saliency::debug::TrackbarPositions(pars->model);

  color::debug::create_trackbar(&debug_bar_positions->color, &pars->chan.color);
  lines::debug::create_trackbar(&debug_bar_positions->lines, &pars->chan.lines);
  flick::debug::create_trackbar(&debug_bar_positions->flicker, &pars->chan.flicker);
  flow::debug::create_trackbar(&debug_bar_positions->flow, &pars->chan.flow);
  saliency::debug::create_trackbar(&debug_bar_positions->features, &pars->model);
}

void
init_frame_start_stop_status(Source &src)
{
  if (src.status.static_image) {
    // override any start_frame settings if input is an image
    src.opts.start_frame = 0;
    if (src.opts.no_gui && src.opts.stop_frame == -1) {
      // automatically end after 15 frames, if stop_frame not set
      src.opts.stop_frame = 15;
    }
  }
  cap::update_start_stop_frame_status(src.fps, src.opts, src.status);
}

SaliencyMap
setup_saliency_data(const Source &src)
{
  SaliencyMap map_data;

  // saliency map as float array
  map_data.map = imtools::make_black(src.dim.size);

  // binary threshold image
  map_data.binary_img = imtools::make_black(src.dim.size, CV_8UC1);

  // final exported colorized frame
  map_data.image = imtools::make_black(src.dim.size, CV_8UC3);

  if (src.status.export_enabled) {
    auto csv_file = normalize_path(src.opts.out_dir, "saliency_data.csv");
    map_data.file = std::ofstream(csv_file);

    Strings header_cols = {"frame",        "img_width", "img_height", "saliency_thresh", "contour_num",
                           "contour_size", "pt_x",      "pt_y",       "salient_value"};
    write_csv_header(map_data.file, header_cols);
    map_data.file << std::fixed << std::setprecision(5);
  }

  return map_data;
}

cv::VideoWriter
setup_video_writer(CmdLineOpts &opts,
                   const std::string &prefix,
                   const cv::Size &dims,
                   const std::string &codec = "",
                   double fps               = 0)
{
  int four_cc      = codec.empty() ? 0 : imtools::four_cc_str_to_int(codec);
  std::string file = four_cc == 0 ? prefix + "_%05d.jpg" : prefix + ".avi";
  opts.out_file    = normalize_path(opts.out_dir, file);

  cv::VideoWriter writer;
  writer.open(opts.out_file, four_cc, fps, dims);

  if (!writer.isOpened()) {
    std::cerr << "Failed to open file \"" + opts.out_file + "\" for writing. " + "Check codec settings." << std::endl;
    exit(1);
  }

  return writer;
}

void
setup_video_writer(Source &src)
{
  if (!src.status.export_enabled) return;  // no saving output
  std::string stem = "/saliency_output_" + timing::date_time_string();
  src.opts.out_dir = make_dir(src.opts.out_dir + stem);

  std::string codec;
  double fps = 0;
  if (!src.opts.split_output) {
    codec = "MJPG";
    fps   = cap::get_cap_fps(src.cap, 5);
  }

  src.vid = setup_video_writer(src.opts, "saliency", src.dim.size, codec, fps);
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
setup_windows(Source &src, Parameters &pars)
{
  if (src.opts.no_gui) {
    std::cout << "Output windows suppressed." << std::endl;
    return;
  }

  cv::Point no_move(-1, -1);

  // main display, always shown
  cv::Point start_pos(20, 20);
  cv::Rect main_rect = cv::Rect(start_pos, src.dim.resize);
  src.layouts.main   = imtools::setup_window_layout(2, 1, main_rect, "Saliency");
  main_rect          = cv::getWindowImageRect(src.layouts.main.winname);
  auto menu_bar_ht   = main_rect.y - start_pos.y;

  if (src.opts.right_click_esc) {
    std::cout << "Escape by right-clicking Saliency window!" << std::endl;
    cv::setMouseCallback("Saliency", mouse_exit_callback_fn, &src.status.right_mouse_btn_down);
  }

  if (!src.opts.debug) return;

  // flicker display, right of main
  auto flick_pos  = src.opts.win_align ? cv::Point(main_rect.br().x + 1, start_pos.y) : no_move;
  auto flick_rect = cv::Rect(flick_pos, src.dim.resize);
  if (pars.chan.flicker.toggled) {
    src.layouts.flicker = imtools::setup_window_layout(1, 1, flick_rect, pars.chan.flicker.debug_window_name);
    flick_rect          = cv::getWindowImageRect(src.layouts.flicker.winname);
  } else {
    flick_rect.width  = 0;
    flick_rect.height = 0;
  }

  // color display, below main and flicker
  auto color_pos  = src.opts.win_align ? cv::Point(main_rect.x, std::max(main_rect.br().y + 1, flick_rect.br().y + 1)) :
                                         no_move;
  auto color_rect = cv::Rect(color_pos, src.dim.resize);
  if (pars.chan.color.toggled) {
    src.layouts.color = imtools::setup_window_layout(3, 1, color_rect, pars.chan.color.debug_window_name);
    color_rect        = cv::getWindowImageRect(src.layouts.color.winname);
  } else {
    color_rect.width  = 0;
    color_rect.height = 0;
  }

  // lines, right of flicker
  auto lines_pos = src.opts.win_align ? cv::Point(std::max(color_rect.br().x + 1, flick_rect.br().x + 1), start_pos.y) :
                                        no_move;
  auto lines_rect = cv::Rect(lines_pos, src.dim.resize);
  cv::Rect(cv::Point(main_rect.width + main_rect.x + 1, main_rect.y), src.dim.resize);
  if (pars.chan.lines.toggled) {
    auto n_show       = static_cast<double>(pars.chan.lines.kernels.size());
    auto n_l_row      = static_cast<int>(std::ceil(n_show / 4.));
    src.layouts.lines = imtools::setup_window_layout(4, n_l_row, lines_rect, pars.chan.lines.debug_window_name);
    lines_rect        = cv::getWindowImageRect(src.layouts.lines.winname);
  } else {
    lines_rect.width  = 0;
    lines_rect.height = 0;
  }

  // flow, below lines
  auto flow_pos  = src.opts.win_align ? cv::Point(lines_rect.x - 1, std::max(lines_rect.br().y + 1, start_pos.y)) :
                                        no_move;
  auto flow_rect = cv::Rect(flow_pos, src.dim.resize);
  if (pars.chan.flow.toggled) {
    src.layouts.flow = imtools::setup_window_layout(4, 1, flow_rect, pars.chan.flow.debug_window_name);
    flow_rect        = cv::getWindowImageRect(src.layouts.flow.winname);
  } else {
    flow_rect.width  = 0;
    flow_rect.height = 0;
  }

  // feature maps bottom left
  auto maps_pos  = src.opts.win_align ? cv::Point(main_rect.x, std::max(main_rect.br().y + 1, color_rect.br().y + 1)) :
                                        no_move;
  auto maps_rect = cv::Rect(maps_pos, src.dim.resize);
  if (pars.model.toggle) {
    src.layouts.features = imtools::setup_window_layout(3, 2, maps_rect, pars.model.debug_window_name);
    maps_rect            = cv::getWindowImageRect(src.layouts.features.winname);
  } else {
    maps_rect.width  = 0;
    maps_rect.height = 0;
  }

  //  cv::setWindowProperty(src.layouts.main.winname, cv::WND_PROP_TOPMOST, 1);
}

auto
initialize(int argc, const char *const *argv)
{
  auto [opts, parser] = parse_command_line_args(argc, argv);
  check_for_help(opts, parser);
  opts.bin_path = normalize_path(opts.bin_path);
  std::cout << "Running saliency at: " << opts.bin_path << std::endl;

  auto [capture_device, first_image] = cap::get_device(opts.device_switch, opts.device_values);

  Source src;
  src.cap  = capture_device;
  src.img  = cap::initialize_source_images(first_image);
  src.fps  = cap::initialize_fps_counter(5, 1, 30);
  src.dim  = imtools::get_image_dims(first_image, 240, true);
  src.opts = opts;

  // --debug option is on or --dir not specified in command line options
  src.status.export_enabled = !opts.debug && !opts.out_dir.empty();
  src.status.static_image   = src.opts.device_switch == DeviceSwitch::IMAGE_FILE;
  setup_video_writer(src);
  init_frame_start_stop_status(src);

  Parameters pars = params::initialize_parameters(src.opts.pars_file, src.dim.size, src.status.static_image);

  return std::make_tuple(src, pars);
}

}  // namespace preprocess

#endif  // SALIENCY_PREPROCESSING_H
