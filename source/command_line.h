#ifndef SALIENCY_COMMAND_LINE_H
#define SALIENCY_COMMAND_LINE_H

#include <opencv2/opencv.hpp>

enum class DeviceSwitch
{
  IMAGE_FILE,    // image file input
  VIDEO_FILE,    // video file input
  CAMERA_INDEX,  // camera input device
  NONE           // no input
};

struct DeviceValue
{
  std::string img;
  std::string vid;
  int cam = -1;
};

struct CmdLineOpts
{
  bool debug           = false;
  bool split_output    = false;
  bool no_async        = false;
  bool no_gui          = false;
  bool right_click_esc = false;
  int start_frame      = 0;
  int stop_frame       = -1;
  int escape_key_code  = 27;
  std::string pars_file;
  std::string out_dir;
  std::string out_file;
  DeviceSwitch device_switch = DeviceSwitch::NONE;
  DeviceValue device_values;
};

cv::CommandLineParser
make_cmd_line_parser(int argc, const char *const *argv)
{
  const std::string about =
    "VideoSalientCpp\n"
    "A bottom-up visual saliency model";

  const std::string keys =
    "{h help ?    | | print this help message                                                                       }"
    "{img         | | full path to image file                                                                       }"
    "{vid         | | full path to video file                                                                       }"
    "{cam         | | usb camera index, use 0 for default                                                           }"
    "{dir         | | full path to where root saliency output directory will be created                             }"
    "{par         | | full path to the YAML parameters file                                                         }"
    "{split       | | output will be saved as a series of images instead of video                                   }"
    "{debug       | | toggle visualization of feature parameters. --dir output will be disabled                     }"
    "{no_gui      | | turn off displaying any output windows and using OpenCV GUI functionality. Will ignore --debug}"
    "{alt_exit    | | sets program exit to also allow right-clicking on the \"Saliency\" window                     }";

  cv::CommandLineParser parser(argc, argv, keys);
  parser.about(about);
  return parser;
}

void
show_help_then_exit(const CmdLineOpts &opts, const cv::CommandLineParser &parser)
{
  if (opts.device_switch != DeviceSwitch::NONE) return;
  // no input option / help
  parser.printMessage();
  exit(0);
}

auto
parse_command_line_args(int argc, const char *const *argv)
{
  auto parser = make_cmd_line_parser(argc, argv);
  if (!parser.check()) {
    parser.printErrors();
    exit(1);
  }

  CmdLineOpts opts;
  if (parser.has("dir")) opts.out_dir = parser.get<std::string>("dir");
  if (parser.has("par")) opts.pars_file = parser.get<std::string>("par");
  if (parser.has("split")) opts.split_output = parser.get<bool>("split");
  if (parser.has("debug")) opts.debug = parser.get<bool>("debug");
  if (parser.has("no_gui")) opts.no_gui = parser.get<bool>("no_gui");
  if (parser.has("alt_exit")) opts.right_click_esc = parser.get<bool>("alt_exit");

  if (opts.no_gui && opts.out_dir.empty()) {
    parser.printMessage();
    std::cerr << "!!Error: -no_gui is set but -dir is not specified. Check program options." << std::endl;
    exit(1);
  }

  if (opts.split_output && opts.out_dir.empty()) {
    parser.printMessage();
    std::cerr << "!!Warning: -split is set but -dir is not specified. Check program options." << std::endl;
  }

  // opts returned based on device priority
  if (parser.has("img")) {
    opts.device_values.img = parser.get<std::string>("img");
    opts.device_switch     = DeviceSwitch::IMAGE_FILE;
    return std::make_tuple(opts, parser);
  }

  if (parser.has("cam")) {
    opts.device_values.cam = parser.get<int>("cam");
    opts.device_switch     = DeviceSwitch::CAMERA_INDEX;
    return std::make_tuple(opts, parser);
  }

  if (parser.has("vid")) {
    opts.device_values.vid = parser.get<std::string>("vid");
    opts.device_switch     = DeviceSwitch::VIDEO_FILE;
    return std::make_tuple(opts, parser);
  }

  return std::make_tuple(opts, parser);
}

#endif  // SALIENCY_COMMAND_LINE_H
