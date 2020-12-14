#ifndef SALIENCY_CAPTURE_H
#define SALIENCY_CAPTURE_H

#include "command_line.h"
#include "image_tools.h"

namespace cap {

template<typename T>
auto
try_vid_cap_api(const T &input, const std::vector<cv::VideoCaptureAPIs> &apis)
{
  cv::VideoCapture cap_dev;
  bool cap_open;

  for (auto &api : apis) {
    cap_dev  = cv::VideoCapture(input, api);
    cap_open = cap_dev.isOpened();
    if (cap_open) break;
    std::cerr << "Video capture API " << api << " failed...\n";
  }

  return std::make_tuple(cap_open, cap_dev);
}

cv::VideoCapture
initialize_cam(const int &camera_index)
{
  std::vector<cv::VideoCaptureAPIs> apis = {cv::CAP_FFMPEG, cv::CAP_DSHOW, cv::CAP_OPENCV_MJPEG, cv::CAP_ANY};
  auto [cap_open, cam_cap]               = try_vid_cap_api<int>(camera_index, apis);

  if (!cap_open) {
    std::cerr << "\n!!Could not load camera: " << camera_index << "\n"
              << "- Use option -cam=0\n"
              << "- See program help (-h) for argument input usage\n"
              << std::endl;
    exit(1);
  }
  std::cout << "Input from camera device index: '" << camera_index << "'. Stream backend: " << cam_cap.getBackendName()
            << std::endl;
  return cam_cap;
}

cv::VideoCapture
initialize_vid(const std::string &video_file)
{
  std::vector<cv::VideoCaptureAPIs> apis = {
    cv::CAP_FFMPEG, cv::CAP_IMAGES, cv::CAP_DSHOW, cv::CAP_OPENCV_MJPEG, cv::CAP_ANY};
  auto [cap_open, video_cap] = try_vid_cap_api<std::string>(video_file, apis);

  if (!cap_open) {
    std::cerr << "\n!!Failed to load video file: " << video_file << "\n"
              << "- Use option -vid=\"path/to/video/file\"\n"
              << "- Make sure FFMPEG library is in current working directory\n"
              << "- See program help (-h) for argument input usage\n"
              << std::endl;
    exit(1);
  }

  std::cout << "Input from video file: " << video_file << ", backend: " << video_cap.get(cv::CAP_PROP_BACKEND)
            << std::endl;

  return video_cap;
}

cv::Mat
initialize_img(const std::string &image_file)
{
  cv::Mat source_image = cv::imread(image_file);
  if (source_image.empty()) {
    std::cerr << "\n!!Failed to load image file: " << image_file << "\n"
              << "- Use option -img=\"path/to/image/file\"\n"
              << "- See program help (-h) for argument input usage\n"
              << std::endl;
    exit(1);
  }
  std::cout << "Input from image file: " << image_file << std::endl;
  return source_image;
}

bool
read_next_frame(cv::VideoCapture &capture_device, cv::Mat &image)
{
  if (!capture_device.isOpened()) {
    std::cerr << "\n!!Capture device is not open" << std::endl;
    return false;
  }

  // possible end of video or unsuccessful grab
  if (!capture_device.grab()) return false;

  if (!capture_device.retrieve(image)) {
    std::cerr << "Next frame was not decoded for capture device!" << std::endl;
    return false;
  }

  return true;
}

std::tuple<cv::VideoCapture, cv::Mat>
get_device(DeviceSwitch &device_switch, DeviceValue &device_args)
{
  cv::VideoCapture capture_device;
  cv::Mat first_image;

  switch (device_switch) {
      // read image file
    case DeviceSwitch::IMAGE_FILE: {
      first_image = initialize_img(device_args.img);
      if (!first_image.size) exit(1);
      break;
    }
      // load video file
    case DeviceSwitch::VIDEO_FILE: {
      capture_device = initialize_vid(device_args.vid);
      if (!read_next_frame(capture_device, first_image)) exit(1);
      break;
    }
      // start camera device
    case DeviceSwitch::CAMERA_INDEX: {
      capture_device = initialize_cam(device_args.cam);
      if (!read_next_frame(capture_device, first_image)) exit(1);
      break;
    }

    default: exit(1);
  };

  return std::make_tuple(capture_device, first_image);
}

FPSCounter
initialize_fps_counter(size_t buffer_size = 5, int start_frame = 1, int print_fps = -1)
{
  FPSCounter counter;
  counter.fps_buff.resize(buffer_size);
  counter.fps_buff_size = buffer_size;
  counter.frame         = start_frame;
  counter.fps           = 0;
  counter.print_after   = print_fps;
  counter.clock         = timing::now();
  return counter;
}

void
update_fps(FPSCounter &counter, int next_frame = -1)
{
  next_frame    = next_frame > 0 ? next_frame : counter.frame + 1;
  auto frames   = next_frame - counter.frame;
  counter.frame = next_frame;

  auto [sec_dur, new_tp] = timing::elapsed<timing::Seconds>(counter.clock);
  counter.clock          = new_tp;

  auto index              = mod_index(next_frame, counter.fps_buff_size);
  counter.fps_buff[index] = static_cast<double>(frames) / sec_dur;

  if (next_frame % counter.fps_buff_size == 0) counter.fps = mean(counter.fps_buff);

  if (counter.print_after > 0 && counter.frame % counter.print_after == 0) {
    std::cout << "- frame: " << counter.frame << " @" << std::setprecision(4) << counter.fps << " FPS" << std::endl;
  }
}

double
get_cap_fps(cv::VideoCapture &dev, double min_fps)
{
  if (!dev.isOpened()) return min_fps;
  double fps = dev.get(cv::CAP_PROP_FPS);
  return std::max(fps, min_fps);
}

void
make_image_alts(Images &images)
{
  // make three channel float image unit scale
  imtools::convert_8UC3_to_32FC3U(images.I8UC3, images.I32FC3U);

  // make single channel gray image
  imtools::flatten(images.I8UC3, images.I8UC1);

  // make float single channel
  imtools::flatten(images.I32FC3U, images.I32FC1U);
}

ImageSet
initialize_source_images(const cv::Mat &initial_bgr_img)
{
  ImageSet img;

  // copy regular 8-bit image and also use to init previous
  img.curr.I8UC3 = initial_bgr_img.clone();
  img.prev.I8UC3 = initial_bgr_img.clone();

  // make alternative versions of prev and curr
  make_image_alts(img.prev);
  make_image_alts(img.curr);

  return img;
}

void
update_start_stop_frame_status(const FPSCounter &fps, const CmdLineOpts &opts, ProgramStatus &status)
{
  status.start_detection = fps.frame >= opts.start_frame;
  status.stop_detection  = 0 < opts.stop_frame && opts.stop_frame < fps.frame;
}

void
update_source(Source &src)
{
  // compute & display frames per second
  update_fps(src.fps);
  update_start_stop_frame_status(src.fps, src.opts, src.status);
  // no need to capture next if ending early
  if (src.status.stop_detection) return;

  // use existing images if device is a single input image, don't try to read next and make copies
  if (src.status.static_image) {
    src.status.frame_was_captured = true;
    return;
  }

  // assign current images to previous
  std::swap(src.img.prev.I8UC3, src.img.curr.I8UC3);
  std::swap(src.img.prev.I8UC1, src.img.curr.I8UC1);
  std::swap(src.img.prev.I32FC3U, src.img.curr.I32FC3U);
  std::swap(src.img.prev.I32FC1U, src.img.curr.I32FC1U);

  // get new images from capture device
  if (!read_next_frame(src.cap, src.img.curr.I8UC3)) {
    src.status.frame_was_captured = false;
    return;
  }

  // make alternate copies of current
  make_image_alts(src.img.curr);
}
}  // namespace cap

#endif  // SALIENCY_CAPTURE_H
