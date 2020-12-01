/**
    opencv gui test
    @file highgui_test.cpp
    @author Joseph M. Burling
*/

#include "program.h"

static const std::string gui_window_name = "HighguiTestWindow";

static void
trackbar_callback_fn(int pos, void *user_data)
{
  auto *msg = (std::string *)user_data;
  std::stringstream new_msg;
  new_msg << "Trackbar position: " << pos;
  *msg = new_msg.str();
  std::cout << *msg << std::endl;
  cv::setTrackbarPos("Trackbar", gui_window_name, pos);
}

static void
mouse_callback_fn(int event, int x, int y, int flags, void *user_data)
{
  if (event == cv::EVENT_LBUTTONDOWN) {
    std::cout << "Mouse click exit" << std::endl;
    exit(0);
  }
}

int
main(int argc, char **argv)
{
  std::cout << "Testing OpenCV GUI" << std::endl;

  try {
    auto opts            = parse_command_line_args(argc, argv);
    auto [device, image] = cap::get_device(opts.device_switch, opts.device_values);

    if (image.empty()) {
      std::cerr << "Empty input device/image" << std::endl;
      exit(1);
    }

    auto cap_open                = device.isOpened();
    auto wait_time               = static_cast<int>(1000. / cap::get_cap_fps(device, 2));
    int notch_pos                = 4;
    int n_read                   = cap_open ? 20 : 0;  // only read next frames if video capture device open
    std::string callback_message = "null";

    // setup window and trackbar
    cv::namedWindow(gui_window_name);
    cv::moveWindow(gui_window_name, 20, 20);
    cv::createTrackbar("Trackbar", gui_window_name, &notch_pos, 8, &trackbar_callback_fn, &callback_message);
    imtools::add_text(image, "Saliency Test Image Text", 2, 2, 1);
    cv::imshow(gui_window_name, image);
    int key = cv::waitKey(wait_time * 10);
    cv::setMouseCallback(gui_window_name, mouse_callback_fn, 0);
    for (int n = 0; n < n_read; n++) {  // read next n frames
      if (key > 0 || !cap::read_next_frame(device, image)) break;
      cv::imshow(gui_window_name, image);
      key = cv::waitKey(wait_time);
    }

    if (key == 115) {  // 's' key pressed, save image
      auto temp_dir = std::filesystem::temp_directory_path();
      temp_dir /= "saliency_highgui_test_image.png";
      std::cout << "Saving file to: " << temp_dir << std::endl;
      cv::imwrite(temp_dir.string(), image);
    }

  } catch (std::exception &e) {
    std::cerr << e.what() << '\n';
    exit(1);
  }
}
