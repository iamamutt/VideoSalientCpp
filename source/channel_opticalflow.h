#ifndef SALIENCY_CHANNEL_OPTICALFLOW_H
#define SALIENCY_CHANNEL_OPTICALFLOW_H

#include "image_tools.h"

namespace flow {

struct Point
{
  double value = 0;  // flow intensity
  double vel_x = 0;  // +/- velocity horz
  double vel_y = 0;  // +/- velocity vert
  double dist  = 0;  // euclidean distance of points between frames
  double error = 0;  // eigen errors
  cv::Point2f p0;    // frame n-1 point data
  cv::Point2f p1;    // frame n point data
};

// main data object for optical flow
struct Parameters
{
  bool toggled = true;
  float weight;                     // weight applied to all pixels in each output image
  int max_n_pts;                    // maximum number of allotted points
  double min_pt_dist;               // minimum distance between new points/features
  double max_dist;                  // max distance based on diagonal length of flow window
  int dilate_iter;                  // number of iterations for dilation
  int erode_iter;                   // number of erode iterations
  cv::Size lk_win_size;             // windows size for sparse flow
  cv::Mat dilate_shape;             // dilation shape of points
  cv::Point dilate_ctr;             // dilation shape center
  cv::TermCriteria term_crit;       // termination criteria
  std::vector<uchar> pt_status;     // binary status indicator for points
  std::vector<float> pt_error;      // eigen errors for points
  std::vector<int> pt_static;       // incrementer for non-moving points
  std::vector<cv::Point2f> p0_pts;  // point coordinates for previous frame
  std::vector<cv::Point2f> p1_pts;  // point coordinates for current frame
  std::vector<Point> points;        // final, processed point data
  std::string debug_window_name = "FlowChannel";

  explicit Parameters(int flow_window_size    = 73,
                      int max_num_points      = 150,
                      double min_point_dist   = 8,
                      unsigned morph_half_win = 3,
                      int morph_iters         = 8,
                      float _weight           = 1)
    : weight(_weight),
      max_n_pts(max_num_points),
      min_pt_dist(min_point_dist),
      dilate_iter(morph_iters),
      dilate_ctr(cv::Point(-1, -1)),
      term_crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 50, 0.03)
  {
    flow_window_size = std::max(flow_window_size, 5);
    lk_win_size      = cv::Size(flow_window_size, flow_window_size);
    dilate_iter      = std::max(dilate_iter, 1);
    min_pt_dist      = std::max(min_pt_dist, 3.);
    dilate_shape     = imtools::get_morph_shape(morph_half_win);
    erode_iter       = static_cast<int>(std::max(.5 * dilate_iter, 1.));
    max_dist         = euclidean_dist(0, 0, lk_win_size.width, lk_win_size.height, .5);
  }
};

// return index of flow image based on flow direction
uint
get_direction_index(const double &vel_x, const double &vel_y)
{
  uint index;
  if (abs(vel_y) > abs(vel_x)) {  // vertical more
    if (vel_y >= 0) {
      index = 2;  // down
    } else {
      index = 0;  // up
    }
  } else {  // horizontal more
    if (vel_x >= 0) {
      index = 3;  // right
    } else {
      index = 1;  // left
    }
  }
  return index;
}

// initializes and refines set of features (corners) to track using previous image
std::vector<cv::Point2f>
initialize_points(const cv::Mat &I8UC1, const Parameters &pars, int n_offset = 0)
{
  std::vector<cv::Point2f> corners;
  auto n_pts_needed = std::max(pars.max_n_pts - n_offset, 0);
  if (n_pts_needed < 1) return corners;
  static const cv::Size corner_sub_px_size = cv::Size(7, 7);
  cv::goodFeaturesToTrack(I8UC1, corners, n_pts_needed, 0.01, pars.min_pt_dist, cv::noArray(), 3, true);
  cv::cornerSubPix(I8UC1, corners, corner_sub_px_size, cv::Size(-1, -1), pars.term_crit);
  return corners;
}

// add more points if necessary before flow calculation
void
add_more_points(const cv::Mat &I8UC1, Parameters &pars)
{
  auto n_points = pars.points.size();
  if (n_points >= pars.max_n_pts) return;
  bool add_pt;

  for (auto &new_pt : initialize_points(I8UC1, pars)) {
    add_pt = true;
    // check distance to existing points
    for (auto &existing_pt : pars.points) {
      if (imtools::l2_dist(existing_pt.p0, new_pt) < pars.min_pt_dist) {
        // point too close, don't add
        add_pt = false;
        break;
      }
    }
    if (add_pt) {
      pars.p0_pts.push_back(new_pt);
      pars.pt_static.push_back(0);
      n_points++;
    }
    if (n_points >= pars.max_n_pts) break;
  }
}

// update estimated flow data after flow calculation
void
update_points(Parameters &pars)
{
  std::vector<cv::Point2f> p1_pts;
  std::vector<Point> points;
  std::vector<int> static_status;
  int static_count;
  double flow_dist;

  for (size_t i = 0; i < pars.pt_status.size(); i++) {
    // remove if the flow for features[i] has not been found
    if (pars.pt_status[i] == 0) continue;

    // remove points that haven't moved at least 2 pixels after 4 frames
    static_count = pars.pt_static[i];
    flow_dist    = imtools::l2_dist(pars.p0_pts[i], pars.p1_pts[i]);
    if (flow_dist < 2.) {
      // not moving, increment static counter
      static_count++;
    } else {
      // moving again, reset counter
      static_count = 0;
    };
    if (static_count > 4) continue;

    // process points to keep
    Point point;
    point.vel_x = velocity(pars.p0_pts[i].x, pars.p1_pts[i].x);
    point.vel_y = velocity(pars.p0_pts[i].y, pars.p1_pts[i].y);
    point.dist  = flow_dist;
    point.value = std::min(flow_dist / pars.max_dist, 1.);
    point.p0    = pars.p0_pts[i];
    point.p1    = pars.p1_pts[i];
    point.error = pars.pt_error[i];
    points.emplace_back(point);
    p1_pts.push_back(pars.p1_pts[i]);
    static_status.push_back(static_count);
  }

  // move next pts to prev points and clear other flow data
  pars.points.clear();
  pars.p0_pts.clear();
  pars.pt_static.clear();
  pars.p1_pts.clear();
  pars.pt_error.clear();
  pars.pt_status.clear();
  pars.points    = points;
  pars.p0_pts    = p1_pts;
  pars.pt_static = static_status;
}

void
draw_point(cv::Mat &ICV_32FC1, const Point &pt, int row_lim, int col_lim, const double &scale = 1)
{
  auto y                    = std::clamp<int>(round(pt.p1.y), 0, row_lim - 1);
  auto x                    = std::clamp<int>(round(pt.p1.x), 0, col_lim - 1);
  ICV_32FC1.at<float>(y, x) = static_cast<float>(pt.value * scale);
}

MatVec
make_fields(int height, int width, Parameters &pars)
{
  MatVec flow_fields;
  for (int i = 0; i < 4; ++i) {
    flow_fields.emplace_back(cv::Mat::zeros(height, width, CV_32FC1));
  }

  unsigned direction;
  for (auto &pt : pars.points) {
    if (pt.dist < 2.) continue;
    direction = get_direction_index(pt.vel_x, pt.vel_y);
    draw_point(flow_fields[direction], pt, height, width, 3.);
  }

  // dilate/erode point diameters for each field map
  for (auto &&field : flow_fields) {
    cv::morphologyEx(field, field, cv::MORPH_DILATE, pars.dilate_shape, pars.dilate_ctr, pars.dilate_iter);
    cv::boxFilter(field, field, -1, cv::Size(7, 7));
    cv::morphologyEx(field, field, cv::MORPH_ERODE, pars.dilate_shape, pars.dilate_ctr, pars.erode_iter);
  }

  // add static points but don't grow
  for (auto &pt : pars.points) {
    if (pt.dist >= 2.) continue;
    direction = get_direction_index(pt.vel_x, pt.vel_y);
    draw_point(flow_fields[direction], pt, height, width);
  }

  return flow_fields;
}

MatVec
detect(const cv::Mat &prev_8UC1, const cv::Mat &curr_8UC1, Parameters &pars)
{
  MatVec flows;
  if (!pars.toggled) return flows;
  add_more_points(prev_8UC1, pars);
  cv::calcOpticalFlowPyrLK(prev_8UC1, curr_8UC1, pars.p0_pts, pars.p1_pts, pars.pt_status, pars.pt_error,
                           pars.lk_win_size, 3, pars.term_crit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 1e-3);
  update_points(pars);
  flows = make_fields(prev_8UC1.rows, prev_8UC1.cols, pars);
  if (pars.weight == 1) return flows;
  for (auto &&img : flows) img *= pars.weight;
  return flows;
}

// *******************************************************
// Interactively select parameters and display adjustments
// *******************************************************
namespace debug {
  void
  callback_lk_win(int pos, void *user_data)
  {
    auto *flow        = (flow::Parameters *)user_data;
    flow->lk_win_size = cv::Size(pos * 2 + 1, pos * 2 + 1);
    cv::setTrackbarPos("Win Size", flow->debug_window_name, pos);
  }

  void
  callback_max_pt(int pos, void *user_data)
  {
    auto *pars      = (flow::Parameters *)user_data;
    pars->max_n_pts = pos * 10;
    cv::setTrackbarPos("Max Points", pars->debug_window_name, pos);
  }

  static void
  callback_min_pt_dist(int pos, void *user_data)
  {
    auto *pars        = (flow::Parameters *)user_data;
    pars->min_pt_dist = 5. * pos;
    cv::setTrackbarPos("Min Point Dist", pars->debug_window_name, pos);
    std::cout << "- Min Point Dist updated" << std::endl;
  }

  void
  callback_morph_win(int pos, void *user_data)
  {
    auto *pars         = (flow::Parameters *)user_data;
    pars->dilate_shape = imtools::get_morph_shape(pos);
    cv::setTrackbarPos("Morph Size", pars->debug_window_name, pos);
  }

  void
  callback_morph_iter(int pos, void *user_data)
  {
    auto *pars        = (flow::Parameters *)user_data;
    pars->dilate_iter = pos;
    pars->erode_iter  = static_cast<int>(std::max(.5 * pars->dilate_iter, 1.));
    cv::setTrackbarPos("Morph Iter", pars->debug_window_name, pos);
    std::cout << "- Morph Iter updated" << std::endl;
  }

  struct TrackbarPositions
  {
    int lk_win_size;
    int max_n_pts;
    int min_pt_dist;
    int dilate_shape_size;
    int dilate_iter;

    explicit TrackbarPositions(const flow::Parameters &defaults = flow::Parameters())
    {
      lk_win_size       = static_cast<int>((defaults.lk_win_size.height - 1) / 2.);
      max_n_pts         = static_cast<int>(defaults.max_n_pts / 10.);
      min_pt_dist       = static_cast<int>(defaults.min_pt_dist / 5.);
      dilate_shape_size = static_cast<int>((defaults.dilate_shape.rows - 1.) / 2.);
      dilate_iter       = static_cast<int>(defaults.dilate_iter);
    }
  };

  void
  create_trackbar(flow::debug::TrackbarPositions *notches, flow::Parameters *pars)
  {
    if (!pars->toggled) return;
    cv::namedWindow(pars->debug_window_name);

    cv::createTrackbar("Win Size", pars->debug_window_name, &notches->lk_win_size, 50, &callback_lk_win, pars);
    cv::setTrackbarMin("Win Size", pars->debug_window_name, 5);

    cv::createTrackbar("Max Points", pars->debug_window_name, &notches->max_n_pts, 50, &callback_max_pt, pars);
    cv::setTrackbarMin("Max Points", pars->debug_window_name, 1);

    cv::createTrackbar(
      "Min Point Dist", pars->debug_window_name, &notches->min_pt_dist, 50, &callback_min_pt_dist, pars);
    cv::setTrackbarMin("Min Point Dist", pars->debug_window_name, 1);

    cv::createTrackbar(
      "Morph Size", pars->debug_window_name, &notches->dilate_shape_size, 25, &callback_morph_win, pars);
    cv::setTrackbarMin("Morph Size", pars->debug_window_name, 1);

    cv::createTrackbar("Morph Iter", pars->debug_window_name, &notches->dilate_iter, 50, &callback_morph_iter, pars);
    cv::setTrackbarMin("Morph Iter", pars->debug_window_name, 1);
  }

  Strings
  texify_pars(const flow::Parameters &pars)
  {
    std::stringstream win_size;
    std::stringstream max_pts;
    std::stringstream pt_dist;
    std::stringstream morph_win;
    std::stringstream morph_iter;

    win_size << "winsize: " << pars.lk_win_size;
    max_pts << "maxpts: " << pars.max_n_pts;
    pt_dist << "minptdist: " << pars.min_pt_dist;
    morph_win << "morphwin: " << pars.dilate_shape.size;
    morph_iter << "morphiter: " << pars.dilate_iter;

    Strings text_pars = {win_size.str(), max_pts.str(), pt_dist.str(), morph_win.str(), morph_iter.str()};

    return text_pars;
  }

  cv::Scalar
  d_color(uint index)
  {
    const std::vector<cv::Scalar> colors = {
      cv::Scalar(0, 255, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255)};
    return colors[index];
  }

  void
  visualize(const MatVec &flow_fields, const flow::Parameters &pars, const cv::Size &resize, const DisplayData &disp)
  {
    if (flow_fields.empty() || !pars.toggled) return;

    // copy grayscale flow images to color images
    MatVec colorized_flow;
    for (auto &field : flow_fields) colorized_flow.emplace_back(imtools::colorize_32FC1U(field));

    // add flow lines to images
    unsigned direction;
    cv::Scalar color;
    for (auto &pt : pars.points) {
      direction = get_direction_index(pt.vel_x, pt.vel_y);
      cv::line(colorized_flow[direction], pt.p0, pt.p1, d_color(direction), 2);
    }

    // resize all images
    for (auto &&img : colorized_flow) img = imtools::imresize(img, resize);

    // add parameter text to images
    auto param_text = texify_pars(pars);
    for (auto &&flow_img : colorized_flow) imtools::add_text(flow_img, param_text);

    // combine all images into a single image and show
    imtools::show_layout_imgs(colorized_flow, disp);
  }
}  // namespace debug
}  // namespace flow

#endif  // SALIENCY_CHANNEL_OPTICALFLOW_H
