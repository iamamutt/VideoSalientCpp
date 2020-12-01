#ifndef SALIENCY_PROGRAM_H
#define SALIENCY_PROGRAM_H

#include "postprocessing.h"
#include "preprocessing.h"

void
run_program(int argc, const char *const *argv)
{
  // initialize capture device and read/load parameters
  auto [src, pars] = preprocess::initialize(argc, argv);
  //  params::parameter_defaults();

  // for visualization of model parameter adjustments
  static AllTrackbarPositions debug_bar_notch_values;
  preprocess::create_trackbars(src.opts.debug, &pars, &debug_bar_notch_values);

  // setup windows after trackbars to get window sizes with bars
  preprocess::setup_windows(src.dim.resize, src.layouts, pars, src.opts.debug, src.opts.right_click_esc);

  // stored feature maps
  ChannelImages sep_channel_images;
  FeatureMaps feature_maps;
  cv::Mat saliency_map;

  // start run through frames
  bool frame_captured = true;
  bool window_open    = true;
  while (frame_captured && window_open) {
    // detect saliency from source images and update intermediate map data
    saliency_map = sal::detect(src, pars, sep_channel_images, feature_maps);

    // show feature channels if debug mode is on
    postprocess::visualize_channel_images(src, sep_channel_images, feature_maps, pars);

    // save output data and update display image
    postprocess::update_output(src, saliency_map, pars.model);

    // updates for the next frame
    frame_captured = cap::update_source(src);

    // check if ESC key entered
    window_open = imtools::win_opened(27, 1, &pars.rbutton_pressed, src.opts.right_click_esc, "Saliency");
  }

  // write model parameters to file if output directory option exists
  postprocess::write_final_parameters(src.opts, pars);
}

#endif  // SALIENCY_PROGRAM_H
