#ifndef SALIENCY_PROGRAM_H
#define SALIENCY_PROGRAM_H

#include "postprocessing.h"
#include "preprocessing.h"

void
run_program(int argc, const char *const *argv)
{
  // initialize capture device and read/load parameters
  auto [src, pars] = preprocess::initialize(argc, argv);

  // for visualization of model parameter adjustments
  static AllTrackbarPositions debug_bar_notch_values;
  preprocess::create_trackbars(src.opts, &pars, &debug_bar_notch_values);

  // setup windows after trackbars to get window sizes with bars
  preprocess::setup_windows(src, pars);

  // stored feature map variants for each channel
  ChannelImages sep_channel_images;

  // single map per feature channel
  FeatureMaps feature_maps;

  // initialize saliency data
  SaliencyMap saliency_map = preprocess::setup_saliency_data(src);

  // start run through frames
  while (!src.status.end_program) {
    if (src.status.start_detection) {
      // detect saliency from source images and update intermediate map data
      sal::detect(src, pars, sep_channel_images, feature_maps, saliency_map);

      // collect summary data from saliency map
      sal::analyze(saliency_map, pars.model);

      // save output data
      postprocess::write_data(src, saliency_map);
    }

    // show main window and feature channels if debug mode is on
    postprocess::map_visualization(src, sep_channel_images, feature_maps, saliency_map, pars);

    // updates for the next frame
    cap::update_source(src);

    // check if ending criteria met
    postprocess::continue_program(src);
  }

  // write model parameters to file if output directory option exists
  postprocess::write_final_parameters(src, pars);
}

#endif  // SALIENCY_PROGRAM_H
