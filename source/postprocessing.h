#ifndef SALIENCY_POSTPROCESSING_H
#define SALIENCY_POSTPROCESSING_H

#include "saliency.h"

namespace postprocess {

void
update_output(Source &src, const cv::Mat &saliency_unit_flt, const ModelParameters &pars)
{
  auto salient_colorized = sal::saliency_output_image(saliency_unit_flt, pars);

  MatVec out_imgs = {imtools::imresize(src.img.curr.I8UC3, src.dim), imtools::imresize(salient_colorized, src.dim)};
  imtools::show_layout_imgs(out_imgs, src.layouts.main);

  if (!src.vid.isOpened()) return;
  src.vid.write(salient_colorized);
}

void
write_final_parameters(const CmdLineOpts &opts, const Parameters &pars)
{
  if (opts.out_dir.empty()) return;
  std::string filename = normalize_path(opts.out_dir, "saliency_parameters.yml");
  params::write_parameters(filename, pars);
  std::cout << "Parameters written to file: " << filename << std::endl;
}

void
image_to_file(const std::string &filename, cv::Mat &image)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  fs << "Data" << image;
  fs.release();
  std::cout << "Writing Done." << std::endl;
}

void
visualize_channel_images(const Source &src,
                         const ChannelImages &sep_channel_images,
                         const FeatureMaps &feature_maps,
                         Parameters &pars)
{
  if (!src.opts.debug) return;
  color::debug::visualize(sep_channel_images.color, pars.chan.color, src.dim.resize, src.layouts.color);
  lines::debug::visualize(sep_channel_images.lines, pars.chan.lines, src.dim.resize, src.layouts.lines);
  flick::debug::visualize(sep_channel_images.flicker, pars.chan.flicker, src.dim.resize, src.layouts.flicker);
  flow::debug::visualize(sep_channel_images.flow, pars.chan.flow, src.dim.resize, src.layouts.flow);
  sal::debug::visualize(feature_maps, pars.model, src.dim.resize, src.layouts.features);
}
}  // namespace postprocess

#endif  // SALIENCY_POSTPROCESSING_H
