#ifndef SALIENCY_POSTPROCESSING_H
#define SALIENCY_POSTPROCESSING_H

#include "saliency_map.h"

namespace postprocess {

void
image_to_file(const std::string &filename, const cv::Mat &image)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  fs << "Data" << image;
  fs.release();
}

static void
write_saliency_file_data(SaliencyMap &data, const cv::Size &dim, const int &frame)
{
  for (size_t i = 0; i < data.contours.size(); ++i) {
    data.file << frame << ","                         // frame
              << dim.width << ","                     // img_width
              << dim.height << ","                    // img_height
              << data.threshold / 255. << ","         // saliency_thresh
              << i + 1 << ","                         // contour_num
              << data.contour_size[i] << ","          // contour_size
              << data.salient_coords[i].x + 1 << ","  // pt_x
              << data.salient_coords[i].y + 1 << ","  // pt_y
              << data.salient_values[i] << "\n";      // salient_value
  }

  data.file << std::flush;
};

void
write_data(Source &src, SaliencyMap &saliency_map)
{
  if (src.status.export_enabled) {
    // write frame to video
    src.vid.write(saliency_map.image);
    write_saliency_file_data(saliency_map, src.dim.size, src.fps.frame);

    // save extra output if input is image, only for first frame
    if (src.fps.frame == 1 && src.status.static_image) {
      auto yml_file = normalize_path(src.opts.out_dir, "saliency_map.yml");
      image_to_file(yml_file, saliency_map.map);
    }
  }
}

void
write_final_parameters(const Source &src, const Parameters &pars)
{
  if (!src.status.export_enabled) return;
  std::string filename = normalize_path(src.opts.out_dir, "saliency_parameters.yml");
  params::write_parameters(filename, pars);
  std::cout << "Parameters written to file: " << filename << std::endl;
}

void
map_visualization(const Source &src,
                  const ChannelImages &sep_channel_images,
                  const FeatureMaps &feature_maps,
                  const SaliencyMap &saliency_map,
                  Parameters &pars)
{
  if (src.opts.no_gui) return;  // skip showing anything

  // show output saliency viewer
  MatVec out_imgs;
  out_imgs.emplace_back(imtools::imresize(src.img.curr.I8UC3, src.dim));
  out_imgs.emplace_back(imtools::imresize(saliency_map.image, src.dim));
  imtools::show_layout_imgs(out_imgs, src.layouts.main);

  if (!src.opts.debug) return;  // skip showing of debug images

  // show debug windows
  color::debug::visualize(sep_channel_images.color, pars.chan.color, src.dim.resize, src.layouts.color);
  lines::debug::visualize(sep_channel_images.lines, pars.chan.lines, src.dim.resize, src.layouts.lines);
  flick::debug::visualize(sep_channel_images.flicker, pars.chan.flicker, src.dim.resize, src.layouts.flicker);
  flow::debug::visualize(sep_channel_images.flow, pars.chan.flow, src.dim.resize, src.layouts.flow);
  saliency::debug::visualize(feature_maps, pars.model, src.dim.resize, src.layouts.features);
}

void
continue_program(Source &src, const Parameters &pars)
{
  if (!src.status.frame_was_captured || src.status.stop_detection) {
    src.status.end_program = true;
    return;
  }

  int key_code = cv::waitKey(1);

  if (src.opts.debug && key_code == 'p') {
    // keycode 112='p'
    std::cout << "Saving parameters to file 'saliency_debug_parameters.yml" << std::endl;
    params::write_parameters("saliency_debug_parameters.yml", pars);
  }

  auto win_is_open = imtools::win_opened(key_code, src.opts.escape_key_code, &src.status.right_mouse_btn_down,
                                         src.opts.right_click_esc, src.layouts.main.winname);

  src.status.end_program = !win_is_open;
}
}  // namespace postprocess

#endif  // SALIENCY_POSTPROCESSING_H
