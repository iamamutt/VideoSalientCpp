/**
    write the parameter defaults to file
    @file write_default_params.cpp
    @author Joseph M. Burling
*/

#include "command_line.h"
#include "parameters.h"
#include "tools.h"

int
main(int argc, char **argv)
{
  auto [opts, parser] = parse_command_line_args(argc, argv);
  if (opts.out_dir.empty()) opts.out_dir = normalize_path(opts.bin_path, "../share");
  auto pars_file = normalize_path(opts.out_dir, "parameters.yml");
  std::cout << "Writing default parameters to: " << pars_file << std::endl;
  params::parameter_defaults(pars_file);
  exit(0);
}
