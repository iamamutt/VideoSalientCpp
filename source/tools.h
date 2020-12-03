#ifndef SALIENCY_TOOLS_H
#define SALIENCY_TOOLS_H

#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <utility>

constexpr double
pi()
{
  return 3.14159265358979323846264338327950288419716939937510582097494459;
}

template<typename T>
T
median(std::vector<T> values)
{
  assert(!values.empty());
  const auto middle_iter = values.begin() + values.size() / 2;
  std::nth_element(values.begin(), middle_iter, values.end());
  if (values.size() % 2 == 0) {
    const auto left_mid_iter = std::max_element(values.begin(), middle_iter);
    return (*left_mid_iter + *middle_iter) / 2;
  } else {
    return *middle_iter;
  }
}

template<typename T = double>
T
mean(const std::vector<T> &values)
{
  auto n = values.size();
  T average;
  auto zero = static_cast<T>(0);
  if (n != 0) {
    average = std::accumulate(values.begin(), values.end(), zero) / n;
  }
  return average;
}

int
str2int(const std::string &str)
{
  std::stringstream ss;
  int num;
  ss >> num;
  return num;
};

std::string
to_lower(const std::string &str)
{
  std::string data = str;
  std::transform(data.begin(), data.end(), data.begin(), [](unsigned char c) { return std::tolower(c); });
  return data;
}

template<typename T = double>
std::vector<T>
linspace(double start, double stop, int num)
{
  std::vector<T> values;

  if (num == 0) return values;

  if (num == 1) {
    values.emplace_back(static_cast<T>(start));
    return values;
  }

  double delta = (stop - start) / static_cast<double>(num - 1);

  for (int i = 0; i < (num - 1); ++i) {
    values.emplace_back(static_cast<T>(start + delta * i));
  }

  values.emplace_back(static_cast<T>(stop));
  return values;
}

double
laplace_gauss_pdf(double qx, double qy, double sigma)
{
  auto c    = 1. / (pi() * std::pow(sigma, 4.));
  auto d    = (std::pow(qx, 2.) + std::pow(qy, 2.)) / (2. * std::pow(sigma, 2.));
  auto dens = c * (1. - d) * exp(-1. * d);
  return dens;
}

double
velocity(double s0, double s1, double scale = 1)
{
  return (s1 - s0) * scale;
}

double
euclidean_dist(double x1, double y1, double x2, double y2, double scale = 1)
{
  return sqrt(pow(velocity(y1, y2, scale), 2) + pow(velocity(x1, x2, scale), 2));
}

double
euclidean_dist(const cv::Point2f &p0, const cv::Point2f &p1, double scale = 1)
{
  return euclidean_dist(p0.x, p0.y, p1.x, p1.y, scale);
}

int
mod_index(int number, int size_limit)
{
  return ((number % size_limit) + size_limit) % size_limit;
}

uint
odd_int(uint value)
{
  value += ((value + 1) % 2);
  return value;
}

template<typename T>
std::vector<T>
slice(std::vector<T> const &v, int start, int stop)
{
  auto first = v.cbegin() + start;
  auto last  = v.cbegin() + stop + 1;
  std::vector<T> vec(first, last);
  return vec;
}

double
sigma_k(int k)
{
  return std::max(0.3 * ((k - 1) * 0.5 - 1) + 0.8, 0.5);
}

double
sigma_prop_k(int k, double p)
{
  return std::max((0.3 * (k - 2) + 0.8) * 2.34 * p, 0.585);
}

std::string
replace_file_ext(const std::string &file, const std::string &ext)
{
  std::filesystem::path path(file);
  path.replace_extension(ext);
  return path.generic_string();
}

std::string
normalize_path(const std::string &parent, const std::string &stem = "")
{
  std::filesystem::path path(parent);
  path /= stem;
  std::filesystem::path first_dir = *path.begin();
  if (first_dir == "." || first_dir == "..") path = std::filesystem::weakly_canonical(path);
  path = path.make_preferred();
  return path.generic_string();
};

std::string
make_dir(const std::string &proposed_path)
{
  std::filesystem::path path(normalize_path(proposed_path));
  std::cout << path << std::endl;
  try {
    std::filesystem::create_directories(path);
  } catch (const std::filesystem::filesystem_error &err) {
    throw err;
  }
  std::cout << "directory created at: " << path << std::endl;
  return path.generic_string();
};

void
write_csv_header(std::ofstream &file, const std::vector<std::string> &name)
{
  auto n = name.size();
  for (int i = 0; i < n; ++i) {
    file << name[i];
    if (i < n - 1) {
      file << ",";
    }
  }
  file << "\n";
};

#endif  // SALIENCY_TOOLS_H
