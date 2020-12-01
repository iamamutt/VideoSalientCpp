#ifndef SALIENCY_TIMING_H
#define SALIENCY_TIMING_H

#include <chrono>
#include <ctime>    // localtime
#include <iomanip>  // put_time
#include <iostream>
#include <sstream>  // stringstream
#include <tuple>

namespace timing {

using Nanoseconds  = std::chrono::duration<long long int, std::nano>;
using Milliseconds = std::chrono::duration<long double, std::milli>;
using Seconds      = std::chrono::duration<long double, std::ratio<1, 1>>;

using DefaultClock     = std::conditional<std::chrono::high_resolution_clock::is_steady,
                                      std::chrono::high_resolution_clock,
                                      std::chrono::steady_clock>::type;
using DefaultTimePoint = std::chrono::time_point<DefaultClock, Nanoseconds>;

std::string
date_time_string()
{
  auto now            = std::chrono::system_clock::now();
  std::time_t seconds = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&seconds), "%Y-%m-%d_%H_%M_%S");
  return ss.str();
}

template<typename TP>
TP
time_point_now()
{
  auto time_point = DefaultClock::now();
  // converts default time point to time point TP
  auto time_point_recast = std::chrono::time_point_cast<typename TP::duration, DefaultClock>(time_point);
  return time_point_recast;
}

constexpr auto now = time_point_now<DefaultTimePoint>;

template<typename Duration = Seconds>
auto
elapsed(const DefaultTimePoint &time_point)
{
  auto new_time_point = now();
  auto duration       = new_time_point - time_point;

  // rescale duration from nanoseconds to Duration type
  auto duration_recast = std::chrono::duration_cast<Duration>(duration);

  return std::make_tuple(duration_recast.count(), new_time_point);
}
}  // namespace timing

#endif  // SALIENCY_TIMING_H
