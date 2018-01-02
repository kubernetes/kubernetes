#ifndef CERT_TRANS_MONITORING_LATENCY_H_
#define CERT_TRANS_MONITORING_LATENCY_H_

#include <string>

#include "base/macros.h"
#include "monitoring/counter.h"
#include "monitoring/event_metric.h"
#include "monitoring/monitoring.h"

namespace cert_trans {

class ScopedLatency;


// A helper class for monitoring latency.
// This class creates two Counter metrics, one called "|base_name|_overall_sum"
// which contains the sum of all latencies broken down by labels, and another
// called "|base_name|_count" which contains the number of latency measurements
// taken, also broken down by labels.
//
// To actually measure latency, you can either call RecordLatency() directly
// with a latency sample, or use the ScopedLatency() method to return an object
// which will automatically add a latency measurement consisting of the
// duration between the call to ScopedLatency() and the destruction of the
// returned object.
//
// The |TimeUnit| template parameter is used to specify the unit of the values
// added to the counter, e.g. specifying std::chrono::milliseconds will
// duration_cast all recorded latencies to milliseconds before adding to the
// total count.
//
// |LabelTypes...| works as in the Counter<> and Gauge<> templates.
//
// Example usage:
//
//   static Latency<std::chrono::milliseconds, std::string> latency_by_name(
//      "latency_by_name", "name");
//   ...
//
//   void DoStuffForName(const string& name) {
//     // This will record the latency of this method for each different |name|
//     ScopedLatency latency(latency_by_name.ScopedLatency(name));
//     ...
//     // do stuff
//     ...
//   }
//
//   // Here's an example where there's no one scope so can't easily use the
//   // ScopedLatency helper.
//   void FinishedDoingStuffCallback(const string& name,
//                                   const steady_clock::time_point started_at)
//                                   {
//      latency_by_name.RecordLatency(name, steady_clock::now() - started_at);
//   }
template <class TimeUnit, class... LabelTypes>
class Latency {
 public:
  Latency(const std::string& base_name,
          const typename NameType<LabelTypes>::name&... label_names,
          const std::string& help);

  void RecordLatency(const LabelTypes&... labels,
                     std::chrono::duration<double> latency);

  ScopedLatency GetScopedLatency(const LabelTypes&... labels);

 private:
  EventMetric<LabelTypes...> metric_;

  DISALLOW_COPY_AND_ASSIGN(Latency);
};


// Helper class to automatically calculate and record latency.
// Measures the duration between its construction and destruction times, and
// automatically registers that with the Latency<> class which created it.
class ScopedLatency {
 public:
  ScopedLatency(ScopedLatency&& other) = default;

  ~ScopedLatency() {
    record_latency_(std::chrono::steady_clock::now() - start_);
  }

 private:
  ScopedLatency(
      const std::function<void(std::chrono::duration<double>)>& record_latency)
      : record_latency_(record_latency),
        start_(std::chrono::steady_clock::now()) {
  }

  const std::function<void(std::chrono::duration<double>)> record_latency_;
  const std::chrono::steady_clock::time_point start_;

  template <class TimeUnit, class... LabelTypes>
  friend class Latency;

  DISALLOW_COPY_AND_ASSIGN(ScopedLatency);
};


template <class TimeUnit, class... LabelTypes>
Latency<TimeUnit, LabelTypes...>::Latency(
    const std::string& base_name,
    const typename NameType<LabelTypes>::name&... label_names,
    const std::string& help)
    : metric_(base_name, label_names..., help) {
}


template <class TimeUnit, class... LabelTypes>
void Latency<TimeUnit, LabelTypes...>::RecordLatency(
    const LabelTypes&... labels, std::chrono::duration<double> latency) {
  metric_.RecordEvent(labels...,
                      std::chrono::duration_cast<TimeUnit>(latency).count());
}


template <class TimeUnit, class... LabelTypes>
ScopedLatency Latency<TimeUnit, LabelTypes...>::GetScopedLatency(
    const LabelTypes&... labels) {
  return cert_trans::ScopedLatency(
      std::bind(&Latency<TimeUnit, LabelTypes...>::RecordLatency, this,
                labels..., std::placeholders::_1));
}


}  // namespace cert_trans


#endif  // CERT_TRANS_MONITORING_LATENCY_H_
