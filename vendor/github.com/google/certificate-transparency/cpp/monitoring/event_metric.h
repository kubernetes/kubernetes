#ifndef CERT_TRANS_MONITORING_EVENT_METRIC_H_
#define CERT_TRANS_MONITORING_EVENT_METRIC_H_

#include <memory>
#include <mutex>
#include <string>

#include "base/macros.h"
#include "monitoring/counter.h"
#include "monitoring/monitoring.h"

namespace cert_trans {

// A helper class for monitoring counter-type values for which it's interesting
// to also have the number of times the counter has been updated (e.g. anything
// for which you may want to calculate an average increase per event etc.)
//
// This class creates two Counter metrics, one called "|base_name|_overall_sum"
// which contains the sum of all events broken down by labels, and another
// called "|base_name|_count" which contains the number of measurements taken,
// also broken down by labels.
template <class... LabelTypes>
class EventMetric {
 public:
  EventMetric(const std::string& base_name,
              const typename NameType<LabelTypes>::name&... label_names,
              const std::string& help);

  // Records an increment of |amount| specified by |labels|.
  // This increments the "|base_name|_overall_sum" metric by |amount|, and
  // increments the "|base_name|_count" metric by 1.
  void RecordEvent(const LabelTypes&... labels, double amount);

 private:
  std::mutex mutex_;
  std::unique_ptr<Counter<LabelTypes...>> totals_;
  std::unique_ptr<Counter<LabelTypes...>> counts_;

  DISALLOW_COPY_AND_ASSIGN(EventMetric);
};


template <class... LabelTypes>
EventMetric<LabelTypes...>::EventMetric(
    const std::string& base_name,
    const typename NameType<LabelTypes>::name&... label_names,
    const std::string& help)
    : totals_(Counter<LabelTypes...>::New(base_name + "_overall_sum",
                                          label_names..., help)),
      counts_(Counter<LabelTypes...>::New(base_name + "_count", label_names...,
                                          help + " (count)")) {
}


template <class... LabelTypes>
void EventMetric<LabelTypes...>::RecordEvent(const LabelTypes&... labels,
                                             double amount) {
  std::lock_guard<std::mutex> lock(mutex_);
  totals_->IncrementBy(labels..., amount);
  counts_->Increment(labels...);
}


}  // namespace cert_trans


#endif  // CERT_TRANS_MONITORING_EVENT_METRIC_H_
