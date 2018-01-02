#ifndef CERT_TRANS_MONITORING_COUNTER_H_
#define CERT_TRANS_MONITORING_COUNTER_H_


#include <gflags/gflags.h>
#include <memory>
#include <mutex>
#include <string>

#include "base/macros.h"
#include "monitoring/labelled_values.h"
#include "monitoring/metric.h"

namespace cert_trans {

// A metric which can only increase (e.g. total_requests_served).
template <class... LabelTypes>
class Counter : public Metric {
 public:
  static Counter<LabelTypes...>* New(
      const std::string& name,
      const typename NameType<LabelTypes>::name&... label_names,
      const std::string& help);

  void Increment(const LabelTypes&... labels);

  void IncrementBy(const LabelTypes&... labels, double amount);

  double Get(const LabelTypes&... labels) const;

  std::map<std::vector<std::string>, Metric::TimestampedValue> CurrentValues()
      const override;

 private:
  Counter(const std::string& name,
          const typename NameType<LabelTypes>::name&... label_names,
          const std::string& help);

  LabelledValues<LabelTypes...> values_;

  DISALLOW_COPY_AND_ASSIGN(Counter);
};


// static
template <class... LabelTypes>
Counter<LabelTypes...>* Counter<LabelTypes...>::New(
    const std::string& name,
    const typename NameType<LabelTypes>::name&... label_names,
    const std::string& help) {
  return new Counter(name, label_names..., help);
}


template <class... LabelTypes>
Counter<LabelTypes...>::Counter(
    const std::string& name,
    const typename NameType<LabelTypes>::name&... label_names,
    const std::string& help)
    : Metric(COUNTER, name, {label_names...}, help),
      values_(name, label_names...) {
}


template <class... LabelTypes>
void Counter<LabelTypes...>::Increment(const LabelTypes&... labels) {
  values_.Increment(labels...);
}


template <class... LabelTypes>
void Counter<LabelTypes...>::IncrementBy(const LabelTypes&... labels,
                                         double amount) {
  values_.IncrementBy(labels..., amount);
}


template <class... LabelTypes>
double Counter<LabelTypes...>::Get(const LabelTypes&... labels) const {
  return values_.Get(labels...);
}


template <class... LabelTypes>
std::map<std::vector<std::string>, Metric::TimestampedValue>
Counter<LabelTypes...>::CurrentValues() const {
  return values_.CurrentValues();
}


}  // namespace cert_trans

#endif  // CERT_TRANS_MONITORING_COUNTER_H_
