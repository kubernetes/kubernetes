#ifndef CERT_TRANS_MONITORING_GAUGE_H_
#define CERT_TRANS_MONITORING_GAUGE_H_

#include <chrono>
#include <mutex>
#include <string>

#include "base/macros.h"
#include "monitoring/labelled_values.h"
#include "monitoring/metric.h"


namespace cert_trans {

// A metric whose values can go down as well as up (e.g. memory usage.)
template <class... LabelTypes>
class Gauge : public Metric {
 public:
  static Gauge<LabelTypes...>* New(
      const std::string& name,
      const typename NameType<LabelTypes>::name&... label_names,
      const std::string& help);

  double Get(const LabelTypes&...) const;

  void Set(const LabelTypes&... labels, double value);

  // TODO(alcutter): Not over the moon about having this here.
  std::map<std::vector<std::string>, Metric::TimestampedValue> CurrentValues()
      const override;

 private:
  Gauge(const std::string& name,
        const typename NameType<LabelTypes>::name&... label_names,
        const std::string& help);

  LabelledValues<LabelTypes...> values_;

  DISALLOW_COPY_AND_ASSIGN(Gauge);
};


// static
template <class... LabelTypes>
Gauge<LabelTypes...>* Gauge<LabelTypes...>::New(
    const std::string& name,
    const typename NameType<LabelTypes>::name&... label_names,
    const std::string& help) {
  return new Gauge(name, label_names..., help);
}


template <class... LabelTypes>
Gauge<LabelTypes...>::Gauge(
    const std::string& name,
    const typename NameType<LabelTypes>::name&... label_names,
    const std::string& help)
    : Metric(GAUGE, name, {label_names...}, help),
      values_(name, label_names...) {
}


template <class... LabelTypes>
double Gauge<LabelTypes...>::Get(const LabelTypes&... labels) const {
  return values_.Get(labels...);
}


template <class... LabelTypes>
void Gauge<LabelTypes...>::Set(const LabelTypes&... labels, double value) {
  values_.Set(labels..., value);
}


template <class... LabelTypes>
std::map<std::vector<std::string>, Metric::TimestampedValue>
Gauge<LabelTypes...>::CurrentValues() const {
  return values_.CurrentValues();
}


}  // namespace cert_trans


#endif  // CERT_TRANS_MONITORING_GAUGE_H_
