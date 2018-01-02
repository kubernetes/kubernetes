#ifndef CERT_TRANS_MONITORING_LABELLED_VALUES_H_
#define CERT_TRANS_MONITORING_LABELLED_VALUES_H_

#include <glog/logging.h>
#include <chrono>
#include <map>
#include <mutex>

#include "monitoring/metric.h"

namespace cert_trans {


template <class... LabelTypes>
class LabelledValues {
 public:
  LabelledValues(const std::string& name,
                 const typename NameType<LabelTypes>::name&... label_names);

  double Get(const LabelTypes&...) const;

  void Set(const LabelTypes&... labels, double value);

  void Increment(const LabelTypes&...);

  void IncrementBy(const LabelTypes&..., double value);

  std::map<std::vector<std::string>, Metric::TimestampedValue> CurrentValues()
      const;

 private:
  const std::string name_;
  const std::vector<std::string> label_names_;
  mutable std::mutex mutex_;
  std::map<std::tuple<LabelTypes...>,
           std::pair<std::chrono::system_clock::time_point, double>> values_;

  DISALLOW_COPY_AND_ASSIGN(LabelledValues);
};


namespace {


template <std::size_t>
struct i__ {};


template <class Tuple>
void label_values(const Tuple&, std::vector<std::string>*, i__<0>) {
}


template <class Tuple, size_t Pos>
void label_values(const Tuple& t, std::vector<std::string>* values, i__<Pos>) {
  std::ostringstream oss;
  oss << std::get<std::tuple_size<Tuple>::value - Pos>(t);
  CHECK_NOTNULL(values)->push_back(oss.str());
  label_values(t, values, i__<Pos - 1>());
}


template <class... Types>
std::vector<std::string> label_values(const std::tuple<Types...>& t) {
  std::vector<std::string> ret;
  label_values(t, &ret, i__<sizeof...(Types)>());
  return ret;
}


}  // namespace


template <class... LabelTypes>
LabelledValues<LabelTypes...>::LabelledValues(
    const std::string& name,
    const typename NameType<LabelTypes>::name&... label_names)
    : name_(name), label_names_{label_names...} {
}


template <class... LabelTypes>
double LabelledValues<LabelTypes...>::Get(const LabelTypes&... labels) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::tuple<LabelTypes...> key(labels...);
  const auto it(values_.find(key));
  if (it == values_.end()) {
    return 0;
  }
  return it->second.second;
}


template <class... LabelTypes>
void LabelledValues<LabelTypes...>::Set(const LabelTypes&... labels,
                                        double value) {
  std::lock_guard<std::mutex> lock(mutex_);
  values_[std::tuple<LabelTypes...>(labels...)] =
      make_pair(std::chrono::system_clock::now(), value);
}


template <class... LabelTypes>
void LabelledValues<LabelTypes...>::Increment(const LabelTypes&... labels) {
  IncrementBy(labels..., 1);
}


template <class... LabelTypes>
void LabelledValues<LabelTypes...>::IncrementBy(const LabelTypes&... labels,
                                                double amount) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::tuple<LabelTypes...> key(labels...);
  const auto it(values_.find(key));
  if (it == values_.end()) {
    values_[key] = make_pair(std::chrono::system_clock::now(), amount);
    return;
  }
  values_[key] = make_pair(std::chrono::system_clock::now(),
                           values_[key].second + amount);
}


template <class... LabelTypes>
std::map<std::vector<std::string>, Metric::TimestampedValue>
LabelledValues<LabelTypes...>::CurrentValues() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::map<std::vector<std::string>, Metric::TimestampedValue> ret;

  for (const auto& v : values_) {
    ret[label_values(v.first)] = v.second;
  }
  return ret;
}


}  // namespace cert_trans

#endif  // CERT_TRANS_MONITORING_LABELLED_VALUES_H_
