#include "monitoring/prometheus/exporter.h"
#include "monitoring/metric.h"
#include "monitoring/prometheus/metrics.pb.h"
#include "monitoring/registry.h"

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::map;
using std::set;
using std::string;
using std::vector;

namespace cert_trans {
namespace {

void AddLabelTypes(::io::prometheus::client::Metric* metric,
                   const std::vector<std::string>& names,
                   const std::vector<std::string>& values) {
  CHECK_NOTNULL(metric);
  CHECK_EQ(names.size(), values.size());
  for (size_t i(0); i < names.size(); ++i) {
    ::io::prometheus::client::LabelPair* label_pair(metric->add_label());
    label_pair->set_name(names[i]);
    label_pair->set_value(values[i]);
  }
}


::io::prometheus::client::MetricFamily PopulateMetricFamily(
    const Metric& metric) {
  ::io::prometheus::client::MetricFamily family;
  family.set_name(metric.Name());
  family.set_help(metric.Help());
  switch (metric.Type()) {
    case Metric::COUNTER:
      family.set_type(io::prometheus::client::MetricType::COUNTER);
      break;
    case Metric::GAUGE:
      family.set_type(io::prometheus::client::MetricType::GAUGE);
      break;
    default:
      LOG(FATAL) << "Unknown metric type: " << metric.Type();
  }
  const map<vector<string>, Metric::TimestampedValue> values(
      metric.CurrentValues());
  const vector<string> label_names(metric.LabelNames());
  for (auto it(values.begin()); it != values.end(); ++it) {
    io::prometheus::client::Metric* m(family.add_metric());
    AddLabelTypes(m, label_names, it->first);
    m->set_timestamp_ms(
        duration_cast<milliseconds>(it->second.first.time_since_epoch())
            .count());
    switch (metric.Type()) {
      case Metric::COUNTER:
        m->mutable_counter()->set_value(it->second.second);
        break;
      case Metric::GAUGE:
        m->mutable_gauge()->set_value(it->second.second);
        break;
      default:
        LOG(FATAL) << "Unknown metric type: " << metric.Type();
    }
  }

  return family;
}


}  // namespace

void ExportMetricsToPrometheus(std::ostream* os) {
  const set<const Metric*> metrics(Registry::Instance()->GetMetrics());

  for (auto it(metrics.begin()); it != metrics.end(); ++it) {
    const ::io::prometheus::client::MetricFamily family(
        PopulateMetricFamily(**it));
    CHECK(WriteDelimitedToOstream(family, os));
  }
}


void ExportMetricsToHtml(std::ostream* os) {
  const set<const Metric*> metrics(Registry::Instance()->GetMetrics());
  *os << "<html>\n"
      << "<body>\n"
      << "  <h1>Metrics</h1>\n";

  *os << "<table>\n";
  bool bg_flip(false);
  for (const auto* m : metrics) {
    *os << "<tr><td style='background-color:#"
        << (bg_flip ? "bbffbb" : "eeffee") << "'><code>\n";
    bg_flip = !bg_flip;

    const ::io::prometheus::client::MetricFamily family(
        PopulateMetricFamily(*m));
    *os << family.DebugString();

    *os << "\n</code></td></tr>\n";
  }
  *os << "</table>\n"
      << "</body>\n"
      << "</html>\n";
}


}  // namespace cert_trans
