#include "monitoring/gcm/exporter.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sstream>

#include "monitoring/monitoring.h"
#include "monitoring/registry.h"
#include "net/url.h"
#include "util/json_wrapper.h"
#include "util/sync_task.h"

DEFINE_string(google_compute_monitoring_base_url, "", "GCM base URL.");
DEFINE_int32(google_compute_monitoring_push_interval_seconds, 15,
             "Seconds between pushing metric values to GCM.");
DEFINE_string(google_compute_metadata_url,
              "http://metadata/computeMetadata/v1/instance/service-accounts",
              "URL of GCE metadata server.");
DEFINE_string(google_compute_monitoring_service_account, "default",
              "Which GCE service account to use for pushing metrics.");
DEFINE_int32(google_compute_monitoring_credentials_refresh_interval_minutes,
             30,
             "Interval between refreshing the auth credentials to write to "
             "GCM.");
DEFINE_int32(google_compute_monitoring_retry_delay_seconds, 5,
             "Seconds between retrying failed GCM requests.");


namespace cert_trans {

using std::bind;
using std::chrono::minutes;
using std::chrono::seconds;
using std::chrono::system_clock;
using std::make_pair;
using std::mutex;
using std::ostringstream;
using std::placeholders::_1;
using std::string;
using std::thread;
using std::unique_lock;
using std::unique_ptr;
using util::Executor;
using util::SyncTask;
using util::Task;

Counter<>* num_gcm_create_metric_failures =
    Counter<>::New("num_gcm_create_metricpush_failures",
                   "Number of failures to create metric metadata in GCM.");

Counter<>* num_gcm_push_failures =
    Counter<>::New("num_gcm_push_failures",
                   "Number of failures to push metric data to GCM.");

Counter<>* num_gcm_token_fetch_failures =
    Counter<>::New("num_gcm_token_fetch_failures",
                   "Number of failures to fetch GCM auth token");

namespace {


// Label and Metric -name prefix.
const char kCloudPrefix[] = "custom.cloudmonitoring.googleapis.com/ct/";


inline std::string RFC3339Time(const system_clock::time_point& when) {
  const std::time_t now_c(system_clock::to_time_t(when));
  char buf[256];
  CHECK(std::strftime(buf, sizeof(buf), "%FT%T.00Z", std::localtime(&now_c)));
  return buf;
}


}  // namespace


GCMExporter::GCMExporter(const string& instance_name, UrlFetcher* fetcher,
                         Executor* executor)
    : instance_name_(instance_name),
      fetcher_(CHECK_NOTNULL(fetcher)),
      executor_(CHECK_NOTNULL(executor)),
      task_(executor_),
      metrics_created_(false) {
  executor_->Add(bind(&GCMExporter::PushMetrics, this));
}


GCMExporter::~GCMExporter() {
  task_.task()->Cancel();
  task_.Wait();
}


void GCMExporter::RefreshCredentials() {
  VLOG(1) << "Refreshing GCM credentials...";
  UrlFetcher::Request req(
      (URL(FLAGS_google_compute_metadata_url + "/" +
           FLAGS_google_compute_monitoring_service_account + "/token")));
  req.headers.insert(make_pair("Metadata-Flavor", "Google"));

  UrlFetcher::Response* const resp(new UrlFetcher::Response);
  fetcher_->Fetch(req, resp, task_.task()->AddChild(
                                 bind(&GCMExporter::RefreshCredentialsDone,
                                      this, resp, _1)));
}


void GCMExporter::RefreshCredentialsDone(UrlFetcher::Response* resp,
                                         Task* task) {
  unique_ptr<UrlFetcher::Response> resp_deleter(resp);
  if (!task->status().ok() || resp->status_code != 200) {
    LOG(WARNING) << "Failed to refresh GCM credentials, status: "
                 << task->status() << ", response code: " << resp->status_code;
    num_gcm_token_fetch_failures->Increment();
    executor_->Delay(
        seconds(FLAGS_google_compute_monitoring_retry_delay_seconds),
        task_.task()->AddChild(bind(&GCMExporter::RefreshCredentials, this)));
    return;
  }
  token_refreshed_at_ = system_clock::now();

  JsonObject reply(resp->body);
  CHECK(reply.Ok()) << "Failed to parse metadata JSON:\n" << resp->body;
  JsonString bearer(reply, "access_token");
  CHECK(bearer.Ok());
  bearer_token_ = bearer.Value();

  VLOG(1) << "GCM credentials refreshed";

  PushMetrics();
}


namespace {


void AddLabelDescription(const string& key, const string& desc,
                         JsonArray* labels) {
  JsonObject label;
  label.Add("key", kCloudPrefix + key);
  label.Add("description", desc);
  CHECK_NOTNULL(labels)->Add(&label);
}


}  // namespace


void GCMExporter::CreateMetrics() {
  const std::set<const Metric*> metrics(Registry::Instance()->GetMetrics());
  for (auto& m : metrics) {
    CHECK_NOTNULL(m);

    // See
    // https://cloud.google.com/monitoring/v2beta2/metricDescriptors#resource
    // for a description of the structure we're building here.

    JsonArray labels;
    AddLabelDescription("instance",
                        "Instance from which the sample originates.", &labels);
    for (const auto& label : m->LabelNames()) {
      AddLabelDescription(label, label, &labels);
    }

    JsonObject desc;
    switch (m->Type()) {
      case Metric::COUNTER:
        // only gauge type metrics are supported for custom metrics currently:
        // https://cloud.google.com/monitoring/api/metrics#metric-types
        desc.Add("metricType", "gauge");
        break;
      case Metric::GAUGE:
        desc.Add("metricType", "gauge");
        break;
      default:
        LOG(FATAL) << "Unknown type: " << m->Type();
    }
    desc.Add("valueType", "double");

    JsonObject metric;
    metric.Add("name", kCloudPrefix + m->Name());
    metric.Add("description", m->Help());
    metric.Add("labels", labels);
    metric.Add("typeDescriptor", desc);

    do {
      UrlFetcher::Request req((URL(FLAGS_google_compute_monitoring_base_url +
                                   "/metricDescriptors")));
      req.verb = UrlFetcher::Verb::POST;
      req.headers.insert(make_pair("Content-Type", "application/json"));
      req.headers.insert(
          make_pair("Authorization", "Bearer " + bearer_token_));
      req.body = metric.ToString();

      UrlFetcher::Response resp;
      SyncTask task(executor_);
      VLOG(1) << "Creating metric m.Name()...";
      VLOG(2) << req.body;
      fetcher_->Fetch(req, &resp, task.task());
      task.Wait();
      if (!task.status().ok() || resp.status_code != 200) {
        LOG(WARNING) << "Failed to create/update metric metadata; status: "
                     << task.status()
                     << ", response_code: " << resp.status_code;
        num_gcm_create_metric_failures->Increment();
        // TODO(alcutter): consider breaking this up into separate child tasks.
        sleep(FLAGS_google_compute_monitoring_retry_delay_seconds);
        continue;
      }
      VLOG(1) << "Metrics Created.";
      VLOG(2) << resp.body;
      break;
    } while (true);
  }
  metrics_created_ = true;
}


namespace {


void AddLabel(const string& key, const string& value, JsonObject* labels) {
  CHECK_NOTNULL(labels)->Add((kCloudPrefix + key).c_str(), value);
}


}  // namespace


void GCMExporter::PushMetrics() {
  if (task_.task()->CancelRequested()) {
    task_.task()->Return(util::Status::CANCELLED);
    return;
  }

  if (system_clock::now() - token_refreshed_at_ >
      minutes(
          FLAGS_google_compute_monitoring_credentials_refresh_interval_minutes)) {
    // If necessary, asynchronously refresh credentials, and then call this
    // method again when done.
    RefreshCredentials();
    return;
  }

  if (!metrics_created_) {
    CreateMetrics();
  }

  // Build up the JSON write request into this object:
  JsonObject metric_write;
  metric_write.Add("kind", "cloudmonitoring#writeTimeseriesRequest");

  JsonObject common_labels;
  AddLabel("instance", instance_name_, &common_labels);
  metric_write.Add("commonLabels", common_labels);

  const std::set<const Metric*> metrics(Registry::Instance()->GetMetrics());
  JsonArray timeseries;
  for (auto& m : metrics) {
    CHECK_NOTNULL(m);
    for (auto& p : m->CurrentValues()) {
      JsonObject labels;
      for (size_t i(0); i < p.first.size(); ++i) {
        AddLabel(m->LabelName(i), p.first[i], &labels);
      }

      JsonObject desc;
      desc.Add("labels", labels);
      desc.Add("metric", kCloudPrefix + m->Name());

      JsonObject ts;
      ts.Add("timeseriesDesc", desc);

      JsonObject point;
      // According to
      // https://cloud.google.com/monitoring/v2beta2/timeseries/write
      // GAUGE types should have a zero size timerange here
      // Which implies we need to use the current time rather than the time the
      // value was set because there's a [short ~5m] horizon over which GCM
      // won't accept samples.
      const auto now(system_clock::now());
      point.Add("start", RFC3339Time(now));
      point.Add("end", RFC3339Time(now));
      point.Add("doubleValue", p.second.second);
      ts.Add("point", point);

      timeseries.Add(&ts);
    }
  }
  metric_write.Add("timeseries", timeseries);

  UrlFetcher::Request req(
      (URL(FLAGS_google_compute_monitoring_base_url + "/timeseries:write")));
  req.verb = UrlFetcher::Verb::POST;
  req.headers.insert(make_pair("Content-Type", "application/json"));
  req.headers.insert(make_pair("Authorization", "Bearer " + bearer_token_));
  req.body = metric_write.ToString();

  UrlFetcher::Response* resp(new UrlFetcher::Response);
  VLOG(1) << "Pushing metrics...";
  VLOG(2) << req.body;
  fetcher_->Fetch(req, resp,
                  task_.task()->AddChild(
                      bind(&GCMExporter::PushMetricsDone, this, resp, _1)));
}


void GCMExporter::PushMetricsDone(UrlFetcher::Response* resp, Task* task) {
  unique_ptr<UrlFetcher::Response> resp_deleter(resp);
  if (!task->status().ok() || resp->status_code != 200) {
    num_gcm_push_failures->Increment();
    LOG(WARNING) << "Failed to push metrics to GCM, status: " << task->status()
                 << ", reponse code: " << resp->status_code;
  } else {
    VLOG(1) << "Metrics pushed.";
    VLOG(2) << resp->body;
  }

  executor_->Delay(
      seconds(FLAGS_google_compute_monitoring_push_interval_seconds),
      task_.task()->AddChild(bind(&GCMExporter::PushMetrics, this)));
}


}  // namespace cert_trans
