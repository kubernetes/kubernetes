#ifndef CERT_TRANS_MONITORING_GCM_EXPORTER_H_
#define CERT_TRANS_MONITORING_GCM_EXPORTER_H_

#include <chrono>
#include <condition_variable>
#include <memory>
#include <thread>

#include "net/url_fetcher.h"
#include "util/executor.h"
#include "util/sync_task.h"

namespace cert_trans {


class GCMExporter {
 public:
  GCMExporter(const std::string& instance_name, UrlFetcher* fetcher,
              util::Executor* executor);
  ~GCMExporter();

 private:
  void RefreshCredentials();
  void RefreshCredentialsDone(UrlFetcher::Response* resp, util::Task* task);

  void CreateMetrics();

  void PushMetrics();
  void PushMetricsDone(UrlFetcher::Response* resp, util::Task* task);

  const std::string instance_name_;
  UrlFetcher* const fetcher_;
  util::Executor* const executor_;
  util::SyncTask task_;
  bool metrics_created_;
  std::chrono::system_clock::time_point token_refreshed_at_;
  std::string bearer_token_;

  friend class GCMExporterTest;
};


}  // namespace cert_trans

#endif  // CERT_TRANS_MONITORING_GCM_EXPORTER_H_
