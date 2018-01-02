#ifndef CERT_TRANS_MONITORING_PROMETHEUS_H_
#define CERT_TRANS_MONITORING_PROMETHEUS_H_

#include <glog/logging.h>

#include "util/protobuf_util.h"

namespace cert_trans {

void ExportMetricsToPrometheus(std::ostream* os);


void ExportMetricsToHtml(std::ostream* os);


}  // namespace cert_trans

#endif  // CERT_TRANS_MONITORING_PROMETHEUS_H_
