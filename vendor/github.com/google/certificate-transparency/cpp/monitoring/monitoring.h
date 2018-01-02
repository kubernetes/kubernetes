#ifndef CERT_TRANS_MONITORING_MONITORING_H_
#define CERT_TRANS_MONITORING_MONITORING_H_

#include <gflags/gflags.h>

#include "monitoring/counter.h"
#include "monitoring/gauge.h"

DECLARE_string(monitoring);

namespace cert_trans {


const char kPrometheus[] = "prometheus";
const char kGcm[] = "gcm";


}  // namespace cert_trans


#endif  // CERT_TRANS_MONITORING_MONITORING_H_
