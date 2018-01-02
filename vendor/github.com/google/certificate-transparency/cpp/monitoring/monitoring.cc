#include <gflags/gflags.h>

#include "monitoring/monitoring.h"

DEFINE_string(monitoring, "prometheus",
              "Which monitoring system to use, one of: prometheus, gcm");
