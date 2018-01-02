#ifndef CERT_TRANS_MONITORING_REGISTRY_H_
#define CERT_TRANS_MONITORING_REGISTRY_H_

#include <mutex>
#include <set>
#include <sstream>

#include "base/macros.h"

namespace cert_trans {
class Metric;


class Registry {
 public:
  static Registry* Instance();

  // Registers a new Metric to be exported.
  // |metric| must remain valid for at least the lifetime of this object
  void AddMetric(const Metric* metric);

  // Resets the registry, removing all references to added Metric objects.
  // This method is only for use in testing.
  void ResetForTestingOnly();

  // Returns the set of currently known metrics.
  std::set<const Metric*> GetMetrics() const;

 private:
  Registry() = default;

  mutable std::mutex mutex_;
  std::set<const Metric*> metrics_;

  friend class RegistryTest;

  DISALLOW_COPY_AND_ASSIGN(Registry);
};


}  // namespace cert_trans

#endif  // CERT_TRANS_MONITORING_REGISTRY_H_
