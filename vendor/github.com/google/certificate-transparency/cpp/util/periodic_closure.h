#ifndef CERT_TRANS_UTIL_PERIODIC_CLOSURE_H_
#define CERT_TRANS_UTIL_PERIODIC_CLOSURE_H_

#include <chrono>
#include <functional>
#include <memory>

#include "base/macros.h"
#include "util/libevent_wrapper.h"

namespace cert_trans {


// Arranges for "closure" to be called every "period". If "closure"
// runs for too long, it will skip to the next period that is in the
// future.
//
// This object should always be destroyed either from a libevent
// callback, or while the libevent dispatcher is not running. It
// should also NOT be destroyed from within "closure".
class PeriodicClosure {
 public:
  PeriodicClosure(const std::shared_ptr<libevent::Base>& base,
                  const std::chrono::duration<double>& period,
                  const std::function<void()>& closure);

 private:
  typedef std::chrono::steady_clock clock;

  void Run();

  const std::shared_ptr<libevent::Base> base_;
  const clock::duration period_;
  const std::function<void()> closure_;

  libevent::Event event_;
  clock::time_point target_run_time_;

  DISALLOW_COPY_AND_ASSIGN(PeriodicClosure);
};


}  // namespace cert_trans

#endif  // CERT_TRANS_UTIL_PERIODIC_CLOSURE_H_
