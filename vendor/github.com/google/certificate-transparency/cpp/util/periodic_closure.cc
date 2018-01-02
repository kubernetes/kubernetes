#include "util/periodic_closure.h"

#include <glog/logging.h>
#include <functional>

using std::bind;
using std::chrono::duration_cast;
using std::function;
using std::shared_ptr;

namespace cert_trans {


PeriodicClosure::PeriodicClosure(const shared_ptr<libevent::Base>& base,
                                 const std::chrono::duration<double>& period,
                                 const function<void()>& closure)
    : base_(base),
      period_(duration_cast<clock::duration>(period)),
      closure_(closure),
      event_(*base_, -1, 0, bind(&PeriodicClosure::Run, this)),
      target_run_time_(clock::now() + period_) {
  LOG_IF(WARNING, !clock::is_steady)
      << "clock used for PeriodicClosure is not steady";

  event_.Add(target_run_time_ - clock::now());
}


void PeriodicClosure::Run() {
  closure_();

  const clock::time_point now(clock::now());
  while (target_run_time_ <= now) {
    target_run_time_ += period_;
  }

  event_.Add(target_run_time_ - now);
}


}  // namespace cert_trans
