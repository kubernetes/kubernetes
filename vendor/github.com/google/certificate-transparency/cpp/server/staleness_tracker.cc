#include <gflags/gflags.h>
#include <glog/logging.h>
#include <algorithm>
#include <string>
#include <vector>

#include "log/cluster_state_controller.h"
#include "log/logged_entry.h"
#include "server/staleness_tracker.h"
#include "util/thread_pool.h"

namespace libevent = cert_trans::libevent;

using cert_trans::StalenessTracker;
using cert_trans::LoggedEntry;
using std::bind;
using std::chrono::seconds;
using std::lock_guard;
using std::mutex;

DEFINE_int32(staleness_check_delay_secs, 5,
             "number of seconds between node staleness checks");


StalenessTracker::StalenessTracker(
    const ClusterStateController<LoggedEntry>* controller, ThreadPool* pool,
    libevent::Base* event_base)
    : controller_(CHECK_NOTNULL(controller)),
      pool_(CHECK_NOTNULL(pool)),
      event_base_(CHECK_NOTNULL(event_base)),
      task_(pool_),
      node_is_stale_(controller_->NodeIsStale()) {
  event_base_->Delay(seconds(FLAGS_staleness_check_delay_secs),
                     task_.task()->AddChild(
                         bind(&StalenessTracker::UpdateNodeStaleness, this)));
}


StalenessTracker::~StalenessTracker() {
  task_.task()->Return();
  task_.Wait();
}


bool StalenessTracker::IsNodeStale() const {
  lock_guard<mutex> lock(mutex_);
  return node_is_stale_;
}


void StalenessTracker::UpdateNodeStaleness() {
  if (!task_.task()->IsActive()) {
    // We're shutting down, just return.
    return;
  }

  const bool node_is_stale(controller_->NodeIsStale());
  {
    lock_guard<mutex> lock(mutex_);
    node_is_stale_ = node_is_stale;
  }

  event_base_->Delay(seconds(FLAGS_staleness_check_delay_secs),
                     task_.task()->AddChild(
                         bind(&StalenessTracker::UpdateNodeStaleness, this)));
}
