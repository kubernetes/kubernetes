#ifndef CERT_TRANS_SERVER_STALENESS_TRACKER_H_
#define CERT_TRANS_SERVER_STALENESS_TRACKER_H_

#include <memory>
#include <mutex>
#include <string>

#include "log/logged_entry.h"
#include "util/libevent_wrapper.h"
#include "util/sync_task.h"
#include "util/task.h"

namespace cert_trans {

template <class T>
class ClusterStateController;
class ThreadPool;


class StalenessTracker {
 public:
  // Does not take ownership of its parameters, which must outlive
  // this instance.
  StalenessTracker(const ClusterStateController<LoggedEntry>* controller,
                   ThreadPool* pool, libevent::Base* event_base);
  virtual ~StalenessTracker();

  // Check if we consider our node to be stale
  bool IsNodeStale() const;
  // Update our view of node staleness from the controller. This causes
  // periodic updates to be scheduled
  void UpdateNodeStaleness();

 private:
  const ClusterStateController<LoggedEntry>* const controller_;
  ThreadPool* const pool_;
  libevent::Base* const event_base_;

  util::SyncTask task_;
  mutable std::mutex mutex_;
  bool node_is_stale_;

  DISALLOW_COPY_AND_ASSIGN(StalenessTracker);
};


}  // namespace cert_trans

#endif  // CERT_TRANS_SERVER_STALENESS_TRACKER_H_
