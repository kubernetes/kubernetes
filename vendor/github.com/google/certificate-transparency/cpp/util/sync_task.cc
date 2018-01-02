#include "util/sync_task.h"

#include <glog/logging.h>

using cert_trans::Notification;
using std::bind;

namespace util {


SyncTask::SyncTask(Executor* executor)
    : task_(bind(&Notification::Notify, &notifier_), CHECK_NOTNULL(executor)) {
}


SyncTask::~SyncTask() {
  CHECK(IsDone());
}


bool SyncTask::IsDone() const {
  // We should not use task_.IsDone(), because it becomes true before
  // the callback is called, and a user could then decide to delete
  // the SyncTask, which could cause a crash.
  return notifier_.HasBeenNotified();
}


void SyncTask::Wait() const {
  notifier_.WaitForNotification();
}


void SyncTask::Cancel() {
  task_.Cancel();
}


}  // namespace util
