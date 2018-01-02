#ifndef CERT_TRANS_UTIL_SYNC_TASK_H_
#define CERT_TRANS_UTIL_SYNC_TASK_H_

#include "base/macros.h"
#include "base/notification.h"
#include "util/task.h"

namespace util {


// Helper class to allow using a util::Task in a synchronous manner.
class SyncTask {
 public:
  SyncTask(Executor* executor);

  // REQUIRES: IsDone() returns true.
  ~SyncTask();

  Task* task() {
    return &task_;
  }

  // Returns true once the task is completed.
  bool IsDone() const;

  // Returns the status of the task.
  // REQUIRES: IsDone() returns true.
  Status status() const {
    return task_.status();
  }

  // Blocks until IsDone() returns true.
  void Wait() const;

  // Request the task to cancel itself and return. This does not
  // block, so the task should not be deleted until IsDone() returns
  // true.
  void Cancel();

 private:
  cert_trans::Notification notifier_;
  Task task_;

  DISALLOW_COPY_AND_ASSIGN(SyncTask);
};


}  // namespace util

#endif  // CERT_TRANS_UTIL_SYNC_TASK_H_
