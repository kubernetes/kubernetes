#include "util/etcd_delete.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <functional>
#include <mutex>

using std::bind;
using std::move;
using std::mutex;
using std::placeholders::_1;
using std::string;
using std::unique_lock;
using std::vector;
using util::Status;
using util::Task;
using util::TaskHold;

DEFINE_int32(etcd_delete_concurrency, 4,
             "number of etcd keys to delete at a time");

namespace cert_trans {
namespace {


class DeleteState {
 public:
  DeleteState(EtcdClient* client, vector<string>&& keys, Task* task)
      : client_(CHECK_NOTNULL(client)),
        task_(CHECK_NOTNULL(task)),
        outstanding_(0),
        keys_(move(keys)),
        it_(keys_.begin()) {
    CHECK_GT(FLAGS_etcd_delete_concurrency, 0);

    if (it_ == keys_.end()) {
      // Nothing to do!
      task_->Return();
    } else {
      StartNextRequest(unique_lock<mutex>(mutex_));
    }
  }

  ~DeleteState() {
    CHECK_EQ(outstanding_, 0);
  }

 private:
  void RequestDone(Task* child_task);
  void StartNextRequest(unique_lock<mutex>&& lock);

  EtcdClient* const client_;
  Task* const task_;
  mutex mutex_;
  int outstanding_;
  const vector<string> keys_;
  vector<string>::const_iterator it_;
};


void DeleteState::RequestDone(Task* child_task) {
  unique_lock<mutex> lock(mutex_);
  --outstanding_;

  // If a child task has an error (except for not found, this is close
  // enough to success), return that error, and do not start any more
  // requests.
  if (!child_task->status().ok() &&
      child_task->status().CanonicalCode() != util::error::NOT_FOUND) {
    lock.unlock();
    task_->Return(child_task->status());
    return;
  }

  if (it_ != keys_.end()) {
    StartNextRequest(move(lock));
  } else {
    if (outstanding_ < 1) {
      // No more keys to get, and this was the last one to complete.
      lock.unlock();
      task_->Return();
    }
  }
}


void DeleteState::StartNextRequest(unique_lock<mutex>&& lock) {
  CHECK(lock.owns_lock());

  if (task_->CancelRequested()) {
    // In case the task uses an inline executor.
    lock.unlock();
    task_->Return(Status::CANCELLED);
    return;
  }

  while (outstanding_ < FLAGS_etcd_delete_concurrency && it_ != keys_.end() &&
         task_->IsActive()) {
    CHECK(lock.owns_lock());
    const string& key(*it_);
    ++it_;
    ++outstanding_;

    // In case the task uses an inline executor.
    lock.unlock();

    client_->ForceDelete(key, task_->AddChild(
                                  bind(&DeleteState::RequestDone, this, _1)));

    // We must be holding the lock to evaluate the loop condition.
    lock.lock();
  }
}


}  // namespace


void EtcdForceDeleteKeys(EtcdClient* client, vector<string>&& keys,
                         Task* task) {
  TaskHold hold(CHECK_NOTNULL(task));
  DeleteState* const state(new DeleteState(client, move(keys), task));
  task->DeleteWhenDone(state);
}


}  // namespace cert_trans
