#include "base/notification.h"

#include <glog/logging.h>

using std::chrono::milliseconds;
using std::lock_guard;
using std::mutex;
using std::unique_lock;

namespace cert_trans {


void Notification::Notify() {
  lock_guard<mutex> lock(lock_);
  CHECK(!notified_);
  notified_ = true;
  // *Do* notify this under lock, because otherwise someone can delete this
  // Notification while the thread calling into Notify() is still here.
  // (This removes the TSAN "noise" about the pthread_cond_broadcast /
  // pthread_cond_destroy race.)
  cv_.notify_all();
}


bool Notification::HasBeenNotified() const {
  lock_guard<mutex> lock(lock_);
  return notified_;
}


void Notification::WaitForNotification() const {
  unique_lock<mutex> lock(lock_);
  cv_.wait(lock, [this]() { return notified_; });
}


bool Notification::WaitForNotificationWithTimeout(
    const milliseconds& timeout) const {
  unique_lock<mutex> lock(lock_);
  cv_.wait_for(lock, timeout, [this]() { return notified_; });
  return notified_;
}


}  // namespace cert_trans
