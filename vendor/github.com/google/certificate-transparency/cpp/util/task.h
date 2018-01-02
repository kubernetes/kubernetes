// This class is used to coordinate asynchronous work. It runs a
// callback once the work is done, and also supports cancellation, as
// well as memory management.
//
// Typically, a function/method that starts an asynchronous operation
// will take a pointer to a util::Task. The caller keeps ownership of
// the util::Task, and once the operation is complete, the callee
// calls util::Task::Return(), with the status in case of an
// error.
//
// The task object can help with the implementation of the callee, by
// providing memory management, and notifying it when a cancellation
// has been requested.
//
// A task is provided with a util::Executor which it will use to run
// its callbacks. The executor will not be accessed after the done
// callback has started.
//
// Once util::Task::Return() is called, the done callback is run on
// the executor.

#ifndef CERT_TRANS_UTIL_TASK_H_
#define CERT_TRANS_UTIL_TASK_H_

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "base/macros.h"
#include "util/executor.h"
#include "util/status.h"

namespace util {

// The task can be in one of three states: ACTIVE (the initial state),
// PREPARED (the task has a status), or DONE (the done callback can
// run).
//
// A task enters the PREPARED state on the first Return() call.
//
// The task changes from PREPARED to DONE when the following
// conditions are met:
//
//  - there are no remaining holds on the task
//  - all cancellation callbacks have returned
//  - all child task done callbacks have returned
//
class Task {
 public:
  Task(const std::function<void(Task*)>& done_callback, Executor* executor);

  // REQUIRES: task is in DONE state.
  // Tasks can be deleted in their done callback.
  ~Task();

  // Returns the executor passed into the constructor, which will be
  // used for callbacks.
  Executor* executor() const {
    return executor_;
  }

  // Requests that the asynchronous operation (and all its
  // descendants) be cancelled. There is no guarantee that the task is
  // PREPARED or DONE by the time this method returns. Also, the
  // cancellation is merely a request, and could be completely
  // ignored.
  void Cancel();

  // REQUIRES: Return() has been called, which can be verified by
  // calling IsActive().
  Status status() const;

  // Methods used by the implementer of an asynchronous operation (the
  // callee).

  // If the task is ACTIVE, prepares it with the specified Status
  // object, returning true. If the task is no longer ACTIVE (meaning
  // that Return() has already been called), the task is not changed,
  // and false is returned.
  //
  // Note that once Return() is called, the task can reach the DONE
  // state asynchronously and run the callbacks for this task, which
  // might delete the task and state used by the callee. So you must
  // be careful with what is used after calling Return(), including
  // through destructors of locally scoped objects (such as
  // std::lock_guard, for example). An option is to use a TaskHold to
  // ensure the task does not reach the DONE state prematurely.
  bool Return(const Status& status = Status::OK);

  // This can be used to prevent the task from advancing to the DONE
  // state.
  void AddHold();
  void RemoveHold();

  // These two methods allow inspecting the current state of the task.
  bool IsActive() const;
  bool IsDone() const;

  // Returns true once Cancel() is called.
  bool CancelRequested() const;

  // The "cancel_cb" callback will be called the first time Cancel()
  // is called. If Cancel() is never called, then it will just be
  // destroyed without being called. So this function could be called
  // zero or one time, and should thus not be solely responsible for
  // memory management.
  //
  // It is okay to call WhenCancelled() more than once on the same
  // task. There is no ordering guarantee, and since they are called
  // using the executor, they could even be run concurrently.
  //
  // If this method is called after the Cancel() has been called, the
  // callback will be sent to the executor immediately. If the task is
  // not ACTIVE anymore (in other words, if Return() has already been
  // called), the callback will not be run.
  //
  // All cancellation callbacks will be complete before the task
  // enters the DONE state. Effectively, each cancellation callback
  // has a hold on the task while they are running. That hold is
  // removed once the cancellation callback returns, but the callback
  // is allowed to take a hold of its own, if it wants to delay the
  // DONE state further.
  void WhenCancelled(const std::function<void()>& cancel_cb);

  // Child tasks are owned by this task (the child task will be
  // deleted automatically after their done callback has run). All
  // child tasks will be cancelled automatically if this task is
  // cancelled or enters the PREPARED state (when Return() is
  // called). Each child task has a hold on this task that is released
  // once the child's done callback returns, so the parent task will
  // not reach the DONE state until all of its child tasks have
  // finished running. The executor of the child task is the same as
  // that of the parent.
  Task* AddChild(const std::function<void(Task*)>& done_callback) {
    return AddChildWithExecutor(done_callback, executor_);
  }

  // Variant of AddChild() that allows setting a different executor.
  Task* AddChildWithExecutor(const std::function<void(Task*)>& done_callback,
                             Executor* executor);

  // Functions to call once the task is DONE. This could be called
  // before, during, or after the done callback, and so cannot rely on
  // the task itself. An example would be to free the memory that was
  // used to execute the asynchronous operation.
  void CleanupWhenDone(const std::function<void()>& cleanup_cb);

  // Arranges to delete the object once the task is DONE. As this
  // could be run before the done callback, this is meant to be used
  // by the implementation of the asynchronous operation, rather than
  // for data used by the caller.
  template <class T>
  void DeleteWhenDone(T* obj) {
    CleanupWhenDone(std::bind(std::default_delete<T>(), obj));
  }

 private:
  enum State {
    ACTIVE = 0,
    PREPARED = 1,
    DONE = 2,
  };

  void TryDoneTransition(std::unique_lock<std::mutex>* lock);
  void RunCancelCallback(const std::function<void()>& cb);
  void RunCleanupAndDoneCallbacks();
  void RunChildDoneCallback(const std::function<void(Task*)>& done_callback,
                            Task* child_task);

  const std::function<void(Task*)> done_callback_;
  Executor* const executor_;

  mutable std::mutex lock_;
  State state_;
  Status status_;  // not protected by lock_
  bool cancelled_;
  int holds_;
  // References to child tasks are kept as shared pointers to avoid
  // some races.
  std::vector<std::shared_ptr<Task>> child_tasks_;
  std::vector<std::function<void()>> cancel_callbacks_;
  std::vector<std::function<void()>> cleanup_callbacks_;

  DISALLOW_COPY_AND_ASSIGN(Task);
};


// Helper class, that adds a hold on a task, and automatically removes
// it when it goes out of scope.
class TaskHold {
 public:
  TaskHold(Task* task) : task_(task) {
    task_->AddHold();
  }
  ~TaskHold() {
    task_->RemoveHold();
  }

 private:
  Task* const task_;

  DISALLOW_COPY_AND_ASSIGN(TaskHold);
};


}  // namespace util

#endif  // CERT_TRANS_UTIL_Task_H_
