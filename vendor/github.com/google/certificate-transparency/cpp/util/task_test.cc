#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <atomic>
#include <functional>
#include <thread>

#include "base/notification.h"
#include "util/executor.h"
#include "util/status_test_util.h"
#include "util/task.h"
#include "util/testing.h"
#include "util/thread_pool.h"

using cert_trans::Notification;
using cert_trans::ThreadPool;
using std::atomic_fetch_add;
using std::atomic_int;
using std::bind;
using std::chrono::milliseconds;
using std::placeholders::_1;
using std::this_thread::sleep_for;
using std::unique_ptr;

DEFINE_int32(task_test_jiffy_ms, 100,
             "amount of time (in milliseconds) after which \"things ought to "
             "be done\"");

namespace {


class InlineExecutor : public util::Executor {
  void Add(const std::function<void()>& closure) override {
    closure();
  }
  void Delay(const std::chrono::duration<double>& delay,
             util::Task* task) override {
    LOG(FATAL) << "Not Implemented.";
  }
};


void Delay() {
  sleep_for(milliseconds(FLAGS_task_test_jiffy_ms));
}


class DeleteMarker {
 public:
  explicit DeleteMarker(Notification* notifier)
      : notifier_(CHECK_NOTNULL(notifier)) {
  }
  ~DeleteMarker() {
    notifier_->Notify();
  }

 private:
  Notification* const notifier_;

  DISALLOW_COPY_AND_ASSIGN(DeleteMarker);
};


template <class T>
class TaskTest : public ::testing::Test {};


typedef ::testing::Test TaskDeathTest;


template <class TestExecutor, bool sync_done>
class TaskTester {
 public:
  TaskTester()
      : executor_(new TestExecutor),
        task_(bind(&TaskTester::DoneCallback, this, _1), executor_.get()) {
    if (sync_done) {
      ContinueDone();
    }
  }

  ~TaskTester() {
    EXPECT_TRUE(done_started_.WaitForNotificationWithTimeout(
        milliseconds(FLAGS_task_test_jiffy_ms)));
    if (!done_continue_.HasBeenNotified()) {
      done_continue_.Notify();
    }
    EXPECT_TRUE(done_finished_.WaitForNotificationWithTimeout(
        milliseconds(FLAGS_task_test_jiffy_ms)));
  }

  util::Task* task() {
    return &task_;
  }

  void WaitForDoneToStart() {
    EXPECT_TRUE(done_started_.WaitForNotificationWithTimeout(
        milliseconds(FLAGS_task_test_jiffy_ms)));
  }

  bool HasDoneStarted() {
    return done_started_.HasBeenNotified();
  }

  void ContinueDone() {
    if (!done_continue_.HasBeenNotified()) {
      done_continue_.Notify();
    }
  }

  void WaitForDoneToFinish() {
    EXPECT_TRUE(done_finished_.WaitForNotificationWithTimeout(
        milliseconds(FLAGS_task_test_jiffy_ms)));
  }

 private:
  void DoneCallback(util::Task* task) {
    CHECK(!task->IsActive());
    CHECK(task->IsDone());
    done_started_.Notify();
    EXPECT_TRUE(done_continue_.WaitForNotificationWithTimeout(
        milliseconds(FLAGS_task_test_jiffy_ms)));
    done_finished_.Notify();
  }

  const unique_ptr<TestExecutor> executor_;
  util::Task task_;
  Notification done_started_;
  Notification done_continue_;
  Notification done_finished_;

  DISALLOW_COPY_AND_ASSIGN(TaskTester);
};


typedef ::testing::Types<TaskTester<ThreadPool, false>,
                         TaskTester<InlineExecutor, true>> MyTypes;
TYPED_TEST_CASE(TaskTest, MyTypes);


void DoNothing(util::Task* task) {
}


TEST_F(TaskDeathTest, DestroyIncompleteTask) {
  InlineExecutor executor;

  EXPECT_DEATH(util::Task(DoNothing, &executor),
               "Check failed: state_ == DONE");
}


template <class TypeParam>
void WaitForReturnCallback(TypeParam* s, Notification* notifier) {
  s->WaitForDoneToStart();
  EXPECT_FALSE(s->task()->IsActive());
  EXPECT_TRUE(s->task()->IsDone());
  EXPECT_FALSE(s->task()->CancelRequested());
  EXPECT_OK(s->task()->status());
  s->ContinueDone();
  notifier->Notify();
}


TYPED_TEST(TaskTest, StateChanges) {
  ThreadPool pool;
  TypeParam s;

  EXPECT_TRUE(s.task()->IsActive());
  EXPECT_FALSE(s.task()->IsDone());
  EXPECT_FALSE(s.task()->CancelRequested());

  Notification notifier;
  pool.Add(bind(&WaitForReturnCallback<TypeParam>, &s, &notifier));
  EXPECT_TRUE(s.task()->Return());

  EXPECT_TRUE(notifier.WaitForNotificationWithTimeout(
      milliseconds(FLAGS_task_test_jiffy_ms)));
}


TYPED_TEST(TaskTest, DeleteWhenDone) {
  Notification n1;
  Notification n2;
  TypeParam s;

  s.task()->DeleteWhenDone(new DeleteMarker(&n1));
  s.task()->DeleteWhenDone(new DeleteMarker(&n2));

  EXPECT_FALSE(n1.HasBeenNotified());
  EXPECT_FALSE(n2.HasBeenNotified());
  s.task()->Return();
  s.ContinueDone();
  EXPECT_TRUE(n1.WaitForNotificationWithTimeout(
      milliseconds(FLAGS_task_test_jiffy_ms)));
  EXPECT_TRUE(n2.WaitForNotificationWithTimeout(
      milliseconds(FLAGS_task_test_jiffy_ms)));
}


TYPED_TEST(TaskTest, Return) {
  TypeParam s;
  util::Status status(util::error::INVALID_ARGUMENT, "expected status");

  EXPECT_TRUE(s.task()->Return(status));
  EXPECT_FALSE(s.task()->IsActive());

  s.WaitForDoneToStart();
  EXPECT_EQ(status, s.task()->status());
}


TYPED_TEST(TaskTest, HoldBlocksDone) {
  TypeParam s;
  EXPECT_TRUE(s.task()->IsActive());
  EXPECT_FALSE(s.task()->IsDone());

  s.task()->AddHold();
  EXPECT_TRUE(s.task()->IsActive());
  EXPECT_FALSE(s.task()->IsDone());

  s.task()->Return();
  EXPECT_FALSE(s.task()->IsActive());
  EXPECT_FALSE(s.task()->IsDone());

  s.task()->RemoveHold();
  EXPECT_FALSE(s.task()->IsActive());
  EXPECT_TRUE(s.task()->IsDone());
}


TYPED_TEST(TaskTest, MultipleReturn) {
  TypeParam s;
  util::Status status1(util::error::INVALID_ARGUMENT, "expected status");
  util::Status status2(util::error::DEADLINE_EXCEEDED, "unexpected status");
  EXPECT_NE(status1, status2);

  EXPECT_TRUE(s.task()->Return(status1));
  EXPECT_FALSE(s.task()->Return(status2));
  EXPECT_FALSE(s.task()->IsActive());

  s.WaitForDoneToStart();
  EXPECT_EQ(status1, s.task()->status());
}


TYPED_TEST(TaskTest, Cancel) {
  TypeParam s;
  EXPECT_FALSE(s.task()->CancelRequested());
  s.task()->Cancel();
  EXPECT_TRUE(s.task()->CancelRequested());
  s.task()->Return();
}


TYPED_TEST(TaskTest, CancelAfterReturn) {
  TypeParam s;
  EXPECT_FALSE(s.task()->CancelRequested());
  s.task()->Return();
  s.task()->Cancel();
  EXPECT_FALSE(s.task()->CancelRequested());
}


TYPED_TEST(TaskTest, MultipleCancels) {
  TypeParam s;
  EXPECT_FALSE(s.task()->CancelRequested());
  s.task()->Cancel();
  s.task()->Cancel();
  EXPECT_TRUE(s.task()->CancelRequested());
  s.task()->Return();
}


TYPED_TEST(TaskTest, CancelCallback) {
  TypeParam s;
  Notification notifier;

  s.task()->WhenCancelled(bind(&Notification::Notify, &notifier));

  Delay();
  EXPECT_FALSE(notifier.HasBeenNotified());

  s.task()->Cancel();
  EXPECT_TRUE(notifier.WaitForNotificationWithTimeout(
      milliseconds(FLAGS_task_test_jiffy_ms)));
  EXPECT_TRUE(notifier.HasBeenNotified());

  s.task()->Return();
}


void NotifyAndWait(Notification* cancelled, Notification* finish_cancel) {
  cancelled->Notify();
  // Wait for up to two jiffies, because there's a delay of one jiffy
  // in WaitForCancelAndReturn().
  EXPECT_TRUE(finish_cancel->WaitForNotificationWithTimeout(
      milliseconds(FLAGS_task_test_jiffy_ms) * 2));
}


template <class TypeParam>
void WaitForCancelAndReturn(TypeParam* s, Notification* cancelled,
                            Notification* finish_cancel) {
  EXPECT_TRUE(cancelled->WaitForNotificationWithTimeout(
      milliseconds(FLAGS_task_test_jiffy_ms)));
  s->task()->Return();
  Delay();
  EXPECT_FALSE(s->task()->IsDone());
  EXPECT_FALSE(s->HasDoneStarted());
  finish_cancel->Notify();
  s->ContinueDone();
}


TYPED_TEST(TaskTest, CancelCallbackBlocksDone) {
  ThreadPool pool;
  Notification cancelled;
  Notification finish_cancel;
  // This object must be defined after the two Notification above,
  // because it will be using them all (so they should live at least
  // as long).
  TypeParam s;

  s.task()->WhenCancelled(bind(NotifyAndWait, &cancelled, &finish_cancel));
  pool.Add(
      bind(WaitForCancelAndReturn<TypeParam>, &s, &cancelled, &finish_cancel));
  s.task()->Cancel();

  // Make sure everything is completed.
  Delay();
}


TYPED_TEST(TaskTest, CancelCallbackNotCalledOnReturn) {
  TypeParam s;
  Notification notifier;

  s.task()->WhenCancelled(bind(&Notification::Notify, &notifier));

  s.task()->Return();
  s.ContinueDone();
  s.WaitForDoneToFinish();

  EXPECT_FALSE(notifier.HasBeenNotified());
}


TYPED_TEST(TaskTest, CancelCallbackNotCalledAfterReturn) {
  TypeParam s;
  Notification notifier;

  s.task()->Return();
  s.task()->WhenCancelled(bind(&Notification::Notify, &notifier));
  s.ContinueDone();
  s.WaitForDoneToFinish();

  EXPECT_FALSE(notifier.HasBeenNotified());
}


TYPED_TEST(TaskTest, CancelCallbackAfterCancel) {
  TypeParam s;
  Notification notifier;

  s.task()->Cancel();
  s.task()->WhenCancelled(bind(&Notification::Notify, &notifier));
  Delay();
  s.task()->Return();

  EXPECT_TRUE(notifier.WaitForNotificationWithTimeout(
      milliseconds(FLAGS_task_test_jiffy_ms)));
}


TYPED_TEST(TaskTest, ReturnStillRunsCancelCallbacks) {
  TypeParam s;
  Notification notifier;

  s.task()->Cancel();
  s.task()->WhenCancelled(bind(&Notification::Notify, &notifier));
  s.task()->Return();

  EXPECT_TRUE(notifier.WaitForNotificationWithTimeout(
      milliseconds(FLAGS_task_test_jiffy_ms)));
}


void Increment(atomic_int* atomic) {
  atomic->fetch_add(1);
}


TYPED_TEST(TaskTest, MultiCancel) {
  TypeParam s;
  atomic_int count(0);
  const int kRuns(3);

  for (int i = 0; i < kRuns; ++i) {
    s.task()->WhenCancelled(bind(Increment, &count));
  }

  s.task()->Cancel();
  s.task()->Return();
  s.WaitForDoneToStart();
  EXPECT_EQ(kRuns, count.load());
}


TYPED_TEST(TaskTest, ChildBlocksDone) {
  TypeParam s;

  util::Task* const child_task(s.task()->AddChild(DoNothing));
  EXPECT_FALSE(s.task()->CancelRequested());
  EXPECT_TRUE(s.task()->IsActive());
  EXPECT_FALSE(s.task()->IsDone());
  EXPECT_FALSE(child_task->IsDone());

  s.task()->Return();
  EXPECT_FALSE(s.task()->IsActive());
  EXPECT_FALSE(s.task()->IsDone());

  child_task->Return();
}


TYPED_TEST(TaskTest, ChildCancel) {
  TypeParam s;

  util::Task* const child_task(s.task()->AddChild(DoNothing));
  EXPECT_FALSE(s.task()->CancelRequested());
  EXPECT_FALSE(child_task->CancelRequested());

  s.task()->Cancel();
  EXPECT_TRUE(s.task()->CancelRequested());
  EXPECT_TRUE(child_task->CancelRequested());

  child_task->Return();
  s.task()->Return();
}


TYPED_TEST(TaskTest, ChildReturn) {
  TypeParam s;

  util::Task* const child_task(s.task()->AddChild(DoNothing));
  EXPECT_FALSE(s.task()->CancelRequested());
  EXPECT_FALSE(child_task->CancelRequested());

  s.task()->Return();
  EXPECT_FALSE(s.task()->CancelRequested());
  EXPECT_TRUE(child_task->CancelRequested());

  child_task->Return();
}


}  // namespace


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
