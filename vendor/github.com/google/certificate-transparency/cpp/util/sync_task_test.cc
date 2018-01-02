#include <gtest/gtest.h>
#include <chrono>
#include <thread>

#include "util/sync_task.h"
#include "util/testing.h"
#include "util/thread_pool.h"

using cert_trans::ThreadPool;
using std::bind;
using std::chrono::milliseconds;
using std::this_thread::sleep_for;

namespace {


TEST(SyncTaskTest, StateChange) {
  ThreadPool pool;
  util::SyncTask s(&pool);
  EXPECT_FALSE(s.IsDone());

  util::Status status(util::error::INTERNAL, "my own private status");
  s.task()->Return(status);
  s.Wait();

  EXPECT_TRUE(s.IsDone());
  EXPECT_EQ(status, s.status());
}


void DelayReturn(util::Task* task, const util::Status& status) {
  sleep_for(milliseconds(100));
  task->Return(status);
}


TEST(SyncTaskTest, Wait) {
  ThreadPool pool;
  util::SyncTask s(&pool);
  EXPECT_FALSE(s.IsDone());

  util::Status status(util::error::INTERNAL, "my own private status");
  pool.Add(bind(DelayReturn, s.task(), status));
  s.Wait();

  EXPECT_TRUE(s.IsDone());
  EXPECT_EQ(status, s.status());
}


void CancelTask(util::Task* task) {
  task->Return(util::Status::CANCELLED);
}


TEST(SyncTaskTest, CancelBefore) {
  ThreadPool pool;
  util::SyncTask s(&pool);
  EXPECT_FALSE(s.IsDone());

  s.task()->WhenCancelled(bind(CancelTask, s.task()));
  s.Cancel();
  s.Wait();

  EXPECT_TRUE(s.IsDone());
  EXPECT_EQ(util::Status::CANCELLED, s.status());
}


TEST(SyncTaskTest, CancelAfter) {
  ThreadPool pool;
  util::SyncTask s(&pool);
  EXPECT_FALSE(s.IsDone());

  s.task()->WhenCancelled(bind(CancelTask, s.task()));

  util::Status status(util::error::INTERNAL, "my own private status");
  s.task()->Return(status);

  s.Cancel();
  s.Wait();

  EXPECT_TRUE(s.IsDone());
  EXPECT_EQ(status, s.status());
}


}  // namespace


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
