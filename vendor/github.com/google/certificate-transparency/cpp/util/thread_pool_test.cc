#include "util/thread_pool.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <atomic>
#include <memory>

#include "base/notification.h"
#include "util/sync_task.h"
#include "util/testing.h"

namespace cert_trans {

using std::chrono::milliseconds;
using std::chrono::system_clock;
using std::unique_ptr;
using util::SyncTask;

class ThreadPoolTest : public ::testing::Test {
 public:
  ThreadPoolTest() : pool_of_one_(1) {
  }

 protected:
  ThreadPool pool_of_one_;
};

typedef class ThreadPoolTest ThreadPoolDeathTest;


TEST_F(ThreadPoolTest, Delay) {
  SyncTask task(&pool_of_one_);
  pool_of_one_.Delay(milliseconds(200), task.task());
  EXPECT_FALSE(task.IsDone());
  task.Wait();
}


TEST_F(ThreadPoolDeathTest, AddingMoreTasksAfterClosedGoesBang) {
  unique_ptr<ThreadPool> my_pool_of_one(new ThreadPool(1));
  SyncTask task(my_pool_of_one.get());
  my_pool_of_one->Delay(milliseconds(200), task.task());
  EXPECT_DEATH(my_pool_of_one.reset(), "queue_\\.empty()");
  task.Wait();
}


TEST_F(ThreadPoolTest, DelayDoesNotBlockAThread) {
  SyncTask delay_task(&pool_of_one_);
  pool_of_one_.Delay(milliseconds(200), delay_task.task());

  Notification inner_done;
  pool_of_one_.Add([&inner_done]() {
    LOG(WARNING) << "Inner running";
    inner_done.Notify();
  });

  inner_done.WaitForNotification();

  EXPECT_FALSE(delay_task.IsDone());

  delay_task.Wait();
}


TEST_F(ThreadPoolTest, NaturalOrderingPreserved) {
  SyncTask task1(&pool_of_one_);
  SyncTask task2(&pool_of_one_);

  pool_of_one_.Delay(milliseconds(200), task2.task());
  pool_of_one_.Delay(milliseconds(100), task1.task());

  task1.Wait();
  EXPECT_FALSE(task2.IsDone());
  task2.Wait();
}


TEST_F(ThreadPoolTest, CancelsDelayTasks) {
  unique_ptr<ThreadPool> pool(new ThreadPool(1));

  SyncTask task1(&pool_of_one_);

  pool->Delay(milliseconds(500), task1.task());
  pool.reset();

  task1.Wait();
  EXPECT_EQ(util::Status::CANCELLED, task1.status());
}


}  // namespace cert_trans


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
