#include <gtest/gtest.h>

#include "base/notification.h"
#include "util/testing.h"

using std::chrono::milliseconds;

namespace {


TEST(NotificationTest, BasicTests) {
  cert_trans::Notification notifier;

  ASSERT_FALSE(notifier.HasBeenNotified());
  EXPECT_FALSE(notifier.WaitForNotificationWithTimeout(milliseconds(0)));
  EXPECT_FALSE(notifier.WaitForNotificationWithTimeout(milliseconds(10)));
  notifier.Notify();
  notifier.WaitForNotification();
  EXPECT_TRUE(notifier.WaitForNotificationWithTimeout(milliseconds(0)));
  EXPECT_TRUE(notifier.WaitForNotificationWithTimeout(milliseconds(10)));
  ASSERT_TRUE(notifier.HasBeenNotified());
}


TEST(NotificationDeathTest, NotifyOnce) {
  cert_trans::Notification notifier;

  ASSERT_FALSE(notifier.HasBeenNotified());
  notifier.Notify();
  ASSERT_TRUE(notifier.HasBeenNotified());
  EXPECT_DEATH(notifier.Notify(), "Check failed: !notified_");
}


}  // namespace


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
