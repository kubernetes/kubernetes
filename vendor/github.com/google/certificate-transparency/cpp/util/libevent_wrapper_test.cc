#include "util/libevent_wrapper.h"

#include <gtest/gtest.h>

#include "util/testing.h"

namespace cert_trans {
namespace libevent {

void DoNothing() {
}

class LibEventWrapperTest : public ::testing::Test {
 public:
  void ExpectToBeOnEventThread(const bool expect) {
    EXPECT_EQ(expect, Base::OnEventThread());
  }

  void DispatchAnotherBase() {
    std::shared_ptr<Base> base(std::make_shared<Base>());
    base->Add(std::bind(&DoNothing));
    base->DispatchOnce();
  }
};

typedef class LibEventWrapperTest LibEventWrapperDeathTest;


TEST_F(LibEventWrapperTest, TestOnEventThread) {
  ExpectToBeOnEventThread(false);
  std::shared_ptr<Base> base(std::make_shared<Base>());
  base->Add(
      std::bind(&LibEventWrapperTest::ExpectToBeOnEventThread, this, true));
  base->DispatchOnce();
}


TEST_F(LibEventWrapperTest, TestTurtlesAllTheWayDown) {
  ExpectToBeOnEventThread(false);
  std::shared_ptr<Base> base(std::make_shared<Base>());
  base->Add(std::bind(&LibEventWrapperTest::DispatchAnotherBase, this));
  base->DispatchOnce();
}


TEST_F(LibEventWrapperDeathTest, TestCheckNotOnEventThread) {
  // Should be fine:
  Base::CheckNotOnEventThread();
  // But...
  EXPECT_DEATH(
      {
        std::shared_ptr<Base> base(std::make_shared<Base>());
        base->Add(std::bind(&Base::CheckNotOnEventThread));
        base->DispatchOnce();
      },
      "OnEventThread");
}


}  // namespace libevent
}  // namespace cert_trans


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
