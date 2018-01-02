#ifndef CERT_TRANS_UTIL_STATUS_TEST_UTIL_H_
#define CERT_TRANS_UTIL_STATUS_TEST_UTIL_H_

#include <gmock/gmock-matchers.h>

#include "util/status.h"
#include "util/statusor.h"
#include "util/sync_task.h"
#include "util/task.h"

namespace util {
namespace testing {


class StatusIsMatcher : public ::testing::MatcherInterface<Status> {
 public:
  StatusIsMatcher(const ::testing::Matcher<error::Code>& code_matcher,
                  const ::testing::Matcher<std::string>& message_matcher)
      : code_matcher_(code_matcher), message_matcher_(message_matcher) {
  }

  bool MatchAndExplain(
      Status status, ::testing::MatchResultListener* listener) const override {
    if (!code_matcher_.MatchAndExplain(status.CanonicalCode(), listener)) {
      return false;
    }

    if (!message_matcher_.MatchAndExplain(status.error_message(), listener)) {
      return false;
    }

    return true;
  }

  void DescribeTo(std::ostream* os) const override {
    *os << "util::Status(code ";
    code_matcher_.DescribeTo(os);
    *os << " and message ";
    message_matcher_.DescribeTo(os);
    *os << ")";
  }

 private:
  const ::testing::Matcher<error::Code> code_matcher_;
  const ::testing::Matcher<std::string> message_matcher_;
};


::testing::Matcher<Status> StatusIs(
    const ::testing::Matcher<error::Code>& code_matcher,
    const ::testing::Matcher<std::string>& message_matcher) {
  return ::testing::MakeMatcher(
      new StatusIsMatcher(code_matcher, message_matcher));
}


::testing::Matcher<Status> StatusIs(
    const ::testing::Matcher<error::Code>& code_matcher) {
  return ::testing::MakeMatcher(
      new StatusIsMatcher(code_matcher, ::testing::_));
}


inline ::util::Status ToStatus(const ::util::Status& status) {
  return status;
}


inline ::util::Status ToStatus(::util::Task* task) {
  return task->status();
}


inline ::util::Status ToStatus(const ::util::SyncTask& task) {
  return task.status();
}


template <class T>
::util::Status ToStatus(const ::util::StatusOr<T>& statusor) {
  return statusor.status();
}


#define EXPECT_OK(expr) \
  EXPECT_EQ(util::Status::OK, util::testing::ToStatus(expr))
#define ASSERT_OK(expr) \
  ASSERT_EQ(util::Status::OK, util::testing::ToStatus(expr))


}  // namespace testing
}  // namespace util

#endif  // CERT_TRANS_UTIL_STATUS_TEST_UTIL_H_
