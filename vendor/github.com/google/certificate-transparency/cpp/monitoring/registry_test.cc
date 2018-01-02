#include "monitoring/registry.h"

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>

#include "monitoring/monitoring.h"
#include "util/testing.h"

namespace cert_trans {

using std::ostringstream;
using std::string;
using std::unique_ptr;
using std::set;
using testing::AllOf;
using testing::AnyOf;
using testing::Contains;


class RegistryTest : public ::testing::Test {
 public:
  void TearDown() {
    Registry::Instance()->ResetForTestingOnly();
  }

 protected:
  const set<const Metric*>& GetMetrics() {
    return Registry::Instance()->metrics_;
  }
};


TEST_F(RegistryTest, TestAddMetric) {
  unique_ptr<Counter<>> counter(Counter<>::New("name", "help"));
  unique_ptr<Gauge<>> gauge(Gauge<>::New("name", "help"));
  EXPECT_EQ(static_cast<size_t>(2), GetMetrics().size());
  EXPECT_THAT(GetMetrics(),
              AllOf(Contains(counter.get()), Contains(gauge.get())));
}


}  // namespace cert_trans


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
