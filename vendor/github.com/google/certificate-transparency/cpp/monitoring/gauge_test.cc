#include "monitoring/monitoring.h"

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "util/testing.h"

namespace cert_trans {

using std::string;
using std::vector;
using testing::ElementsAre;

class GaugeTest : public ::testing::Test {
 protected:
  template <class... LabelTypes>
  const string& GetName(const Metric& m) {
    return m.Name();
  }

  template <class... LabelTypes>
  const vector<string>& GetLabelNames(const Metric& m) {
    return m.LabelNames();
  }

  template <class... LabelTypes>
  string GetHelp(const Metric& m) {
    return m.Help();
  }
};


TEST_F(GaugeTest, TestGaugeName) {
  std::unique_ptr<Gauge<>> gauge(Gauge<>::New("name", "help"));
  EXPECT_EQ("name", GetName(*gauge));
}


TEST_F(GaugeTest, TestGaugeLabelNamesEmpty) {
  std::unique_ptr<Gauge<>> gauge(Gauge<>::New("name", "help"));
  EXPECT_TRUE(GetLabelNames(*gauge).empty());
}


TEST_F(GaugeTest, TestGaugeLabelNames) {
  std::unique_ptr<Gauge<string, string, int>> gauge(
      Gauge<string, string, int>::New("name", "one", "two", "three", "help"));
  EXPECT_THAT(GetLabelNames(*gauge), ElementsAre("one", "two", "three"));
}


TEST_F(GaugeTest, TestGaugeHelp) {
  std::unique_ptr<Gauge<>> gauge(Gauge<>::New("name", "help"));
  EXPECT_EQ("help", GetHelp(*gauge));
}


TEST_F(GaugeTest, TestGauge) {
  std::unique_ptr<Gauge<>> gauge(Gauge<>::New("name", "help"));
  EXPECT_EQ(0, gauge->Get());
  gauge->Set(15);
  EXPECT_EQ(15, gauge->Get());
  gauge->Set(1);
  EXPECT_EQ(1, gauge->Get());
}


TEST_F(GaugeTest, TestGaugeWithLabels) {
  std::unique_ptr<Gauge<std::string, int>> gauge(
      Gauge<std::string, int>::New("name", "a string", "an int", "help"));
  EXPECT_EQ(0, gauge->Get("hi", 1));
  gauge->Set("hi", 1, 100);
  EXPECT_EQ(100, gauge->Get("hi", 1));
  gauge->Set("hi", 1, 1);
  EXPECT_EQ(1, gauge->Get("hi", 1));
}


TEST_F(GaugeTest, TestGaugeWithLabelsMultiValues) {
  std::unique_ptr<Gauge<std::string, int>> gauge(
      Gauge<std::string, int>::New("name", "a string", "an int", "help"));
  EXPECT_EQ(0, gauge->Get("alpha", 1));
  EXPECT_EQ(0, gauge->Get("alpha", 2));
  EXPECT_EQ(0, gauge->Get("beta", 1));
  EXPECT_EQ(0, gauge->Get("beta", 2));
  gauge->Set("alpha", 1, 100);
  gauge->Set("alpha", 2, 200);
  gauge->Set("beta", 1, 300);
  gauge->Set("beta", 2, 400);
  EXPECT_EQ(100, gauge->Get("alpha", 1));
  EXPECT_EQ(200, gauge->Get("alpha", 2));
  EXPECT_EQ(300, gauge->Get("beta", 1));
  EXPECT_EQ(400, gauge->Get("beta", 2));
}


}  // namespace cert_trans


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
