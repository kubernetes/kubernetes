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

class CounterTest : public ::testing::Test {
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
  const string& GetHelp(const Metric& m) {
    return m.Help();
  }
};


TEST_F(CounterTest, TestCounterName) {
  std::unique_ptr<Counter<>> counter(Counter<>::New("name", "help"));
  EXPECT_EQ("name", GetName(*counter));
}


TEST_F(CounterTest, TestCounterLabelNamesEmpty) {
  std::unique_ptr<Counter<>> counter(Counter<>::New("name", "help"));
  EXPECT_TRUE(GetLabelNames(*counter).empty());
}


TEST_F(CounterTest, TestCounterLabelNames) {
  std::unique_ptr<Counter<string, string, int>> counter(
      Counter<string, string, int>::New("name", "one", "two", "three",
                                        "help"));
  EXPECT_THAT(GetLabelNames(*counter), ElementsAre("one", "two", "three"));
}


TEST_F(CounterTest, TestCounterHelp) {
  std::unique_ptr<Counter<>> counter(Counter<>::New("name", "help"));
  EXPECT_EQ("help", GetHelp(*counter));
}


TEST_F(CounterTest, TestCounter) {
  std::unique_ptr<Counter<>> counter(Counter<>::New("name", "help"));
  counter->Increment();
  EXPECT_EQ(1, counter->Get());
}


TEST_F(CounterTest, TestCounterWithLabels) {
  std::unique_ptr<Counter<std::string, int>> counter(
      Counter<std::string, int>::New("name", "a string", "an int", "help"));
  counter->Increment("hi", 1);
  EXPECT_EQ(1, counter->Get("hi", 1));
}


TEST_F(CounterTest, TestCounterWithLabelsMultiValues) {
  std::unique_ptr<Counter<std::string, int>> counter(
      Counter<std::string, int>::New("name", "a string", "an int", "help"));
  EXPECT_EQ(0, counter->Get("alpha", 1));
  EXPECT_EQ(0, counter->Get("alpha", 2));
  EXPECT_EQ(0, counter->Get("beta", 1));
  EXPECT_EQ(0, counter->Get("beta", 2));
  counter->Increment("alpha", 1);
  counter->IncrementBy("alpha", 2, 2);
  counter->IncrementBy("beta", 1, 3);
  counter->IncrementBy("beta", 2, 4);
  EXPECT_EQ(1, counter->Get("alpha", 1));
  EXPECT_EQ(2, counter->Get("alpha", 2));
  EXPECT_EQ(3, counter->Get("beta", 1));
  EXPECT_EQ(4, counter->Get("beta", 2));
}


}  // namespace cert_trans


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
