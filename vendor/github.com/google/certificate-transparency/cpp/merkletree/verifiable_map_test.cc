#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>

#include "merkletree/verifiable_map.h"
#include "util/status_test_util.h"
#include "util/testing.h"
#include "util/util.h"


namespace cert_trans {
namespace {

using std::array;
using std::string;
using std::unique_ptr;
using util::StatusOr;
using util::testing::StatusIs;
using util::ToBase64;


class VerifiableMapTest : public testing::Test {
 public:
  VerifiableMapTest() : map_(new Sha256Hasher()) {
  }

 protected:
  VerifiableMap map_;
};


TEST_F(VerifiableMapTest, TestGetNotFound) {
  const string kKey("unknown_key");

  const StatusOr<string> retrieved(map_.Get(kKey));
  EXPECT_THAT(retrieved.status(), StatusIs(util::error::NOT_FOUND));
}

TEST_F(VerifiableMapTest, TestSetGet) {
  const string kKey("key");
  const string kValue("value");
  map_.Set(kKey, kValue);

  const StatusOr<string> retrieved(map_.Get(kKey));
  EXPECT_OK(retrieved);
  EXPECT_EQ(kValue, retrieved.ValueOrDie());
}


// TODO(alcutter): Lots and lots more tests.


}  // namespace
}  // namespace cert_trans


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
