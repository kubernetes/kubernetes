#include "util/json_wrapper.h"

#include <gtest/gtest.h>
#include <memory>

#include "util/testing.h"
#include "util/util.h"

using std::shared_ptr;
using std::string;

class JsonWrapperTest : public ::testing::Test {};

TEST_F(JsonWrapperTest, LargeInt) {
  int64_t big = 0x123456789aLL;
  shared_ptr<json_object> jint(json_object_new_int64(big), json_object_put);
  const char* jsoned = json_object_to_json_string(jint.get());
  JsonInt jint2(json_tokener_parse(jsoned));
  CHECK_EQ(big, jint2.Value());
}

TEST_F(JsonWrapperTest, UnwrapResponse) {
  static string response(
      "{\"leaf_index\":3,\"audit_path\":"
      "[\"j17CTFWsQGwnQkYsebYS7CondFpbzIo+N1jPi9UrqTI=\","
      "\"QSNVV8/waZ5rezVSTFcSPbKtqjalAwVqdF2Vv0/l3/Q=\"]}");
  static string p1v(
      "8f5ec24c55ac406c2742462c79b612ec2a27745a5bcc8a3e3758cf8bd52ba932");
  static string p2v(
      "41235557cff0699e6b7b35524c57123db2adaa36a503056a745d95bf4fe5dff4");

  JsonObject jresponse(response);
  ASSERT_TRUE(jresponse.Ok());

  JsonInt leaf_index(jresponse, "leaf_index");
  ASSERT_TRUE(leaf_index.Ok());
  EXPECT_EQ(leaf_index.Value(), 3);

  JsonArray audit_path(jresponse, "audit_path");
  ASSERT_TRUE(audit_path.Ok());
  EXPECT_EQ(audit_path.Length(), 2);

  JsonString p1(audit_path, 0);
  ASSERT_TRUE(p1.Ok());
  EXPECT_EQ(util::HexString(p1.FromBase64()), p1v);

  JsonString p2(audit_path, 1);
  ASSERT_TRUE(p2.Ok());
  EXPECT_EQ(util::HexString(p2.FromBase64()), p2v);
}

TEST_F(JsonWrapperTest, PartialEvBuffer) {
  const string partial_input("{ \"foo\": 42 ");
  const shared_ptr<evbuffer> buffer(CHECK_NOTNULL(evbuffer_new()),
                                    evbuffer_free);

  evbuffer_add(buffer.get(), partial_input.data(), partial_input.size());

  JsonObject obj(buffer.get());
  EXPECT_FALSE(obj.Ok());
  EXPECT_EQ(partial_input.size(), evbuffer_get_length(buffer.get()));
}

TEST_F(JsonWrapperTest, FragmentedEvBuffer) {
  const shared_ptr<evbuffer> buffer(CHECK_NOTNULL(evbuffer_new()),
                                    evbuffer_free);
  evbuffer_add_printf(buffer.get(), "{ \"foo\": ");

  // Use a separate buffer and evbuffer_add_buffer, to ensure
  // fragmentation.
  {
    const shared_ptr<evbuffer> buffer2(CHECK_NOTNULL(evbuffer_new()),
                                       evbuffer_free);
    evbuffer_add_printf(buffer2.get(), "42 }");
    evbuffer_add_buffer(buffer.get(), buffer2.get());
  }

  evbuffer_iovec chunks[2];
  EXPECT_LT(1, evbuffer_peek(buffer.get(), -1, /*start_at*/ NULL, chunks, 2));

  JsonObject obj(buffer.get());
  EXPECT_TRUE(obj.Ok());
  EXPECT_EQ(0U, evbuffer_get_length(buffer.get()));
}

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
