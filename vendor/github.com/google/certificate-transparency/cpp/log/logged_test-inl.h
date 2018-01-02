/* -*- indent-tabs-mode: nil -*- */
#ifndef CERT_TRANS_LOG_LOGGED_TEST_INL_H_
#define CERT_TRANS_LOG_LOGGED_TEST_INL_H_

#include <string>

#include "proto/cert_serializer.h"
#include "util/testing.h"


template <class Logged>
class LoggedTest : public ::testing::Test {
 protected:
};

TYPED_TEST_CASE(LoggedTest, TestType);

TYPED_TEST(LoggedTest, NonEmptyHash) {
  TypeParam l1;
  l1.RandomForTest();

  EXPECT_FALSE(l1.Hash().empty());
}

TYPED_TEST(LoggedTest, SequenceIsPreserved) {
  TypeParam l1;
  l1.set_sequence_number(42);
  EXPECT_EQ(l1.sequence_number(), (int64_t)42);
}

TYPED_TEST(LoggedTest, SequenceIsNotPreserved) {
  TypeParam l1;
  l1.set_sequence_number(42);
  EXPECT_EQ(l1.sequence_number(), (int64_t)42);

  std::string s1;
  EXPECT_TRUE(l1.SerializeForDatabase(&s1));

  TypeParam l2;
  EXPECT_TRUE(l2.ParseFromDatabase(s1));
  EXPECT_FALSE(l2.has_sequence_number());
}

TYPED_TEST(LoggedTest, DifferentHash) {
  TypeParam l1;
  l1.RandomForTest();

  TypeParam l2;
  l2.RandomForTest();

  EXPECT_NE(l1.Hash(), l2.Hash());
}

TYPED_TEST(LoggedTest, SerializationPreservesHash) {
  TypeParam l1;
  l1.RandomForTest();

  std::string s1;
  EXPECT_TRUE(l1.SerializeForDatabase(&s1));

  TypeParam l2;
  EXPECT_TRUE(l2.ParseFromDatabase(s1));

  EXPECT_EQ(l1.Hash(), l2.Hash());
}

TYPED_TEST(LoggedTest, SerializationPreservesMerkleSerialization) {
  TypeParam l1;
  l1.RandomForTest();

  std::string d1;
  EXPECT_TRUE(l1.SerializeForDatabase(&d1));

  TypeParam l2;
  EXPECT_TRUE(l2.ParseFromDatabase(d1));

  std::string s1;
  EXPECT_TRUE(l1.SerializeForLeaf(&s1));
  std::string s2;
  EXPECT_TRUE(l2.SerializeForLeaf(&s2));

  EXPECT_EQ(s1, s2);
}

TYPED_TEST(LoggedTest, DifferentMerkleSerialization) {
  TypeParam l1;
  l1.RandomForTest();

  TypeParam l2;
  l2.RandomForTest();

  std::string s1;
  EXPECT_TRUE(l1.SerializeForLeaf(&s1));
  std::string s2;
  EXPECT_TRUE(l2.SerializeForLeaf(&s2));

  EXPECT_NE(s1, s2);
}

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  ConfigureSerializerForV1CT();
  srand(time(NULL));
  return RUN_ALL_TESTS();
}

#endif  // CERT_TRANS_LOG_LOGGED_TEST_INL_H_
