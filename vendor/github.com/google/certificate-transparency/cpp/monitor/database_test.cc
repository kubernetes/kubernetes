#include "monitor/database.h"

#include <gtest/gtest.h>
#include <set>

#include "log/test_signer.h"
#include "monitor/test_db.h"
#include "proto/cert_serializer.h"
#include "util/testing.h"

namespace {

using cert_trans::LoggedEntry;
using ct::SignedTreeHead;
using std::string;

template <class T>
class DBTest : public ::testing::Test {
 protected:
  DBTest() : test_db_(), test_signer_() {
  }

  ~DBTest() {
  }

  T* db() const {
    return test_db_.db();
  }

  TestDB<T> test_db_;
  TestSigner test_signer_;
};

typedef testing::Types<monitor::SQLiteDB> Databases;

typedef monitor::Database DB;

TYPED_TEST_CASE(DBTest, Databases);

TYPED_TEST(DBTest, LookupsOnEmptyDB) {
  SignedTreeHead sth;
  EXPECT_EQ(DB::NOT_FOUND, this->db()->LookupLatestWrittenSTH(&sth));
  EXPECT_EQ(DB::NOT_FOUND, this->db()->LookupSTHByTimestamp(0, &sth));

  string res;
  EXPECT_EQ(DB::NOT_FOUND, this->db()->LookupHashByIndex(1, &res));

  DB::VerificationLevel lvl;
  this->test_signer_.CreateUnique(&sth);
  EXPECT_EQ(DB::NOT_FOUND, this->db()->LookupVerificationLevel(sth, &lvl));
}

TYPED_TEST(DBTest, WriteAndLookupSTHs) {
  SignedTreeHead sth, lookup_sth, lookup_sth2;
  this->test_signer_.CreateUnique(&sth);

  EXPECT_EQ(DB::WRITE_OK, this->db()->WriteSTH(sth));

  EXPECT_EQ(DB::LOOKUP_OK, this->db()->LookupLatestWrittenSTH(&lookup_sth));
  TestSigner::TestEqualTreeHeads(sth, lookup_sth);

  EXPECT_EQ(DB::LOOKUP_OK,
            this->db()->LookupSTHByTimestamp(sth.timestamp(), &lookup_sth2));
  TestSigner::TestEqualTreeHeads(sth, lookup_sth2);
}

TYPED_TEST(DBTest, WriteSTHDuplicateTimestamp) {
  SignedTreeHead sth, sth2, lookup_sth;
  this->test_signer_.CreateUnique(&sth);

  EXPECT_EQ(DB::WRITE_OK, this->db()->WriteSTH(sth));

  sth2.CopyFrom(sth);
  sth2.set_tree_size(sth.tree_size() + 1);
  EXPECT_EQ(DB::DUPLICATE_TIMESTAMP, this->db()->WriteSTH(sth2));

  EXPECT_EQ(DB::LOOKUP_OK, this->db()->LookupLatestWrittenSTH(&lookup_sth));
  TestSigner::TestEqualTreeHeads(sth, lookup_sth);
}

TYPED_TEST(DBTest, LookupLatestWrittenSTH) {
  SignedTreeHead sth, sth2, lookup_sth;
  this->test_signer_.CreateUnique(&sth);
  this->test_signer_.CreateUnique(&sth2);
  // Should be newer already but don't rely on this.
  sth2.set_timestamp(sth.timestamp() + 1000);

  EXPECT_EQ(DB::WRITE_OK, this->db()->WriteSTH(sth2));
  EXPECT_EQ(DB::WRITE_OK, this->db()->WriteSTH(sth));

  EXPECT_EQ(DB::LOOKUP_OK, this->db()->LookupLatestWrittenSTH(&lookup_sth));
  TestSigner::TestEqualTreeHeads(sth, lookup_sth);
}

TYPED_TEST(DBTest, WriteEntryAndLookupHash) {
  LoggedEntry logged;
  this->test_signer_.CreateUnique(&logged);

  EXPECT_EQ(DB::WRITE_OK, this->db()->CreateEntry(logged));

  string res;
  EXPECT_EQ(DB::LOOKUP_OK, this->db()->LookupHashByIndex(1, &res));

  string leaf;
  logged.SerializeForLeaf(&leaf);
  TreeHasher hasher(new Sha256Hasher);
  std::string leaf_hash = hasher.HashLeaf(leaf);
  EXPECT_EQ(leaf_hash, res);
}

TYPED_TEST(DBTest, ModifyVerificationLevels) {
  SignedTreeHead sth;
  this->test_signer_.CreateUnique(&sth);

  EXPECT_EQ(DB::WRITE_OK, this->db()->WriteSTH(sth));

  DB::VerificationLevel lvl;
  EXPECT_EQ(DB::LOOKUP_OK, this->db()->LookupVerificationLevel(sth, &lvl));
  EXPECT_EQ(DB::UNDEFINED, lvl);

  EXPECT_EQ(DB::WRITE_OK,
            this->db()->SetVerificationLevel(sth, DB::SIGNATURE_VERIFIED));

  EXPECT_EQ(DB::LOOKUP_OK, this->db()->LookupVerificationLevel(sth, &lvl));
  EXPECT_EQ(DB::SIGNATURE_VERIFIED, lvl);

  EXPECT_EQ(DB::WRITE_OK,
            this->db()->SetVerificationLevel(sth, DB::TREE_CONFIRMED));

  EXPECT_EQ(DB::LOOKUP_OK, this->db()->LookupVerificationLevel(sth, &lvl));
  EXPECT_EQ(DB::TREE_CONFIRMED, lvl);
}

TYPED_TEST(DBTest, DenyUndefinedVerificationLevel) {
  SignedTreeHead sth;
  this->test_signer_.CreateUnique(&sth);

  EXPECT_EQ(DB::WRITE_OK, this->db()->WriteSTH(sth));

  EXPECT_EQ(DB::NOT_ALLOWED,
            this->db()->SetVerificationLevel(sth, DB::UNDEFINED));
}

}  // namespace

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  ConfigureSerializerForV1CT();
  return RUN_ALL_TESTS();
}
