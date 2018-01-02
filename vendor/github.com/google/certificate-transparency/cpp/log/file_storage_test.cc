#include <errno.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <set>
#include <string>

#include "log/file_storage.h"
#include "log/filesystem_ops.h"
#include "log/test_db.h"
#include "util/status_test_util.h"
#include "util/testing.h"
#include "util/util.h"

using cert_trans::FailingFilesystemOps;
using cert_trans::FileStorage;
using std::string;
using util::testing::StatusIs;

namespace {

const unsigned kStorageDepth = 3;

class BasicFileStorageTest : public ::testing::Test {
 protected:
  BasicFileStorageTest() : test_db_() {
  }

  FileStorage* fs() const {
    return test_db_.db();
  }

  TestDB<FileStorage> test_db_;
};

TEST_F(BasicFileStorageTest, Create) {
  string key0("1234xyzw", 8);
  string value0("unicorn", 7);

  string key1("1245abcd", 8);
  string value1("Alice", 5);

  EXPECT_THAT(fs()->LookupEntry(key0, NULL), StatusIs(util::error::NOT_FOUND));
  EXPECT_THAT(fs()->LookupEntry(key1, NULL), StatusIs(util::error::NOT_FOUND));

  EXPECT_OK(fs()->CreateEntry(key0, value0));
  string lookup_result;
  EXPECT_OK(fs()->LookupEntry(key0, &lookup_result));
  EXPECT_EQ(value0, lookup_result);

  EXPECT_OK(fs()->CreateEntry(key1, value1));
  EXPECT_OK(fs()->LookupEntry(key1, &lookup_result));
  EXPECT_EQ(value1, lookup_result);
}

TEST_F(BasicFileStorageTest, Scan) {
  string key0("1234xyzw", 8);
  string value0("unicorn", 7);

  string key1("1245abcd", 8);
  string value1("Alice", 5);

  EXPECT_OK(fs()->CreateEntry(key0, value0));
  EXPECT_OK(fs()->CreateEntry(key1, value1));

  std::set<string> keys;
  keys.insert(key0);
  keys.insert(key1);

  std::set<string> scan_keys = fs()->Scan();
  EXPECT_EQ(keys, scan_keys);
}

TEST_F(BasicFileStorageTest, CreateDuplicate) {
  string key("1234xyzw", 8);
  string value("unicorn", 7);

  EXPECT_THAT(fs()->LookupEntry(key, NULL), StatusIs(util::error::NOT_FOUND));
  EXPECT_OK(fs()->CreateEntry(key, value));
  string lookup_result;
  EXPECT_OK(fs()->LookupEntry(key, &lookup_result));
  EXPECT_EQ(value, lookup_result);

  // Try to log another entry with the same key.
  string new_value("alice", 5);
  EXPECT_THAT(fs()->CreateEntry(key, new_value),
              StatusIs(util::error::ALREADY_EXISTS));
  lookup_result.clear();
  EXPECT_OK(fs()->LookupEntry(key, &lookup_result));

  // Expect to receive the original entry on lookup.
  EXPECT_EQ(value, lookup_result);
}

TEST_F(BasicFileStorageTest, Update) {
  string key("1234xyzw", 8);
  string value("unicorn", 7);

  EXPECT_THAT(fs()->LookupEntry(key, NULL), StatusIs(util::error::NOT_FOUND));
  EXPECT_OK(fs()->CreateEntry(key, value));
  string lookup_result;
  EXPECT_OK(fs()->LookupEntry(key, &lookup_result));
  EXPECT_EQ(value, lookup_result);

  // Update.
  string new_value("alice", 5);
  EXPECT_OK(fs()->UpdateEntry(key, new_value));
  EXPECT_OK(fs()->LookupEntry(key, &lookup_result));

  // Expect to receive the new entry on lookup.
  EXPECT_EQ(new_value, lookup_result);
}

// Test for non-existing keys that are similar to  existing ones.
TEST_F(BasicFileStorageTest, LookupInvalidKey) {
  string key("1234xyzw", 8);
  string value("unicorn", 7);

  string similar_key0("1234xyz", 7);
  string similar_key1("1234xyzv", 8);
  string similar_key2("123", 3);
  string empty_key;

  EXPECT_OK(fs()->CreateEntry(key, value));
  EXPECT_OK(fs()->LookupEntry(key, NULL));
  EXPECT_THAT(fs()->LookupEntry(similar_key0, NULL),
              StatusIs(util::error::NOT_FOUND));
  EXPECT_THAT(fs()->LookupEntry(similar_key1, NULL),
              StatusIs(util::error::NOT_FOUND));
  EXPECT_THAT(fs()->LookupEntry(similar_key2, NULL),
              StatusIs(util::error::NOT_FOUND));
  EXPECT_THAT(fs()->LookupEntry(empty_key, NULL),
              StatusIs(util::error::NOT_FOUND));
}

TEST_F(BasicFileStorageTest, Resume) {
  string key0("1234xyzw", 8);
  string value0("unicorn", 7);

  string key1("1245abcd", 8);
  string value1("Alice", 5);

  EXPECT_OK(fs()->CreateEntry(key0, value0));
  EXPECT_OK(fs()->CreateEntry(key1, value1));

  // A second database.
  FileStorage* db2 = test_db_.SecondDB();

  // Look up and expect to find the entries.
  string lookup_result;
  EXPECT_OK(db2->LookupEntry(key0, &lookup_result));
  EXPECT_EQ(value0, lookup_result);

  EXPECT_OK(db2->LookupEntry(key1, &lookup_result));
  EXPECT_EQ(value1, lookup_result);

  delete db2;
};

TEST_F(BasicFileStorageTest, ScanOnResume) {
  string key0("1234xyzw", 8);
  string value0("unicorn", 7);

  string key1("1245abcd", 8);
  string value1("Alice", 5);

  EXPECT_OK(fs()->CreateEntry(key0, value0));
  EXPECT_OK(fs()->CreateEntry(key1, value1));

  // A second database.
  FileStorage* db2 = test_db_.SecondDB();

  std::set<string> keys;
  keys.insert(key0);
  keys.insert(key1);

  std::set<string> scan_keys = db2->Scan();
  EXPECT_EQ(keys, scan_keys);
  delete db2;
}

class FailingFileStorageDeathTest : public ::testing::Test {
 protected:
  string GetTemporaryDirectory() {
    return util::CreateTemporaryDirectory(tmp_.TmpStorageDir() +
                                          "/ctlogXXXXXX");
  }
  TmpStorage tmp_;
};

TEST(DeathTest, SupportDeath) {
#ifndef EXPECT_DEATH
  FAIL() << "Death tests not supported on this platform.";
#endif
};

// TODO(ekasper): death tests throw the following warning
// (at least on some platforms):
//
// [WARNING] ../src/gtest-death-test.cc:789:: Death tests use fork(),
// which is unsafe particularly in a threaded context.
// For this test, Google Test couldn't detect the number of threads.
//
// Find out why.

TEST_F(FailingFileStorageDeathTest, DieOnFailedCreate) {
  // Profiling run: count file operations.
  FailingFilesystemOps* failing_file_op = new FailingFilesystemOps(-1);
  FileStorage db(GetTemporaryDirectory(), kStorageDepth, failing_file_op);

  // Count ops for constructor.
  int op_count_init = failing_file_op->OpCount();
  ASSERT_GE(op_count_init, 0);

  string key0("1234xyzw", 8);
  string value0("unicorn", 7);

  EXPECT_OK(db.CreateEntry(key0, value0));
  int op_count0 = failing_file_op->OpCount();
  ASSERT_GT(op_count0, op_count_init);

  string key1("1245abcd", 8);
  string value1("Alice", 5);

  EXPECT_OK(db.CreateEntry(key1, value1));
  int op_count1 = failing_file_op->OpCount();
  ASSERT_GT(op_count1, op_count0);

  // Real run. Repeat for each file op individually.
  for (int i = op_count_init; i < op_count0; ++i) {
    FileStorage db(GetTemporaryDirectory(), kStorageDepth,
                   new FailingFilesystemOps(i));
    EXPECT_DEATH_IF_SUPPORTED(db.CreateEntry(key0, value0), "");
  }

  for (int i = op_count0; i < op_count1; ++i) {
    FileStorage db(GetTemporaryDirectory(), kStorageDepth,
                   new FailingFilesystemOps(i));
    EXPECT_OK(db.CreateEntry(key0, value0));
    EXPECT_DEATH_IF_SUPPORTED(db.CreateEntry(key1, value1), "");
  }
};

TEST_F(FailingFileStorageDeathTest, DieOnFailedUpdate) {
  // Profiling run: count file operations.
  FailingFilesystemOps* failing_file_op = new FailingFilesystemOps(-1);
  FileStorage db(GetTemporaryDirectory(), kStorageDepth, failing_file_op);

  string key("1234xyzw", 8);
  string value("unicorn", 7);

  EXPECT_OK(db.CreateEntry(key, value));
  int op_count0 = failing_file_op->OpCount();
  ASSERT_GT(op_count0, 0);

  string new_value("Alice", 5);

  EXPECT_OK(db.UpdateEntry(key, new_value));
  int op_count1 = failing_file_op->OpCount();
  ASSERT_GT(op_count1, op_count0);

  // Real run. Repeat for each file op individually.
  for (int i = op_count0; i < op_count1; ++i) {
    FileStorage db(GetTemporaryDirectory(), kStorageDepth,
                   new FailingFilesystemOps(i));
    EXPECT_OK(db.CreateEntry(key, value));
    EXPECT_DEATH_IF_SUPPORTED(db.UpdateEntry(key, new_value), "");
  }
};

TEST_F(FailingFileStorageDeathTest, ResumeOnFailedCreate) {
  // Profiling run: count file operations.
  FailingFilesystemOps* failing_file_op = new FailingFilesystemOps(-1);
  FileStorage db(GetTemporaryDirectory(), kStorageDepth, failing_file_op);

  string key0("1234xyzw", 8);
  string value0("unicorn", 7);

  int op_count_init = failing_file_op->OpCount();
  EXPECT_OK(db.CreateEntry(key0, value0));
  int op_count0 = failing_file_op->OpCount();
  ASSERT_GT(op_count0, 0);

  string key1("1245abcd", 8);
  string value1("Alice", 5);

  EXPECT_OK(db.CreateEntry(key1, value1));
  int op_count1 = failing_file_op->OpCount();
  ASSERT_GT(op_count1, op_count0);

  // Real run. Repeat for each file op individually.
  for (int i = op_count_init; i < op_count0; ++i) {
    string db_dir = GetTemporaryDirectory();
    FileStorage db(db_dir, kStorageDepth, new FailingFilesystemOps(i));
    EXPECT_DEATH_IF_SUPPORTED(db.CreateEntry(key0, value0), "");
    FileStorage db2(db_dir, kStorageDepth);
    // Entry should not be there, and we should be able to insert it.
    EXPECT_THAT(db2.LookupEntry(key0, NULL), StatusIs(util::error::NOT_FOUND));
    EXPECT_OK(db2.CreateEntry(key0, value0));
    // Look it up to double-check that everything works.
    string lookup_result;
    EXPECT_OK(db2.LookupEntry(key0, &lookup_result));
    EXPECT_EQ(value0, lookup_result);
  }

  for (int i = op_count0; i < op_count1; ++i) {
    string db_dir = GetTemporaryDirectory();
    FileStorage db(db_dir, kStorageDepth, new FailingFilesystemOps(i));
    EXPECT_OK(db.CreateEntry(key0, value0));
    EXPECT_DEATH_IF_SUPPORTED(db.CreateEntry(key1, value1), "");
    FileStorage db2(db_dir, kStorageDepth);
    // First entry should be there just fine.
    string lookup_result;
    EXPECT_OK(db2.LookupEntry(key0, &lookup_result));
    EXPECT_EQ(value0, lookup_result);

    // Second entry should not be there, and we should be able to insert it.
    EXPECT_THAT(db2.LookupEntry(key1, NULL), StatusIs(util::error::NOT_FOUND));
    EXPECT_OK(db2.CreateEntry(key1, value1));
    // Look it up to double-check that everything works.
    EXPECT_OK(db2.LookupEntry(key1, &lookup_result));
    EXPECT_EQ(value1, lookup_result);
  }
}

TEST_F(FailingFileStorageDeathTest, ResumeOnFailedUpdate) {
  // Profiling run: count file operations.
  FailingFilesystemOps* failing_file_op = new FailingFilesystemOps(-1);
  FileStorage db(GetTemporaryDirectory(), kStorageDepth, failing_file_op);

  string key("1234xyzw", 8);
  string value("unicorn", 7);

  EXPECT_OK(db.CreateEntry(key, value));
  int op_count0 = failing_file_op->OpCount();
  ASSERT_GT(op_count0, 0);

  string new_value("Alice", 5);

  EXPECT_OK(db.UpdateEntry(key, new_value));
  int op_count1 = failing_file_op->OpCount();
  ASSERT_GT(op_count1, op_count0);

  // Real run. Repeat for each file op individually.
  for (int i = op_count0; i < op_count1; ++i) {
    string db_dir = GetTemporaryDirectory();
    FileStorage db(db_dir, kStorageDepth, new FailingFilesystemOps(i));
    EXPECT_OK(db.CreateEntry(key, value));
    EXPECT_DEATH_IF_SUPPORTED(db.UpdateEntry(key, new_value), "");
    FileStorage db2(db_dir, kStorageDepth);
    // The entry should be there just fine...
    string lookup_result;
    EXPECT_OK(db2.LookupEntry(key, &lookup_result));
    // ... but it should still have its old value.
    EXPECT_EQ(value, lookup_result);
  }
}

}  // namespace

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
