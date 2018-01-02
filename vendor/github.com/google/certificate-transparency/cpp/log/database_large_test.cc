/* -*- indent-tabs-mode: nil -*- */
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <set>
#include <string>

#include "log/database.h"
#include "log/file_db.h"
#include "log/file_storage.h"
#include "log/leveldb_db.h"
#include "log/logged_entry.h"
#include "log/sqlite_db.h"
#include "log/test_db.h"
#include "log/test_signer.h"
#include "proto/cert_serializer.h"
#include "util/testing.h"
#include "util/util.h"

DEFINE_int32(database_size, 100,
             "Number of entries to put in the test database. Be careful "
             "choosing this, as the database will fill up your disk (entries "
             "are a few kB each). Maximum is limited to 1 000 000. Also note "
             "that SQLite may be very slow with small batch sizes.");

namespace {

using cert_trans::Database;
using cert_trans::FileDB;
using cert_trans::LevelDB;
using cert_trans::LoggedEntry;
using cert_trans::SQLiteDB;
using std::string;


template <class T>
class LargeDBTest : public ::testing::Test {
 protected:
  LargeDBTest() : test_db_(), test_signer_() {
  }

  ~LargeDBTest() {
  }

  void FillDatabase(int entries) {
    LoggedEntry logged_cert;
    for (int i = 0; i < entries; ++i) {
      test_signer_.CreateUniqueFakeSignature(&logged_cert);
      logged_cert.set_sequence_number(i);
      EXPECT_EQ(Database::OK, db()->CreateSequencedEntry(logged_cert));
    }
  }

  int ReadAllSequencedEntries(int num) {
    std::set<string>::const_iterator it;
    LoggedEntry lookup_cert;
    for (int i = 0; i < num; ++i) {
      EXPECT_EQ(Database::LOOKUP_OK,
                this->db()->LookupByIndex(i, &lookup_cert));
    }
    return num;
  }

  T* db() const {
    return test_db_.db();
  }

  TestDB<T> test_db_;
  TestSigner test_signer_;
};

typedef testing::Types<FileDB, SQLiteDB, LevelDB> Databases;

TYPED_TEST_CASE(LargeDBTest, Databases);

TYPED_TEST(LargeDBTest, Benchmark) {
  int entries = FLAGS_database_size;
  CHECK_GE(entries, 0);
  int original_log_level = FLAGS_minloglevel;

  struct rusage ru_before, ru_after;
  getrusage(RUSAGE_SELF, &ru_before);
  uint64_t realtime_before, realtime_after;
  realtime_before = util::TimeInMilliseconds();
  this->FillDatabase(entries);
  realtime_after = util::TimeInMilliseconds();
  getrusage(RUSAGE_SELF, &ru_after);

  FLAGS_minloglevel = 0;
  LOG(INFO) << "Real time spent creating " << FLAGS_database_size
            << " entries: " << realtime_after - realtime_before << " ms";
  LOG(INFO) << "Peak RSS delta (as reported by getrusage()) was "
            << ru_after.ru_maxrss - ru_before.ru_maxrss << " kB";
  FLAGS_minloglevel = original_log_level;

  getrusage(RUSAGE_SELF, &ru_before);
  realtime_before = util::TimeInMilliseconds();
  CHECK_EQ(FLAGS_database_size,
           this->ReadAllSequencedEntries(FLAGS_database_size));
  realtime_after = util::TimeInMilliseconds();
  getrusage(RUSAGE_SELF, &ru_after);

  FLAGS_minloglevel = 0;
  LOG(INFO) << "Real time spent reading " << FLAGS_database_size
            << " entries, sorted by key: " << realtime_after - realtime_before
            << " ms";
  LOG(INFO) << "Peak RSS delta (as reported by getrusage()) was "
            << ru_after.ru_maxrss - ru_before.ru_maxrss << " kB";
  FLAGS_minloglevel = original_log_level;
}

}  // namespace

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  CHECK_GT(FLAGS_database_size, 0) << "Please specify the test database size";
  CHECK_LE(FLAGS_database_size, 1000000)
      << "Database size exceeds allowed maximum";
  ConfigureSerializerForV1CT();
  return RUN_ALL_TESTS();
}
