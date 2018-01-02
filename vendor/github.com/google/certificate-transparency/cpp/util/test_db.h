#ifndef UTIL_TEST_DB_H
#define UTIL_TEST_DB_H

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdlib.h>

#include "base/macros.h"
#include "util/util.h"

DEFINE_string(database_test_dir, "/tmp",
              "Test directory for databases that use the disk. We attempt to "
              "remove all created files and directories but data may be left "
              "behind if the program does not exit cleanly.");

class TmpStorage {
 public:
  TmpStorage() : tmp_dir_(FLAGS_database_test_dir) {
    file_base_ = util::CreateTemporaryDirectory(tmp_dir_ + "/ctlogXXXXXX");
    CHECK_EQ(tmp_dir_ + "/ctlog", file_base_.substr(0, tmp_dir_.size() + 6));
    CHECK_EQ(tmp_dir_.size() + 12, file_base_.length());
  }

  ~TmpStorage() {
    // Check again that it is safe to empty file_base_.
    CHECK_EQ(tmp_dir_ + "/ctlog", file_base_.substr(0, tmp_dir_.size() + 6));
    CHECK_EQ(tmp_dir_.size() + 12, file_base_.length());

    std::string command = "rm -r " + file_base_;
    CHECK_ERR(system(command.c_str()))
        << "Failed to delete temporary directory in " << file_base_;
  }

  std::string TmpStorageDir() const {
    return file_base_;
  }

 private:
  std::string tmp_dir_;
  std::string file_base_;
};

// Helper for generating test instances of the databases for typed tests.
template <class T>
class TestDB {
 public:
  TestDB() : tmp_() {
    Setup();
  }

  void Setup();

  T* db() const {
    return db_.get();
  }

  // Build a second database from the current disk state. Caller owns result.
  // Meant to be used for testing resumes from disk.
  // Concurrent behaviour is undefined (depends on the Database
  // implementation).
  T* SecondDB();

 private:
  TmpStorage tmp_;
  std::unique_ptr<T> db_;

  DISALLOW_COPY_AND_ASSIGN(TestDB);
};

#endif  // UTIL_TEST_DB_H
