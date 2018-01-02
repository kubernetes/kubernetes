#ifndef MONITOR_TEST_DB_H
#define MONITOR_TEST_DB_H

#include "monitor/sqlite_db.h"
#include "util/test_db.h"

template <>
void TestDB<monitor::SQLiteDB>::Setup() {
  db_.reset(new monitor::SQLiteDB(tmp_.TmpStorageDir() + "/sqlite"));
}

template <>
monitor::SQLiteDB* TestDB<monitor::SQLiteDB>::SecondDB() {
  return new monitor::SQLiteDB(tmp_.TmpStorageDir() + "/sqlite");
}

#endif  // MONITOR_TEST_DB_H
