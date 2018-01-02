/* -*- mode: c++; indent-tabs-mode: nil -*- */
#ifndef LOG_TEST_DB_H
#define LOG_TEST_DB_H

#include <sys/stat.h>

#include "log/database.h"
#include "log/file_db.h"
#include "log/file_storage.h"
#include "log/leveldb_db.h"
#include "log/logged_entry.h"
#include "log/sqlite_db.h"
#include "util/test_db.h"

static const unsigned kCertStorageDepth = 3;
static const unsigned kTreeStorageDepth = 8;

template <>
void TestDB<cert_trans::FileDB>::Setup() {
  std::string certs_dir = tmp_.TmpStorageDir() + "/certs";
  std::string tree_dir = tmp_.TmpStorageDir() + "/tree";
  std::string meta_dir = tmp_.TmpStorageDir() + "/meta";
  CHECK_ERR(mkdir(certs_dir.c_str(), 0700));
  CHECK_ERR(mkdir(tree_dir.c_str(), 0700));
  CHECK_ERR(mkdir(meta_dir.c_str(), 0700));

  db_.reset(new cert_trans::FileDB(
      new cert_trans::FileStorage(certs_dir, kCertStorageDepth),
      new cert_trans::FileStorage(tree_dir, kTreeStorageDepth),
      new cert_trans::FileStorage(meta_dir, 0)));
}

template <>
cert_trans::FileDB* TestDB<cert_trans::FileDB>::SecondDB() {
  std::string certs_dir = this->tmp_.TmpStorageDir() + "/certs";
  std::string tree_dir = this->tmp_.TmpStorageDir() + "/tree";
  std::string meta_dir = this->tmp_.TmpStorageDir() + "/meta";
  return new cert_trans::FileDB(
      new cert_trans::FileStorage(certs_dir, kCertStorageDepth),
      new cert_trans::FileStorage(tree_dir, kTreeStorageDepth),
      new cert_trans::FileStorage(meta_dir, 0));
}

template <>
void TestDB<cert_trans::SQLiteDB>::Setup() {
  db_.reset(new cert_trans::SQLiteDB(tmp_.TmpStorageDir() + "/sqlite"));
}

template <>
cert_trans::SQLiteDB* TestDB<cert_trans::SQLiteDB>::SecondDB() {
  return new cert_trans::SQLiteDB(tmp_.TmpStorageDir() + "/sqlite");
}

template <>
void TestDB<cert_trans::LevelDB>::Setup() {
  db_.reset(new cert_trans::LevelDB(tmp_.TmpStorageDir() + "/leveldb"));
}

template <>
cert_trans::LevelDB* TestDB<cert_trans::LevelDB>::SecondDB() {
  // LevelDB won't allow the same DB to be opened concurrently so we have to
  // close the original.
  db_.reset();
  return new cert_trans::LevelDB(tmp_.TmpStorageDir() + "/leveldb");
}

// Not a Database; we just use the same template for setup.
template <>
void TestDB<cert_trans::FileStorage>::Setup() {
  db_.reset(
      new cert_trans::FileStorage(tmp_.TmpStorageDir(), kCertStorageDepth));
}

template <>
cert_trans::FileStorage* TestDB<cert_trans::FileStorage>::SecondDB() {
  return new cert_trans::FileStorage(tmp_.TmpStorageDir(), kCertStorageDepth);
}
#endif  // LOG_TEST_DB_H
