#include "monitor/sqlite_db.h"

#include <glog/logging.h>
#include <sqlite3.h>

#include "log/sqlite_statement.h"

using sqlite::Statement;
using std::string;

namespace monitor {

SQLiteDB::SQLiteDB(const string& dbfile) : db_(NULL) {
  int ret = sqlite3_open_v2(dbfile.c_str(), &db_, SQLITE_OPEN_READWRITE, NULL);
  if (ret == SQLITE_OK)
    return;
  CHECK_EQ(SQLITE_CANTOPEN, ret) << sqlite3_errmsg(db_);

  // We have to close and reopen to avoid memory leaks.
  CHECK_EQ(SQLITE_OK, sqlite3_close(db_)) << sqlite3_errmsg(db_);
  db_ = NULL;

  CHECK_EQ(SQLITE_OK,
           sqlite3_open_v2(dbfile.c_str(), &db_,
                           SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL)) <<
      sqlite3_errmsg(db_);

  // HINT: AUTOINCREMENT starts at 1

  CHECK_EQ(SQLITE_OK,
           sqlite3_exec(db_,
                        "CREATE TABLE leaves("
                        "sequence INTEGER PRIMARY KEY ASC AUTOINCREMENT, "
                        "cert BLOB, "
                        "cert_chain BLOB, "
                        "leaf_hash BLOB, "  // hash of MerkleTreeLeaf
                        "leaf BLOB"         // MerkleTreeLeaf
                        ")",
                        NULL, NULL, NULL)) << sqlite3_errmsg(db_);

  CHECK_EQ(SQLITE_OK, sqlite3_exec(db_,
                                   "CREATE TABLE trees("
                                   "id INTEGER PRIMARY KEY ASC AUTOINCREMENT, "
                                   "valid INTEGER, "
                                   "timestamp INTEGER UNIQUE, "
                                   "tree_size INTEGER, "
                                   "sth BLOB)",
                                   NULL, NULL, NULL)) << sqlite3_errmsg(db_);

  LOG(INFO) << "New SQLite database created in " << dbfile;
}

SQLiteDB::~SQLiteDB() {
  CHECK_EQ(SQLITE_OK, sqlite3_close(db_)) << sqlite3_errmsg(db_);
}

void SQLiteDB::BeginTransaction() {
  CHECK_EQ(SQLITE_OK, sqlite3_exec(db_, "BEGIN;", NULL, NULL, NULL)) <<
      sqlite3_errmsg(db_);
}

void SQLiteDB::EndTransaction() {
  CHECK_EQ(SQLITE_OK, sqlite3_exec(db_, "COMMIT;", NULL, NULL, NULL)) <<
      sqlite3_errmsg(db_);
}

SQLiteDB::WriteResult SQLiteDB::CreateEntry_(const std::string& leaf,
                                             const std::string& leaf_hash,
                                             const std::string& cert,
                                             const std::string& cert_chain) {
  Statement statement(db_,
                      "INSERT INTO leaves(leaf, leaf_hash, cert, cert_chain) "
                      "VALUES(?, ?, ?, ?)");

  statement.BindBlob(0, leaf);
  statement.BindBlob(1, leaf_hash);
  statement.BindBlob(2, cert);
  statement.BindBlob(3, cert_chain);

  if (statement.Step() != SQLITE_DONE)
    return this->WRITE_FAILED;

  return this->WRITE_OK;
}

SQLiteDB::WriteResult SQLiteDB::WriteSTH_(uint64_t timestamp,
                                          int64_t tree_size,
                                          const std::string& sth) {
  CHECK_GE(tree_size, 0);
  Statement statement(db_,
                      "INSERT INTO trees(timestamp, tree_size, sth) "
                      "VALUES(?, ?, ?)");

  statement.BindUInt64(0, timestamp);
  statement.BindUInt64(1, tree_size);
  statement.BindBlob(2, sth);

  int ret = statement.Step();
  if (ret == SQLITE_CONSTRAINT) {
    Statement s2(db_, "SELECT timestamp FROM trees WHERE timestamp = ?");
    s2.BindUInt64(0, timestamp);
    if (s2.Step() != SQLITE_ROW)
      return this->WRITE_FAILED;
    return this->DUPLICATE_TIMESTAMP;
  }
  if (ret != SQLITE_DONE)
    return this->WRITE_FAILED;

  return this->WRITE_OK;
}

SQLiteDB::LookupResult SQLiteDB::LookupLatestWrittenSTH(
    ct::SignedTreeHead* result) const {
  Statement statement(db_,
                      "SELECT sth FROM trees WHERE id IN "
                      "(SELECT MAX(id) FROM trees)");

  int ret = statement.Step();
  if (ret == SQLITE_DONE)
    return this->NOT_FOUND;
  CHECK_EQ(SQLITE_ROW, ret) << sqlite3_errmsg(db_);

  string sth;
  statement.GetBlob(0, &sth);
  CHECK(result->ParseFromString(sth));

  return this->LOOKUP_OK;
}

SQLiteDB::LookupResult SQLiteDB::LookupHashByIndex(int64_t sequence_number,
                                                   std::string* result) const {
  Statement statement(db_, "SELECT leaf_hash FROM leaves WHERE sequence = ?");

  statement.BindUInt64(0, sequence_number);
  int ret = statement.Step();
  if (ret == SQLITE_DONE)
    return this->NOT_FOUND;

  statement.GetBlob(0, result);

  return this->LOOKUP_OK;
}

SQLiteDB::WriteResult SQLiteDB::SetVerificationLevel_(
    const ct::SignedTreeHead& sth, SQLiteDB::VerificationLevel verify_level) {
  Statement statement(db_, "UPDATE trees SET valid = ? WHERE timestamp = ?");
  statement.BindUInt64(0, verify_level);
  statement.BindUInt64(1, sth.timestamp());

  if (statement.Step() != SQLITE_DONE)
    return this->WRITE_FAILED;
  CHECK_EQ(sqlite3_changes(db_), 1);

  return this->WRITE_OK;
}

SQLiteDB::LookupResult SQLiteDB::LookupSTHByTimestamp(
    uint64_t timestamp, ct::SignedTreeHead* result) const {
  Statement statement(db_, "SELECT sth FROM trees WHERE timestamp = ?");

  statement.BindUInt64(0, timestamp);

  int ret = statement.Step();
  if (ret == SQLITE_DONE)
    return this->NOT_FOUND;

  CHECK_EQ(SQLITE_ROW, ret) << sqlite3_errmsg(db_);

  string sth;
  statement.GetBlob(0, &sth);
  result->ParseFromString(sth);

  return this->LOOKUP_OK;
}

SQLiteDB::LookupResult SQLiteDB::LookupVerificationLevel(
    const ct::SignedTreeHead& sth, SQLiteDB::VerificationLevel* result) const {
  Statement statement(
      db_, "SELECT IFNULL(valid, ?) FROM trees WHERE timestamp = ?");
  statement.BindUInt64(0, this->UNDEFINED);
  statement.BindUInt64(1, sth.timestamp());

  int ret = statement.Step();
  if (ret == SQLITE_DONE)
    return this->NOT_FOUND;

  CHECK_EQ(SQLITE_ROW, ret) << sqlite3_errmsg(db_);
  *result = SQLiteDB::VerificationLevel(statement.GetUInt64(0));

  CHECK_EQ(SQLITE_DONE, statement.Step()) << sqlite3_errmsg(db_);
  return this->LOOKUP_OK;
}

}  // namespace monitor
