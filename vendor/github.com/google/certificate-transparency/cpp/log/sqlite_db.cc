#include "log/sqlite_db.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sqlite3.h>

#include "log/sqlite_statement.h"
#include "monitoring/latency.h"
#include "monitoring/monitoring.h"
#include "util/util.h"

using std::unique_ptr;
using std::chrono::milliseconds;
using std::lock_guard;
using std::mutex;
using std::ostringstream;
using std::string;
using std::unique_lock;

// Several of these flags pass their value directly through to SQLite PRAGMA
// statements, see the SQLite documentation
// (https://www.sqlite.org/pragma.html) for a description of the various
// values available and the implications they have.

// TODO(pphaneuf): For now, just a flag, but ideally, when adding a
// new node, it would do an initial load of its local database with
// "synchronous" set to OFF, then put it back before starting normal
// operation.
DEFINE_string(sqlite_synchronous_mode, "FULL",
              "Which SQLite synchronous option to use, see SQLite pragma "
              "documentation for details.");

DEFINE_string(sqlite_journal_mode, "WAL",
              "Which SQLite journal_mode option to use, see SQLite pragma "
              "documentation for defails.");

DEFINE_int32(sqlite_cache_size, 100000,
             "Number of 1KB btree pages to keep in memory.");

DEFINE_bool(sqlite_batch_into_transactions, true,
            "Whether to batch operations into transactions behind the "
            "scenes.");
DEFINE_int32(sqlite_transaction_batch_size, 400,
             "Max number of operations to batch into one transaction.");

namespace cert_trans {
namespace {


static Latency<milliseconds, string> latency_by_op_ms(
    "sqlitedb_latency_by_operation_ms", "operation",
    "Database latency in ms broken out by operation");


sqlite3* SQLiteOpen(const string& dbfile) {
  ScopedLatency scoped_latency(latency_by_op_ms.GetScopedLatency("open"));
  sqlite3* retval;

  const int ret(
      sqlite3_open_v2(dbfile.c_str(), &retval, SQLITE_OPEN_READWRITE, NULL));
  if (ret == SQLITE_OK) {
    return retval;
  }
  CHECK_EQ(SQLITE_CANTOPEN, ret) << sqlite3_errmsg(retval);

  // We have to close and reopen to avoid memory leaks.
  CHECK_EQ(SQLITE_OK, sqlite3_close(retval)) << sqlite3_errmsg(retval);
  retval = nullptr;

  CHECK_EQ(SQLITE_OK,
           sqlite3_open_v2(dbfile.c_str(), &retval,
                           SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                           nullptr)) << sqlite3_errmsg(retval);
  CHECK_EQ(SQLITE_OK, sqlite3_exec(retval,
                                   "CREATE TABLE leaves(hash BLOB, "
                                   "entry BLOB, sequence INTEGER UNIQUE)",
                                   nullptr, nullptr, nullptr)) <<
      sqlite3_errmsg(retval);
  CHECK_EQ(SQLITE_OK, sqlite3_exec(retval,
                                   "CREATE INDEX leaves_hash_idx ON "
                                   "leaves(hash)",
                                   nullptr, nullptr, nullptr)) <<
      sqlite3_errmsg(retval);;
  CHECK_EQ(SQLITE_OK,
           sqlite3_exec(
               retval,
               "CREATE TABLE trees(sth BLOB UNIQUE, timestamp INTEGER UNIQUE)",
               nullptr, nullptr, nullptr)) << sqlite3_errmsg(retval);

  CHECK_EQ(SQLITE_OK,
           sqlite3_exec(retval, "CREATE TABLE node(node_id BLOB UNIQUE)",
                        nullptr, nullptr, nullptr)) << sqlite3_errmsg(retval);

  LOG(INFO) << "New SQLite database created in " << dbfile;

  return retval;
}


}  // namespace


class SQLiteDB::Iterator : public Database::Iterator {
 public:
  Iterator(const SQLiteDB* db, int64_t start_index)
      : db_(CHECK_NOTNULL(db)), next_index_(start_index) {
    CHECK_GE(next_index_, 0);
  }

  bool GetNextEntry(LoggedEntry* entry) override {
    CHECK_NOTNULL(entry);
    unique_lock<mutex> lock(db_->lock_);
    if (next_index_ < db_->tree_size_) {
      CHECK_EQ(db_->LookupByIndex(lock, next_index_, entry), db_->LOOKUP_OK);
      ++next_index_;
      return true;
    }

    const bool retval(db_->LookupNextIndex(lock, next_index_, entry) ==
                      db_->LOOKUP_OK);
    if (retval) {
      next_index_ = entry->sequence_number() + 1;
    }

    return retval;
  }

 private:
  const SQLiteDB* const db_;
  int64_t next_index_;
};


SQLiteDB::SQLiteDB(const string& dbfile)
    : db_(SQLiteOpen(dbfile)),
      tree_size_(0),
      transaction_size_(0),
      in_transaction_(false) {
  unique_lock<mutex> lock(lock_);
  {
    ostringstream oss;
    oss << "PRAGMA synchronous = " << FLAGS_sqlite_synchronous_mode;
    sqlite::Statement statement(db_, oss.str().c_str());
    CHECK_EQ(SQLITE_DONE, statement.Step()) << sqlite3_errmsg(db_);
    LOG(WARNING) << "SQLite \"synchronous\" pragma set to "
                 << FLAGS_sqlite_synchronous_mode;
    if (FLAGS_sqlite_batch_into_transactions) {
      LOG(WARNING) << "SQLite running with batched transactions, you should "
                   << "set sqlite_synchronous_mode = FULL !";
    }
  }

  {
    ostringstream oss;
    oss << "PRAGMA journal_mode = " << FLAGS_sqlite_journal_mode;
    sqlite::Statement statement(db_, oss.str().c_str());
    CHECK_EQ(SQLITE_ROW, statement.Step()) << sqlite3_errmsg(db_);
    string mode;
    statement.GetBlob(0, &mode);
    CHECK_STRCASEEQ(mode.c_str(), FLAGS_sqlite_journal_mode.c_str());
    CHECK_EQ(SQLITE_DONE, statement.Step()) << sqlite3_errmsg(db_);
  }

  {
    ostringstream oss;
    oss << "PRAGMA cache_size = " << FLAGS_sqlite_cache_size;
    sqlite::Statement statement(db_, oss.str().c_str());
    CHECK_EQ(SQLITE_DONE, statement.Step()) << sqlite3_errmsg(db_);
  }

  BeginTransaction(lock);
}


SQLiteDB::~SQLiteDB() {
  CHECK_EQ(SQLITE_OK, sqlite3_close(db_)) << sqlite3_errmsg(db_);
}


Database::WriteResult SQLiteDB::CreateSequencedEntry_(
    const LoggedEntry& logged) {
  ScopedLatency latency(
      latency_by_op_ms.GetScopedLatency("create_sequenced_entry"));
  unique_lock<mutex> lock(lock_);

  MaybeStartNewTransaction(lock);

  sqlite::Statement statement(db_,
                              "INSERT INTO leaves(hash, entry, sequence) "
                              "VALUES(?, ?, ?)");
  const string hash(logged.Hash());
  statement.BindBlob(0, hash);

  string data;
  CHECK(logged.SerializeForDatabase(&data));
  statement.BindBlob(1, data);

  CHECK(logged.has_sequence_number());
  statement.BindUInt64(2, logged.sequence_number());

  int ret = statement.Step();
  if (ret == SQLITE_CONSTRAINT) {
    // Check whether we're trying to store a hash/sequence pair which already
    // exists - if it's identical we'll return OK as it could be the fetcher.
    sqlite::Statement s2(
        db_, "SELECT sequence, hash FROM leaves WHERE sequence = ?");
    s2.BindUInt64(0, logged.sequence_number());
    if (s2.Step() == SQLITE_ROW) {
      string existing_hash;
      s2.GetBlob(1, &existing_hash);

      if (logged.sequence_number() == tree_size_) {
        ++tree_size_;
      }

      if (hash == existing_hash) {
        return this->OK;
      }
    }
    return this->SEQUENCE_NUMBER_ALREADY_IN_USE;
  }
  CHECK_EQ(SQLITE_DONE, ret) << sqlite3_errmsg(db_);

  if (logged.sequence_number() == tree_size_) {
    ++tree_size_;
  }

  return this->OK;
}


Database::LookupResult SQLiteDB::LookupByHash(const string& hash,
                                              LoggedEntry* result) const {
  CHECK_NOTNULL(result);
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("lookup_by_hash"));

  lock_guard<mutex> lock(lock_);

  sqlite::Statement statement(db_,
                              "SELECT entry, sequence FROM leaves "
                              "WHERE hash = ? ORDER BY sequence LIMIT 1");

  statement.BindBlob(0, hash);

  int ret = statement.Step();
  if (ret == SQLITE_DONE) {
    return this->NOT_FOUND;
  }
  CHECK_EQ(SQLITE_ROW, ret) << sqlite3_errmsg(db_);

  string data;
  statement.GetBlob(0, &data);
  CHECK(result->ParseFromDatabase(data));

  if (statement.GetType(1) == SQLITE_NULL) {
    result->clear_sequence_number();
  } else {
    result->set_sequence_number(statement.GetUInt64(1));
    if (result->sequence_number() == tree_size_) {
      ++tree_size_;
    }
  }

  return this->LOOKUP_OK;
}


Database::LookupResult SQLiteDB::LookupByIndex(int64_t sequence_number,
                                               LoggedEntry* result) const {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("lookup_by_index"));
  unique_lock<mutex> lock(lock_);

  return LookupByIndex(lock, sequence_number, result);
}


Database::LookupResult SQLiteDB::LookupByIndex(const unique_lock<mutex>& lock,
                                               int64_t sequence_number,
                                               LoggedEntry* result) const {
  CHECK(lock.owns_lock());
  CHECK_GE(sequence_number, 0);
  CHECK_NOTNULL(result);
  sqlite::Statement statement(db_,
                              "SELECT entry, hash FROM leaves "
                              "WHERE sequence = ?");
  statement.BindUInt64(0, sequence_number);
  int ret = statement.Step();
  if (ret == SQLITE_DONE) {
    return this->NOT_FOUND;
  }

  string data;
  statement.GetBlob(0, &data);
  CHECK(result->ParseFromDatabase(data));

  string hash;
  statement.GetBlob(1, &hash);

  CHECK_EQ(result->Hash(), hash);

  result->set_sequence_number(sequence_number);
  if (result->sequence_number() == tree_size_) {
    ++tree_size_;
  }

  return this->LOOKUP_OK;
}


Database::LookupResult SQLiteDB::LookupNextIndex(
    const unique_lock<mutex>& lock, int64_t sequence_number,
    LoggedEntry* result) const {
  CHECK(lock.owns_lock());
  CHECK_GE(sequence_number, 0);
  CHECK_NOTNULL(result);
  sqlite::Statement statement(db_,
                              "SELECT entry, hash, sequence FROM leaves "
                              "WHERE sequence >= ? ORDER BY sequence");
  statement.BindUInt64(0, sequence_number);
  if (statement.Step() == SQLITE_DONE) {
    return this->NOT_FOUND;
  }

  string data;
  statement.GetBlob(0, &data);
  CHECK(result->ParseFromDatabase(data));

  string hash;
  statement.GetBlob(1, &hash);

  CHECK_EQ(result->Hash(), hash);

  result->set_sequence_number(statement.GetUInt64(2));
  if (result->sequence_number() == tree_size_) {
    ++tree_size_;
  }

  return this->LOOKUP_OK;
}


unique_ptr<Database::Iterator> SQLiteDB::ScanEntries(
    int64_t start_index) const {
  return unique_ptr<Iterator>(new Iterator(this, start_index));
}


Database::WriteResult SQLiteDB::WriteTreeHead_(const ct::SignedTreeHead& sth) {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("write_tree_head"));
  unique_lock<mutex> lock(lock_);

  sqlite::Statement statement(db_,
                              "INSERT INTO trees(timestamp, sth) "
                              "VALUES(?, ?)");
  statement.BindUInt64(0, sth.timestamp());

  string sth_data;
  CHECK(sth.SerializeToString(&sth_data));
  statement.BindBlob(1, sth_data);

  int r2 = statement.Step();
  if (r2 == SQLITE_CONSTRAINT) {
    sqlite::Statement s2(db_,
                         "SELECT timestamp,sth FROM trees "
                         "WHERE timestamp = ?");
    s2.BindUInt64(0, sth.timestamp());
    CHECK_EQ(SQLITE_ROW, s2.Step()) << sqlite3_errmsg(db_);
    string existing_sth_data;
    s2.GetBlob(1, &existing_sth_data);
    if (existing_sth_data == sth_data) {
      LOG(WARNING) << "Attempted to store indentical STH in DB.";
      return this->OK;
    }
    return this->DUPLICATE_TREE_HEAD_TIMESTAMP;
  }
  CHECK_EQ(SQLITE_DONE, r2) << sqlite3_errmsg(db_);

  EndTransaction(lock);
  BeginTransaction(lock);

  // Do not call the callbacks while holding the lock, as they might
  // want to perform some lookups.
  lock.unlock();
  callbacks_.Call(sth);

  return this->OK;
}


Database::LookupResult SQLiteDB::LatestTreeHead(
    ct::SignedTreeHead* result) const {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("latest_tree_head"));
  unique_lock<mutex> lock(lock_);

  return LatestTreeHeadNoLock(lock, result);
}


int64_t SQLiteDB::TreeSize() const {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("tree_size"));
  unique_lock<mutex> lock(lock_);

  CHECK_GE(tree_size_, 0);
  sqlite::Statement statement(
      db_,
      "SELECT sequence FROM leaves WHERE sequence >= ? ORDER BY sequence");
  statement.BindUInt64(0, tree_size_);

  int ret(statement.Step());
  while (ret == SQLITE_ROW) {
    const sqlite3_uint64 sequence(statement.GetUInt64(0));

    if (sequence != static_cast<uint64_t>(tree_size_)) {
      return tree_size_;
    }

    ++tree_size_;
    ret = statement.Step();
  }
  CHECK_EQ(SQLITE_DONE, ret) << sqlite3_errmsg(db_);

  return tree_size_;
}


void SQLiteDB::AddNotifySTHCallback(
    const Database::NotifySTHCallback* callback) {
  unique_lock<mutex> lock(lock_);

  callbacks_.Add(callback);

  ct::SignedTreeHead sth;
  if (LatestTreeHeadNoLock(lock, &sth) == this->LOOKUP_OK) {
    // Do not call the callback while holding the lock, as they might
    // want to perform some lookups.
    lock.unlock();
    (*callback)(sth);
  }
}


void SQLiteDB::RemoveNotifySTHCallback(
    const Database::NotifySTHCallback* callback) {
  lock_guard<mutex> lock(lock_);

  callbacks_.Remove(callback);
}


void SQLiteDB::InitializeNode(const string& node_id) {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("initialize_node"));
  CHECK(!node_id.empty());
  unique_lock<mutex> lock(lock_);
  string existing_id;
  if (NodeId(lock, &existing_id) != this->NOT_FOUND) {
    LOG(FATAL) << "Attempting to initialize DB beloging to node with node_id: "
               << existing_id;
  }
  sqlite::Statement statement(db_, "INSERT INTO node(node_id) VALUES(?)");
  statement.BindBlob(0, node_id);

  const int result(statement.Step());
  CHECK_EQ(SQLITE_DONE, result) << sqlite3_errmsg(db_);
}


Database::LookupResult SQLiteDB::NodeId(string* node_id) {
  unique_lock<mutex> lock(lock_);
  return NodeId(lock, CHECK_NOTNULL(node_id));
}


Database::LookupResult SQLiteDB::NodeId(const unique_lock<mutex>& lock,
                                        string* node_id) {
  ScopedLatency latency(latency_by_op_ms.GetScopedLatency("set_node_id"));
  CHECK(lock.owns_lock());
  CHECK_NOTNULL(node_id);
  sqlite::Statement statement(db_, "SELECT node_id FROM node");

  int result(statement.Step());
  if (result == SQLITE_DONE) {
    return this->NOT_FOUND;
  }
  CHECK_EQ(SQLITE_ROW, result) << sqlite3_errmsg(db_);

  statement.GetBlob(0, node_id);
  result = statement.Step();
  // There can only be one!
  CHECK_EQ(SQLITE_DONE, result) << sqlite3_errmsg(db_);
  return this->LOOKUP_OK;
}


void SQLiteDB::BeginTransaction(const unique_lock<mutex>& lock) {
  CHECK(lock.owns_lock());
  if (FLAGS_sqlite_batch_into_transactions) {
    CHECK_EQ(0, transaction_size_);
    CHECK(!in_transaction_);
    VLOG(1) << "Beginning new transaction.";
    sqlite::Statement s(db_, "BEGIN TRANSACTION");
    CHECK_EQ(SQLITE_DONE, s.Step()) << sqlite3_errmsg(db_);
    in_transaction_ = true;
  }
}


void SQLiteDB::EndTransaction(const unique_lock<mutex>& lock) {
  CHECK(lock.owns_lock());
  if (FLAGS_sqlite_batch_into_transactions) {
    CHECK(in_transaction_);
    VLOG(1) << "Committing transaction.";
    {
      sqlite::Statement s(db_, "END TRANSACTION");
      CHECK_EQ(SQLITE_DONE, s.Step()) << sqlite3_errmsg(db_);
    }
    {
      sqlite::Statement s(db_, "PRAGMA wal_checkpoint(TRUNCATE)");
      CHECK_EQ(SQLITE_ROW, s.Step()) << sqlite3_errmsg(db_);
      CHECK_EQ(SQLITE_DONE, s.Step()) << sqlite3_errmsg(db_);
    }

    transaction_size_ = 0;
    in_transaction_ = false;
  }
}


void SQLiteDB::MaybeStartNewTransaction(const unique_lock<mutex>& lock) {
  CHECK(lock.owns_lock());
  if (FLAGS_sqlite_batch_into_transactions &&
      transaction_size_ >= FLAGS_sqlite_transaction_batch_size) {
    VLOG(1) << "Rolling over into new transaction.";
    EndTransaction(lock);
    BeginTransaction(lock);
  }
  ++transaction_size_;
}


void SQLiteDB::ForceNotifySTH() {
  unique_lock<mutex> lock(lock_);

  ct::SignedTreeHead sth;
  const Database::LookupResult db_result =
      this->LatestTreeHeadNoLock(lock, &sth);
  if (db_result == Database::NOT_FOUND) {
    return;
  }

  CHECK(db_result == Database::LOOKUP_OK);

  // Do not call the callbacks while holding the lock, as they might
  // want to perform some lookups.
  lock.unlock();
  callbacks_.Call(sth);
}


Database::LookupResult SQLiteDB::LatestTreeHeadNoLock(
    const unique_lock<mutex>& lock, ct::SignedTreeHead* result) const {
  CHECK(lock.owns_lock());
  sqlite::Statement statement(db_,
                              "SELECT sth FROM trees WHERE timestamp IN "
                              "(SELECT MAX(timestamp) FROM trees)");

  int ret = statement.Step();
  if (ret == SQLITE_DONE) {
    return this->NOT_FOUND;
  }
  CHECK_EQ(SQLITE_ROW, ret) << sqlite3_errmsg(db_);

  string sth;
  statement.GetBlob(0, &sth);
  CHECK(result->ParseFromString(sth));

  return this->LOOKUP_OK;
}


}  // namespace cert_trans
