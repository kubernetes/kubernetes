#ifndef MONITOR_SQLITE_DB_H
#define MONITOR_SQLITE_DB_H

#include <stdint.h>
#include <string>

#include "base/macros.h"
#include "monitor/database.h"

struct sqlite3;

namespace monitor {

class SQLiteDB : public Database {
 public:
  explicit SQLiteDB(const std::string& dbfile);

  ~SQLiteDB();

  typedef Database::WriteResult WriteResult;
  typedef Database::LookupResult LookupResult;
  typedef Database::VerificationLevel VerificationLevel;

  void BeginTransaction();

  void EndTransaction();

  virtual LookupResult LookupLatestWrittenSTH(
      ct::SignedTreeHead* result) const;

  virtual LookupResult LookupHashByIndex(int64_t sequence_number,
                                         std::string* result) const;

  virtual LookupResult LookupSTHByTimestamp(uint64_t timestamp,
                                            ct::SignedTreeHead* result) const;

  virtual LookupResult LookupVerificationLevel(
      const ct::SignedTreeHead& sth, VerificationLevel* result) const;

 private:
  virtual WriteResult CreateEntry_(const std::string& leaf,
                                   const std::string& leaf_hash,
                                   const std::string& cert,
                                   const std::string& cert_chain);

  virtual WriteResult WriteSTH_(uint64_t timestamp, int64_t tree_size,
                                const std::string& sth);

  virtual WriteResult SetVerificationLevel_(const ct::SignedTreeHead& sth,
                                            VerificationLevel verify_level);

  sqlite3* db_;

  DISALLOW_COPY_AND_ASSIGN(SQLiteDB);
};

}  // namespace monitor

#endif  // MONITOR_SQLITE_DB_H
