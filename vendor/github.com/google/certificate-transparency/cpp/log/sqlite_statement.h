/* -*- mode: c++; indent-tabs-mode: nil -*- */

#ifndef SQLITE_STATEMENT_H
#define SQLITE_STATEMENT_H

#include <glog/logging.h>
#include <sqlite3.h>
#include <string>

#include "base/macros.h"

namespace sqlite {

// Reduce the ugliness of the sqlite3 API.
class Statement {
 public:
  Statement(sqlite3* db, const char* sql) : stmt_(NULL) {
    int ret = sqlite3_prepare_v2(db, sql, -1, &stmt_, NULL);
    if (ret != SQLITE_OK)
      LOG(ERROR) << "ret = " << ret << ", err = " << sqlite3_errmsg(db)
                 << ", sql = " << sql << std::endl;

    CHECK_EQ(SQLITE_OK, ret);
  }

  ~Statement() {
    int ret = sqlite3_finalize(stmt_);
    // can get SQLITE_CONSTRAINT if an insert failed due to a duplicate key.
    CHECK(ret == SQLITE_OK || ret == SQLITE_CONSTRAINT);
  }

  // Fields start at 0! |value| must have lifetime that covers its
  // use, which is up until the SQL statement finishes executing
  // (i.e. after the last Step()).
  void BindBlob(unsigned field, const std::string& value) {
    CHECK_EQ(SQLITE_OK, sqlite3_bind_blob(stmt_, field + 1, value.data(),
                                          value.length(), NULL));
  }

  void BindUInt64(unsigned field, sqlite3_uint64 value) {
    CHECK_EQ(SQLITE_OK, sqlite3_bind_int64(stmt_, field + 1, value));
  }

  void GetBlob(unsigned column, std::string* value) {
    const void* data = sqlite3_column_blob(stmt_, column);
    CHECK_NOTNULL(data);
    value->assign(static_cast<const char*>(data),
                  sqlite3_column_bytes(stmt_, column));
  }

  sqlite3_uint64 GetUInt64(unsigned column) {
    return sqlite3_column_int64(stmt_, column);
  }

  int GetType(unsigned column) {
    return sqlite3_column_type(stmt_, column);
  }

  int Step() {
    return sqlite3_step(stmt_);
  }

 private:
  sqlite3_stmt* stmt_;

  DISALLOW_COPY_AND_ASSIGN(Statement);
};

}  // namespace sqlite

#endif  // SQLITE_STATEMENT_H
