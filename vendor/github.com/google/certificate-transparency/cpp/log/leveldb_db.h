#ifndef CERTIFICATE_LEVELDB_DB_H
#define CERTIFICATE_LEVELDB_DB_H

#include "config.h"

#include <leveldb/db.h>
#ifdef HAVE_LEVELDB_FILTER_POLICY_H
#include <leveldb/filter_policy.h>
#endif
#include <stdint.h>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "base/macros.h"
#include "log/database.h"
#include "proto/ct.pb.h"

namespace cert_trans {


class LevelDB : public Database {
 public:
  static const size_t kTimestampBytesIndexed;

  explicit LevelDB(const std::string& dbfile);
  ~LevelDB() = default;

  // Implement abstract functions, see database.h for comments.
  Database::WriteResult CreateSequencedEntry_(
      const LoggedEntry& logged) override;

  Database::LookupResult LookupByHash(const std::string& hash,
                                      LoggedEntry* result) const override;

  Database::LookupResult LookupByIndex(int64_t sequence_number,
                                       LoggedEntry* result) const override;

  std::unique_ptr<Database::Iterator> ScanEntries(
      int64_t start_index) const override;

  Database::WriteResult WriteTreeHead_(const ct::SignedTreeHead& sth) override;

  Database::LookupResult LatestTreeHead(
      ct::SignedTreeHead* result) const override;

  int64_t TreeSize() const override;

  void AddNotifySTHCallback(
      const Database::NotifySTHCallback* callback) override;

  void RemoveNotifySTHCallback(
      const Database::NotifySTHCallback* callback) override;

  void InitializeNode(const std::string& node_id) override;

  Database::LookupResult NodeId(std::string* node_id) override;

 private:
  class Iterator;

  void BuildIndex();
  Database::LookupResult LatestTreeHeadNoLock(
      ct::SignedTreeHead* result) const;
  void InsertEntryMapping(int64_t sequence_number, const std::string& hash);

  mutable std::mutex lock_;
#ifdef HAVE_LEVELDB_FILTER_POLICY_H
  // filter_policy_ must be valid for at least as long as db_ is, so
  // keep this order.
  const std::unique_ptr<const leveldb::FilterPolicy> filter_policy_;
#endif
  std::unique_ptr<leveldb::DB> db_;

  int64_t contiguous_size_;
  std::unordered_map<std::string, int64_t> id_by_hash_;

  // This is a mapping of the non-contiguous entries of the log (which
  // can happen while it is being fetched). When entries here become
  // contiguous with the beginning of the tree, they are removed.
  std::set<int64_t> sparse_entries_;

  uint64_t latest_tree_timestamp_;
  std::string latest_timestamp_key_;
  cert_trans::DatabaseNotifierHelper callbacks_;

  DISALLOW_COPY_AND_ASSIGN(LevelDB);
};


}  // namespace cert_trans

#endif  // CERTIFICATE_LEVELDB_DB_H
