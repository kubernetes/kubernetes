/* -*- mode: c++; indent-tabs-mode: nil -*- */
#ifndef CERTIFICATE_DB_H
#define CERTIFICATE_DB_H

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
#include "util/statusor.h"

namespace cert_trans {

class FileStorage;


// Database interface that stores certificates and tree head
// signatures in the filesystem.
class FileDB : public Database {
 public:
  // Reference implementation: reads the entire database on boot
  // and builds an in-memory index.
  // Writes to the underlying FileStorage are atomic (assuming underlying
  // file system operations such as 'rename' are atomic) which should
  // guarantee full recoverability from crashes/power failures.
  // The tree head database uses 6-byte primary keys corresponding to the
  // 6 lower bytes of the (unique) timestamp, so the storage depth of
  // the FileDB should be set up accordingly. For example, a storage depth
  // of 8 buckets tree head updates within about 1 minute
  // (timestamps xxxxxxxx0000 - xxxxxxxxFFFF) to the same directory.
  // Takes ownership of |cert_storage|, |tree_storage|, and |meta_storage|.
  FileDB(FileStorage* cert_storage, FileStorage* tree_storage,
         FileStorage* meta_storage);
  ~FileDB();

  static const size_t kTimestampBytesIndexed;

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

  const std::unique_ptr<FileStorage> cert_storage_;
  // Store all tree heads, but currently only support looking up the latest
  // one.
  // Other necessary lookup indices (by tree size, by timestamp range?) TBD.
  const std::unique_ptr<FileStorage> tree_storage_;

  const std::unique_ptr<FileStorage> meta_storage_;

  mutable std::mutex lock_;

  int64_t contiguous_size_;
  std::unordered_map<std::string, int64_t> id_by_hash_;

  // This is a mapping of the non-contiguous entries of the log (which
  // can happen while it is being fetched). When entries here become
  // contiguous with the head of the tree they'll be removed.
  std::set<int64_t> sparse_entries_;

  uint64_t latest_tree_timestamp_;
  // The same as a string;
  std::string latest_timestamp_key_;
  DatabaseNotifierHelper callbacks_;

  DISALLOW_COPY_AND_ASSIGN(FileDB);
};


}  // namespace cert_trans

#endif
