/* -*- mode: c++; indent-tabs-mode: nil -*- */
#ifndef LOG_LOOKUP_H
#define LOG_LOOKUP_H

#include <stdint.h>
#include <map>
#include <mutex>
#include <string>

#include "base/macros.h"
#include "log/database.h"
#include "merkletree/compact_merkle_tree.h"
#include "merkletree/merkle_tree.h"
#include "proto/ct.pb.h"

namespace cert_trans {


// Lookups into the database. Read-only, so could also be a mirror.
// Keeps the entire Merkle Tree in memory to serve audit proofs.
class LogLookup {
 public:
  // The constructor loads the content from the database.
  explicit LogLookup(ReadOnlyDatabase* db);
  ~LogLookup();

  enum LookupResult {
    OK,
    NOT_FOUND,
  };

  LookupResult GetIndex(const std::string& merkle_leaf_hash, int64_t* index);

  // Look up by hash of the logged item.
  // TODO(pphaneuf): Looking up an audit proof without a tree size is
  // unreliable in the case of multiple CT servers (some might be
  // behind). New code should avoid this variant.
  LookupResult AuditProof(const std::string& merkle_leaf_hash,
                          ct::MerkleAuditProof* proof);

  // Look up by index of the logged item and tree_size.
  LookupResult AuditProof(int64_t index, size_t tree_size,
                          ct::ShortMerkleAuditProof* proof);

  // Look up by hash of the logged item and tree_size.
  LookupResult AuditProof(const std::string& merkle_leaf_hash,
                          size_t tree_size, ct::ShortMerkleAuditProof* proof);

  // Get a consitency proof between two tree heads
  std::vector<std::string> ConsistencyProof(size_t first, size_t second) {
    std::lock_guard<std::mutex> lock(lock_);
    return cert_tree_.SnapshotConsistency(first, second);
  }

  const ct::SignedTreeHead& GetSTH() const {
    std::lock_guard<std::mutex> lock(lock_);
    return latest_tree_head_;
  }

  std::string RootAtSnapshot(size_t tree_size);

  std::string LeafHash(const LoggedEntry& logged) const;

  // Creates a CompactMerkleTree based on the current state of our MerkleTree.
  // Takes ownership of |hasher|.
  std::unique_ptr<CompactMerkleTree> GetCompactMerkleTree(
      SerialHasher* hasher);

 private:
  void UpdateFromSTH(const ct::SignedTreeHead& sth);
  int64_t GetIndexInternal(const std::unique_lock<std::mutex>& lock,
                           const std::string& merkle_leaf_hash) const;

  mutable std::mutex lock_;
  // We keep a hash -> index mapping in memory so that we can quickly serve
  // Merkle proofs without having to query the database at all.
  // Note that 32 bytes is an overkill and we can optimize this to use
  // a shorter prefix (possibly with a multimap).
  std::map<std::string, int64_t> leaf_index_;

  ReadOnlyDatabase* const db_;
  MerkleTree cert_tree_;
  ct::SignedTreeHead latest_tree_head_;

  const Database::NotifySTHCallback update_from_sth_cb_;

  DISALLOW_COPY_AND_ASSIGN(LogLookup);
};


}  // namespace cert_trans

#endif
