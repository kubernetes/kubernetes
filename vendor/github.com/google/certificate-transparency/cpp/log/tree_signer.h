#ifndef TREE_SIGNER_H
#define TREE_SIGNER_H

#include <stdint.h>
#include <chrono>

#include "log/cluster_state_controller.h"
#include "log/consistent_store.h"
#include "merkletree/compact_merkle_tree.h"
#include "proto/ct.pb.h"


namespace util {
class Status;
}  // namespace util

class LogSigner;


namespace cert_trans {

class Database;


// Signer for appending new entries to the log.
// This is the single authority that assigns sequence numbers to new entries,
// timestamps and signs tree heads. The signer process assumes there are
// no other signers during its lifetime -- when it discovers the database has
// received tree updates it has not written, it does not try to recover,
// but rather reports an error.
template <class Logged>
class TreeSigner {
 public:
  // No transfer of ownership for params other than merkle_tree whose contents
  // is moved into this object.
  TreeSigner(const std::chrono::duration<double>& guard_window, Database* db,
             std::unique_ptr<CompactMerkleTree> merkle_tree,
             cert_trans::ConsistentStore<Logged>* consistent_store,
             LogSigner* signer);

  enum UpdateResult {
    OK,
    // The database is inconsistent with our view.
    DB_ERROR,
    // We don't have fresh enough data locally to sign.
    INSUFFICIENT_DATA,
  };

  // Latest Tree Head timestamp;
  uint64_t LastUpdateTime() const;

  util::Status SequenceNewEntries();

  // Simplest update mechanism: take all pending entries and append
  // (in random order) to the tree. Checks that the update it writes
  // to the database is consistent with the latest STH.
  UpdateResult UpdateTree();

  // Latest Tree Head (does not build a new tree, just retrieves the
  // result of the most recent build).
  const ct::SignedTreeHead& LatestSTH() const {
    return latest_tree_head_;
  }

 private:
  bool Append(const Logged& logged);
  void AppendToTree(const Logged& logged_cert);
  void TimestampAndSign(uint64_t min_timestamp, ct::SignedTreeHead* sth);

  const std::chrono::duration<double> guard_window_;
  Database* const db_;
  cert_trans::ConsistentStore<Logged>* const consistent_store_;
  LogSigner* const signer_;
  const std::unique_ptr<CompactMerkleTree> cert_tree_;
  ct::SignedTreeHead latest_tree_head_;

  template <class T>
  friend class TreeSignerTest;
};


}  // namespace cert_trans

#endif
