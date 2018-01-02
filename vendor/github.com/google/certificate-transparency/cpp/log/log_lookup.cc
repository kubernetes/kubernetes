#include "log/log_lookup.h"

#include <glog/logging.h>
#include <stdint.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "base/time_support.h"
#include "merkletree/merkle_tree.h"
#include "merkletree/serial_hasher.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"
#include "util/util.h"

using ct::MerkleAuditProof;
using ct::ShortMerkleAuditProof;
using ct::SignedTreeHead;
using std::bind;
using std::lock_guard;
using std::make_pair;
using std::map;
using std::mutex;
using std::placeholders::_1;
using std::string;
using std::unique_lock;
using std::unique_ptr;
using std::vector;
using util::HexString;

namespace cert_trans {


static const int kCtimeBufSize = 26;


LogLookup::LogLookup(ReadOnlyDatabase* db)
    : db_(CHECK_NOTNULL(db)),
      cert_tree_(new Sha256Hasher),
      latest_tree_head_(),
      update_from_sth_cb_(bind(&LogLookup::UpdateFromSTH, this, _1)) {
  db_->AddNotifySTHCallback(&update_from_sth_cb_);
}


LogLookup::~LogLookup() {
  db_->RemoveNotifySTHCallback(&update_from_sth_cb_);
}


void LogLookup::UpdateFromSTH(const SignedTreeHead& sth) {
  lock_guard<mutex> lock(lock_);

  CHECK_EQ(ct::V1, sth.version())
      << "Tree head signed with an unknown version";

  if (sth.timestamp() == latest_tree_head_.timestamp())
    return;

  CHECK_LE(0, sth.tree_size());
  if (sth.timestamp() <= latest_tree_head_.timestamp() ||
      static_cast<uint64_t>(sth.tree_size()) < cert_tree_.LeafCount()) {
    LOG(WARNING) << "Database replied with an STH that is older than ours: "
                 << "Our STH:\n" << latest_tree_head_.DebugString()
                 << "Database STH:\n" << sth.DebugString();
    return;
  }

  // Record the new hashes: append all of them, die on any error.
  // TODO(ekasper): make tree signer write leaves out to the database,
  // so that we don't have to read the entries in.
  string leaf_hash;
  auto it(db_->ScanEntries(cert_tree_.LeafCount()));
  // LeafCount() is potentially unsigned here but as this is using memory
  // the count can never get close to overflow in 64 bits.
  CHECK_LE(cert_tree_.LeafCount(), static_cast<uint64_t>(INT64_MAX));

  for (int64_t sequence_number = cert_tree_.LeafCount();
       sequence_number < sth.tree_size(); ++sequence_number) {
    LoggedEntry logged;
    // TODO(ekasper): perhaps some of these errors can/should be
    // handled more gracefully. E.g. we could retry a failed update
    // a number of times -- but until we know under which conditions
    // the database might fail (database busy?), just die.
    CHECK(it->GetNextEntry(&logged))
        << "Latest STH has " << sth.tree_size() << "entries but we failed to "
        << "retrieve entry number " << sequence_number;
    CHECK(logged.has_sequence_number())
        << "Logged entry has no sequence number";
    CHECK_EQ(sequence_number, logged.sequence_number());

    leaf_hash = LeafHash(logged);
    // TODO(ekasper): plug in the log public key so that we can verify the STH.
    CHECK_EQ(static_cast<size_t>(sequence_number + 1),
             cert_tree_.AddLeafHash(leaf_hash));
    // Duplicate leaves shouldn't really happen but are not a problem either:
    // we just return the Merkle proof of the first occurrence.
    leaf_index_.insert(make_pair(leaf_hash, sequence_number));
  }
  CHECK_EQ(HexString(cert_tree_.CurrentRoot()),
           HexString(sth.sha256_root_hash()))
      << "Computed root hash and stored STH root hash do not match";
  LOG(INFO) << "Found " << sth.tree_size() - latest_tree_head_.tree_size()
            << " new log entries";
  latest_tree_head_.CopyFrom(sth);

  const time_t last_update(static_cast<time_t>(latest_tree_head_.timestamp() /
                                               kNumMillisPerSecond));
  char buf[kCtimeBufSize];
  LOG(INFO) << "Tree successfully updated at " << ctime_r(&last_update, buf);
}


LogLookup::LookupResult LogLookup::GetIndex(const string& merkle_leaf_hash,
                                            int64_t* index) {
  unique_lock<mutex> lock(lock_);
  const int64_t myindex(GetIndexInternal(lock, merkle_leaf_hash));

  if (myindex < 0) {
    return NOT_FOUND;
  } else {
    *index = myindex;
    return OK;
  }
}


// Look up by SHA256-hash of the certificate.
LogLookup::LookupResult LogLookup::AuditProof(const string& merkle_leaf_hash,
                                              MerkleAuditProof* proof) {
  unique_lock<mutex> lock(lock_);

  const int64_t leaf_index(GetIndexInternal(lock, merkle_leaf_hash));
  if (leaf_index < 0) {
    return NOT_FOUND;
  }

  CHECK_GE(leaf_index, 0);
  proof->set_version(ct::V1);
  proof->set_tree_size(cert_tree_.LeafCount());
  proof->set_timestamp(latest_tree_head_.timestamp());
  proof->set_leaf_index(leaf_index);

  proof->clear_path_node();
  vector<string> audit_path = cert_tree_.PathToCurrentRoot(leaf_index + 1);
  for (size_t i = 0; i < audit_path.size(); ++i)
    proof->add_path_node(audit_path[i]);

  proof->mutable_id()->CopyFrom(latest_tree_head_.id());
  proof->mutable_tree_head_signature()->CopyFrom(
      latest_tree_head_.signature());
  return OK;
}


LogLookup::LookupResult LogLookup::AuditProof(int64_t leaf_index,
                                              size_t tree_size,
                                              ShortMerkleAuditProof* proof) {
  lock_guard<mutex> lock(lock_);

  proof->set_leaf_index(leaf_index);

  proof->clear_path_node();
  vector<string> audit_path =
      cert_tree_.PathToRootAtSnapshot(leaf_index + 1, tree_size);
  for (size_t i = 0; i < audit_path.size(); ++i)
    proof->add_path_node(audit_path[i]);

  return OK;
}


// Look up by SHA256-hash of the certificate and tree size.
LogLookup::LookupResult LogLookup::AuditProof(const string& merkle_leaf_hash,
                                              size_t tree_size,
                                              ShortMerkleAuditProof* proof) {
  int64_t leaf_index;
  if (GetIndex(merkle_leaf_hash, &leaf_index) != OK)
    return NOT_FOUND;

  CHECK_GE(leaf_index, 0);
  return AuditProof(leaf_index, tree_size, proof);
}


string LogLookup::RootAtSnapshot(size_t tree_size) {
  lock_guard<mutex> lock(lock_);
  return cert_tree_.RootAtSnapshot(tree_size);
}


string LogLookup::LeafHash(const LoggedEntry& logged) const {
  string serialized_leaf;
  CHECK(logged.SerializeForLeaf(&serialized_leaf));
  // We do not need to take the lock for this call into cert_tree_, as
  // this is merely a const forwarder (to another const, thread-safe
  // method).
  return cert_tree_.LeafHash(serialized_leaf);
}


unique_ptr<CompactMerkleTree> LogLookup::GetCompactMerkleTree(
    SerialHasher* hasher) {
  lock_guard<mutex> lock(lock_);
  return unique_ptr<CompactMerkleTree>(
      new CompactMerkleTree(cert_tree_, hasher));
}


int64_t LogLookup::GetIndexInternal(const unique_lock<mutex>& lock,
                                    const string& merkle_leaf_hash) const {
  CHECK(lock.owns_lock());

  const map<string, int64_t>::const_iterator it(
      leaf_index_.find(merkle_leaf_hash));
  if (it == leaf_index_.end())
    return -1;

  CHECK_GE(it->second, 0);
  return it->second;
}


}  // namespace cert_trans
