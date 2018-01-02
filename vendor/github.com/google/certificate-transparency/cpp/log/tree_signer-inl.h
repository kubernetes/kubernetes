/* -*- indent-tabs-mode: nil -*- */
#ifndef CERT_TRANS_LOG_TREE_SIGNER_INL_H_
#define CERT_TRANS_LOG_TREE_SIGNER_INL_H_

#include "log/tree_signer.h"

#include <glog/logging.h>
#include <stdint.h>
#include <algorithm>
#include <chrono>
#include <set>
#include <unordered_map>

#include "log/database.h"
#include "log/log_signer.h"
#include "proto/serializer.h"
#include "util/status.h"
#include "util/util.h"


namespace cert_trans {


namespace {


bool LessThanBySequence(const ct::SequenceMapping::Mapping& lhs,
                        const ct::SequenceMapping::Mapping& rhs) {
  CHECK(lhs.has_sequence_number());
  CHECK(rhs.has_sequence_number());
  return lhs.sequence_number() < rhs.sequence_number();
}


}  // namespace


// Comparator for ordering pending hashes.
// Order by timestamp then hash.
template <class Logged>
struct PendingEntriesOrder
    : std::binary_function<const cert_trans::EntryHandle<Logged>&,
                           const cert_trans::EntryHandle<Logged>&, bool> {
  bool operator()(const cert_trans::EntryHandle<Logged>& x,
                  const cert_trans::EntryHandle<Logged>& y) const {
    CHECK(x.Entry().contents().sct().has_timestamp());
    CHECK(y.Entry().contents().sct().has_timestamp());
    const uint64_t x_time(x.Entry().contents().sct().timestamp());
    const uint64_t y_time(y.Entry().contents().sct().timestamp());
    if (x_time < y_time) {
      return true;
    } else if (x_time > y_time) {
      return false;
    }

    // Fallback to Hash as a final tie-breaker:
    return x.Entry().Hash() < y.Entry().Hash();
  }
};


template <class Logged>
TreeSigner<Logged>::TreeSigner(
    const std::chrono::duration<double>& guard_window, Database* db,
    std::unique_ptr<CompactMerkleTree> merkle_tree,
    cert_trans::ConsistentStore<Logged>* consistent_store, LogSigner* signer)
    : guard_window_(guard_window),
      db_(db),
      consistent_store_(consistent_store),
      signer_(signer),
      cert_tree_(std::move(merkle_tree)),
      latest_tree_head_() {
  CHECK(cert_tree_);
  // Try to get any STH previously published by this node.
  const util::StatusOr<ct::ClusterNodeState> node_state(
      consistent_store_->GetClusterNodeState());
  CHECK(node_state.ok() ||
        node_state.status().CanonicalCode() == util::error::NOT_FOUND)
      << "Problem fetching this node's previous state: "
      << node_state.status();
  if (node_state.ok()) {
    latest_tree_head_ = node_state.ValueOrDie().newest_sth();
  }
}


template <class Logged>
uint64_t TreeSigner<Logged>::LastUpdateTime() const {
  return latest_tree_head_.timestamp();
}


template <class Logged>
util::Status TreeSigner<Logged>::SequenceNewEntries() {
  const std::chrono::system_clock::time_point now(
      std::chrono::system_clock::now());
  util::StatusOr<int64_t> status_or_sequence_number(
      consistent_store_->NextAvailableSequenceNumber());
  if (!status_or_sequence_number.ok()) {
    return status_or_sequence_number.status();
  }
  int64_t next_sequence_number(status_or_sequence_number.ValueOrDie());
  CHECK_GE(next_sequence_number, 0);
  VLOG(1) << "Next available sequence number: " << next_sequence_number;

  EntryHandle<ct::SequenceMapping> mapping;
  util::Status status(consistent_store_->GetSequenceMapping(&mapping));
  if (!status.ok()) {
    return status;
  }

  // Hashes which are already sequenced.
  std::unordered_map<std::string, std::pair<int64_t, bool /*present*/>>
      sequenced_hashes;
  for (const auto& m : mapping.Entry().mapping()) {
    // Go home clang-format, you're drunk.
    CHECK(
        sequenced_hashes
            .insert(std::make_pair(m.entry_hash(),
                                   std::make_pair(m.sequence_number(), false)))
            .second);
  }

  std::vector<cert_trans::EntryHandle<Logged>> pending_entries;
  status = consistent_store_->GetPendingEntries(&pending_entries);
  if (!status.ok()) {
    return status;
  }
  std::sort(pending_entries.begin(), pending_entries.end(),
            PendingEntriesOrder<Logged>());

  VLOG(1) << "Sequencing " << pending_entries.size() << " entr"
          << (pending_entries.size() == 1 ? "y" : "ies");

  // We're going to update the sequence mapping based on the following rules:
  // 1) existing sequence mappings whose corresponding PendingEntry still
  //    exists will remain in the mappings file.
  // 2) PendingEntries which do not have a corresponding sequence mapping will
  //    gain one.
  // 3) mappings whose corresponding PendingEntry no longer exists will be
  //    removed from the sequence mapping file.
  google::protobuf::RepeatedPtrField<ct::SequenceMapping_Mapping> new_mapping;
  std::map<int64_t, const Logged*> seq_to_entry;
  int num_sequenced(0);
  for (auto& pending_entry : pending_entries) {
    const std::string& pending_hash(pending_entry.Entry().Hash());
    const std::chrono::system_clock::time_point cert_time(
        std::chrono::milliseconds(pending_entry.Entry().timestamp()));
    if (now - cert_time < guard_window_) {
      VLOG(1) << "Entry too recent: "
              << util::ToBase64(pending_entry.Entry().Hash());
      continue;
    }
    const auto seq_it(sequenced_hashes.find(pending_hash));
    ct::SequenceMapping::Mapping* const seq_mapping(new_mapping.Add());

    if (seq_it == sequenced_hashes.end()) {
      // Need to sequence this one.
      VLOG(1) << util::ToBase64(pending_hash) << " = " << next_sequence_number;

      // Record the sequence -> hash mapping
      seq_mapping->set_sequence_number(next_sequence_number);
      seq_mapping->set_entry_hash(pending_entry.Entry().Hash());
      pending_entry.MutableEntry()->set_sequence_number(next_sequence_number);
      ++num_sequenced;
      ++next_sequence_number;
    } else {
      VLOG(1) << "Previously sequenced " << util::ToBase64(pending_hash)
              << " = " << seq_it->second.first;
      CHECK(!seq_it->second.second /*present*/)
          << "Saw same sequenced cert twice.";
      CHECK(!pending_entry.Entry().has_sequence_number());
      seq_it->second.second = true;  // present

      seq_mapping->set_entry_hash(seq_it->first);
      seq_mapping->set_sequence_number(seq_it->second.first);
      pending_entry.MutableEntry()->set_sequence_number(seq_it->second.first);
    }
    CHECK(
        seq_to_entry.insert(
                        std::make_pair(pending_entry.Entry().sequence_number(),
                                       pending_entry.MutableEntry()))
            .second);
  }

  const util::StatusOr<ct::SignedTreeHead> serving_sth(
      consistent_store_->GetServingSTH());
  if (!serving_sth.ok()) {
    LOG(WARNING) << "Failed to get ServingSTH: " << serving_sth.status();
    return serving_sth.status();
  }

  // Sanity check: make sure no hashes above the serving_sth level vanished:
  CHECK_LE(serving_sth.ValueOrDie().tree_size(), INT64_MAX);
  const int64_t serving_tree_size(serving_sth.ValueOrDie().tree_size());
  for (const auto& s : sequenced_hashes) {
    if (!s.second.second /*present*/) {
      // if it disappeared, check it's underwater:
      CHECK_LT(s.second.first, serving_tree_size);
    }
  }

  if (new_mapping.size() > 0) {
    sort(new_mapping.begin(), new_mapping.end(), LessThanBySequence);
    CHECK_LE(new_mapping.Get(0).sequence_number(), serving_tree_size);
  }

  // Update the mapping proto with our new mappings
  mapping.MutableEntry()->mutable_mapping()->Swap(&new_mapping);

  // Store updated sequence->hash mappings in the consistent store
  status = consistent_store_->UpdateSequenceMapping(&mapping);
  if (!status.ok()) {
    return status;
  }

  // Now add the sequenced entries to our local DB so that the local signer can
  // incorporate them.
  for (auto it(seq_to_entry.find(db_->TreeSize())); it != seq_to_entry.end();
       ++it) {
    VLOG(1) << "Adding to local DB: " << it->first;
    CHECK_EQ(it->first, it->second->sequence_number());
    CHECK_EQ(Database::OK, db_->CreateSequencedEntry(*(it->second)));
  }

  VLOG(1) << "Sequenced " << num_sequenced << " entries.";

  return util::Status::OK;
}


// DB_ERROR: the database is inconsistent with our inner self.
// However, if the database itself is giving inconsistent answers, or failing
// reads/writes, then we die.
template <class Logged>
typename TreeSigner<Logged>::UpdateResult TreeSigner<Logged>::UpdateTree() {
  // Try to make local timestamps unique, but there's always a chance that
  // multiple nodes in the cluster may make STHs with the same timestamp.
  // That'll get handled by the Serving STH selection code.
  uint64_t min_timestamp = LastUpdateTime() + 1;

  // Add any newly sequenced entries from our local DB.
  auto it(db_->ScanEntries(cert_tree_->LeafCount()));
  for (int64_t i(cert_tree_->LeafCount());; ++i) {
    Logged logged;
    if (!it->GetNextEntry(&logged) || logged.sequence_number() != i) {
      break;
    }
    CHECK_EQ(logged.sequence_number(), i);
    AppendToTree(logged);
    min_timestamp = std::max(min_timestamp, logged.sct().timestamp());
  }
  int64_t next_seq(cert_tree_->LeafCount());
  CHECK_GE(next_seq, 0);

  // Our tree is consistent with the database, i.e., each leaf in the tree has
  // a matching sequence number in the database (at least assuming overwriting
  // the sequence number is not allowed).
  ct::SignedTreeHead new_sth;
  TimestampAndSign(min_timestamp, &new_sth);

  // We don't actually store this STH anywhere durable yet, but rather let the
  // caller decide what to do with it.  (In practice, this will mean that it's
  // pushed out to this node's ClusterNodeState so that it becomes a candidate
  // for the cluster-wide Serving STH.)
  latest_tree_head_.CopyFrom(new_sth);
  return OK;
}


template <class Logged>
bool TreeSigner<Logged>::Append(const Logged& logged) {
  // Serialize for inclusion in the tree.
  std::string serialized_leaf;
  CHECK(logged.SerializeForLeaf(&serialized_leaf));

  CHECK_LE(cert_tree_->LeafCount(), static_cast<uint64_t>(INT64_MAX));
  CHECK_EQ(logged.sequence_number(),
           static_cast<int64_t>(cert_tree_->LeafCount()));
  // Commit the sequence number of this certificate locally
  Database::WriteResult db_result = db_->CreateSequencedEntry(logged);

  if (db_result != Database::OK) {
    CHECK_EQ(Database::SEQUENCE_NUMBER_ALREADY_IN_USE, db_result);
    LOG(ERROR) << "Attempt to assign duplicate sequence number "
               << cert_tree_->LeafCount();
    return false;
  }

  // Update in-memory tree.
  cert_tree_->AddLeaf(serialized_leaf);
  return true;
}


template <class Logged>
void TreeSigner<Logged>::AppendToTree(const Logged& logged) {
  // Serialize for inclusion in the tree.
  std::string serialized_leaf;
  CHECK(logged.SerializeForLeaf(&serialized_leaf));

  // Update in-memory tree.
  cert_tree_->AddLeaf(serialized_leaf);
}


template <class Logged>
void TreeSigner<Logged>::TimestampAndSign(uint64_t min_timestamp,
                                          ct::SignedTreeHead* sth) {
  sth->set_version(ct::V1);
  sth->set_sha256_root_hash(cert_tree_->CurrentRoot());
  uint64_t timestamp = util::TimeInMilliseconds();
  if (timestamp < min_timestamp)
    // TODO(ekasper): shouldn't really happen if everyone's clocks are in sync;
    // log a warning if the skew is over some threshold?
    timestamp = min_timestamp;
  sth->set_timestamp(timestamp);
  sth->set_tree_size(cert_tree_->LeafCount());
  LogSigner::SignResult ret = signer_->SignTreeHead(sth);
  if (ret != LogSigner::OK)
    // Make this one a hard fail. There is really no excuse for it.
    abort();
}


}  // namespace cert_trans


#endif  // CERT_TRANS_LOG_TREE_SIGNER_INL_H_
