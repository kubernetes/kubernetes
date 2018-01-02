#ifndef CERT_TRANS_LOG_STRICT_CONSISTENT_STORE_INL_H_
#define CERT_TRANS_LOG_STRICT_CONSISTENT_STORE_INL_H_

#include "log/strict_consistent_store.h"

namespace cert_trans {

template <class Logged>
StrictConsistentStore<Logged>::StrictConsistentStore(
    const MasterElection* election, ConsistentStore<Logged>* peer)
    : election_(CHECK_NOTNULL(election)), peer_(CHECK_NOTNULL(peer)) {
}


template <class Logged>
util::StatusOr<int64_t>
StrictConsistentStore<Logged>::NextAvailableSequenceNumber() const {
  if (!election_->IsMaster()) {
    return util::Status(util::error::PERMISSION_DENIED,
                        "Not currently master.");
  }
  return peer_->NextAvailableSequenceNumber();
}


template <class Logged>
util::Status StrictConsistentStore<Logged>::SetServingSTH(
    const ct::SignedTreeHead& new_sth) {
  if (!election_->IsMaster()) {
    return util::Status(util::error::PERMISSION_DENIED,
                        "Not currently master.");
  }
  return peer_->SetServingSTH(new_sth);
}


template <class Logged>
util::Status StrictConsistentStore<Logged>::UpdateSequenceMapping(
    EntryHandle<ct::SequenceMapping>* entry) {
  if (!election_->IsMaster()) {
    return util::Status(util::error::PERMISSION_DENIED,
                        "Not currently master.");
  }
  return peer_->UpdateSequenceMapping(entry);
}


template <class Logged>
util::Status StrictConsistentStore<Logged>::SetClusterConfig(
    const ct::ClusterConfig& config) {
  if (!election_->IsMaster()) {
    return util::Status(util::error::PERMISSION_DENIED,
                        "Not currently master.");
  }
  return peer_->SetClusterConfig(config);
}


template <class Logged>
util::StatusOr<int64_t> StrictConsistentStore<Logged>::CleanupOldEntries() {
  if (!election_->IsMaster()) {
    return util::Status(util::error::PERMISSION_DENIED,
                        "Not currently master.");
  }
  return peer_->CleanupOldEntries();
}


}  // namespace cert_trans


#endif  // CERT_TRANS_LOG_STRICT_CONSISTENT_STORE_INL_H_
