#ifndef CERT_TRANS_LOG_STRICT_CONSISTENT_STORE_H_
#define CERT_TRANS_LOG_STRICT_CONSISTENT_STORE_H_

#include "log/consistent_store.h"
#include "util/masterelection.h"

namespace cert_trans {

// A wrapper around a ConsistentStore which will not allow changes to
// the cluster state which should only be performed by the current master
// unless this node /is/ the current master.
//
// Note that while this is better than just gating the start of a high-level
// action (especially a long running action, e.g. a signing run) with a check
// to IsMaster(), it is still necessarily racy because etcd doesn't support
// atomic updates across keys.)
template <class Logged>
class StrictConsistentStore : public ConsistentStore<Logged> {
 public:
  // Takes ownership of |peer|, but not |election|
  StrictConsistentStore(const MasterElection* election,
                        ConsistentStore<Logged>* peer);

  virtual ~StrictConsistentStore() = default;

  // Methods requiring that the caller is currently master:

  util::StatusOr<int64_t> NextAvailableSequenceNumber() const override;

  util::Status SetServingSTH(const ct::SignedTreeHead& new_sth) override;

  util::Status UpdateSequenceMapping(
      EntryHandle<ct::SequenceMapping>* entry) override;

  util::Status SetClusterConfig(const ct::ClusterConfig& config) override;

  util::StatusOr<int64_t> CleanupOldEntries() override;

  // Other methods:

  util::StatusOr<ct::SignedTreeHead> GetServingSTH() const override {
    return peer_->GetServingSTH();
  }

  util::Status AddPendingEntry(Logged* entry) override {
    return peer_->AddPendingEntry(entry);
  }

  util::Status GetPendingEntryForHash(
      const std::string& hash, EntryHandle<Logged>* entry) const override {
    return peer_->GetPendingEntryForHash(hash, entry);
  }

  util::Status GetPendingEntries(
      std::vector<EntryHandle<Logged>>* entries) const override {
    return peer_->GetPendingEntries(entries);
  }

  util::Status GetSequenceMapping(
      EntryHandle<ct::SequenceMapping>* entry) const override {
    return peer_->GetSequenceMapping(entry);
  }

  util::StatusOr<ct::ClusterNodeState> GetClusterNodeState() const override {
    return peer_->GetClusterNodeState();
  }

  util::Status SetClusterNodeState(
      const ct::ClusterNodeState& state) override {
    return peer_->SetClusterNodeState(state);
  }

  void WatchServingSTH(
      const typename ConsistentStore<Logged>::ServingSTHCallback& cb,
      util::Task* task) override {
    return peer_->WatchServingSTH(cb, task);
  }

  void WatchClusterNodeStates(
      const typename ConsistentStore<Logged>::ClusterNodeStateCallback& cb,
      util::Task* task) override {
    return peer_->WatchClusterNodeStates(cb, task);
  }

  void WatchClusterConfig(
      const typename ConsistentStore<Logged>::ClusterConfigCallback& cb,
      util::Task* task) override {
    return peer_->WatchClusterConfig(cb, task);
  }

 private:
  const MasterElection* const election_;  // Not owned by us
  const std::unique_ptr<ConsistentStore<Logged>> peer_;
};


}  // namespace

#endif  // CERT_TRANS_LOG_STRICT_CONSISTENT_STORE_H_
