#ifndef CERT_TRANS_UTIL_MASTERELECTION_H_
#define CERT_TRANS_UTIL_MASTERELECTION_H_

#include <stdint.h>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "util/etcd.h"
#include "util/sync_task.h"


namespace cert_trans {

class PeriodicClosure;

// Implements a simple MasterElection scheme.
//
// Candidates participate by creating a proposal at |my_proposal_path_|, this
// will have an associated creation index (stored in my_proposal_create_index_)
// assigned by etcd.
//
// In order for a participant to become master, firstly it must have the
// proposal with the lowest creation index (this favours more stable
// participants), and secondly all other participants must agree that it should
// be master by updating their proposal files to contain the path of the
// winning candidate's proposal file.  This provides some protection against
// there being multiple masters due to some classes of bug or network issues.
//
// Proposals are created with a TTL, after which, if they've not had their TTL
// updated in the meantime, they will be automatically deleted by etcd.
// This helps to detect failed candidates and clear up after them.
// In order to keep this from happening to live candidates, each instance
// maintains a periodic callback whose sole job is to update the TTL on its
// proposal file.
//
// TODO(alcutter): Some enhancements:
//   - Recover gracefully from a crash where an old proposal exists for this
//     node (e.g. recover and continue, or delete it, or wait, ...)
class MasterElection {
 public:
  // No transfer of ownership.
  MasterElection(const std::shared_ptr<libevent::Base>& base,
                 EtcdClient* client, const std::string& lock_dir,
                 const std::string& node_id);

  virtual ~MasterElection();

  // Signals that we want to participate in the election.
  virtual void StartElection();

  // Signals that we no longer want to participate in the election.
  // Should never be called from the libevent thread.
  virtual void StopElection();

  // Blocks until this instance has become master or the election has been
  // stopped.
  // Returns true iff we're master at the time the call returns.
  virtual bool WaitToBecomeMaster() const;

  // Returns true iff this instance is currently master at the time of the
  // call.
  virtual bool IsMaster() const;

 protected:
  MasterElection();

 private:
  // The states that the proposal can be in:
  enum class ProposalState {
    NONE,               // There is no proposal
    AWAITING_CREATION,  // We want to create a proposal
    CREATING,           // We're in the process of creating the proposal
    UP_TO_DATE,         // The proposal is current
    AWAITING_UPDATE,    // We'd like to update the proposal
    UPDATING,           // We're updating the proposal
    AWAITING_DELETE,    // We'd like to delete the proposal
    DELETING,           // We're deleting the proposal
  };

  // Performs a transition between the current proposal state, and |to|.
  // Checks that the requested transition is valid.
  void Transition(const std::unique_lock<std::mutex>& lock,
                  const ProposalState to);

  // Creates the proposal.
  // This should only be called on the base_ event thread.
  void CreateProposal();

  // Called once our proposal file has been created.
  void ProposalCreateDone(EtcdClient::Response* resp, util::Task* task);

  // Will call UpdateProposal iff there is currently no other proposal update
  // in-flight.  |backed| should contain the id of the proposal this node is
  // backing.
  bool MaybeUpdateProposal(const std::unique_lock<std::mutex>& lock,
                           const std::string& backed);

  // Updates this node's proposal.  |backed| should contain the id of the
  // proposal this node is backing.
  // This should only be called on the base_ event thread.
  void UpdateProposal(const std::string& backed);

  // Called when our proposal file has been refreshed by the KeepAlive thread.
  void ProposalUpdateDone(EtcdClient::Response* resp, util::Task* task);

  // Deletes this node's proposal.
  // This should only be called on the base_ event thread.
  void DeleteProposal();

  // Called once our proposal file has been deleted from the proposal
  // directory.
  void ProposalDeleteDone(util::Task* task);

  // Thread entry point for the periodic callback to refresh the proposal TTL.
  void ProposalKeepAliveCallback();

  // Updates our local view of the election proposals.
  void UpdateProposalView(const std::vector<EtcdClient::Node>& updates);

  // Works out which proposal /should/ be master based on created_index_.
  // Returns true iff there was an apparent master, false otherwise.
  bool DetermineApparentMaster(EtcdClient::Node* apparent_master) const;

  // Called by the EtcdClient whenever there's been a change in one or
  // more of the proposal files.
  void OnProposalUpdate(const std::vector<EtcdClient::Node>& updates);

  // Internal non-locking accessor for is_master_
  bool IsMaster(const std::unique_lock<std::mutex>& lock) const;

  const std::shared_ptr<libevent::Base> base_;
  EtcdClient* const client_;  // Not owned by us.
  const std::string proposal_dir_;
  const std::string my_proposal_path_;

  mutable std::mutex mutex_;
  ProposalState proposal_state_;
  mutable std::condition_variable proposal_state_cv_;

  // Any thread wanting to know whether we're master should wait on this CV.
  mutable std::condition_variable is_master_cv_;
  bool running_;

  std::unique_ptr<PeriodicClosure> proposal_refresh_callback_;
  std::unique_ptr<util::SyncTask> proposal_watch_;

  // Our local copy of the proposals
  std::map<std::string, EtcdClient::Node> proposals_;

  int64_t my_proposal_create_index_;
  int64_t my_proposal_modified_index_;

  std::string backed_proposal_;

  bool is_master_;
  EtcdClient::Node current_master_;

  friend class ElectionTest;
  friend std::ostream& operator<<(std::ostream& output, ProposalState state);
};


}  // namespace cert_trans

#endif  // CERT_TRANS_UTIL_MASTERELECTION_H_
