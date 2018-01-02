#ifndef CERT_TRANS_LOG_CLUSTER_STATE_CONTROLLER_H_
#define CERT_TRANS_LOG_CLUSTER_STATE_CONTROLLER_H_

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <string>

#include "fetcher/continuous_fetcher.h"
#include "log/etcd_consistent_store.h"
#include "proto/ct.pb.h"
#include "util/libevent_wrapper.h"
#include "util/masterelection.h"
#include "util/statusor.h"

namespace cert_trans {

class Database;


// A class which updates & maintains the states of the individual cluster
// member nodes, and uses this information to determine the overall serving
// state of the cluster.
//
// In particular, this class:
//  - calculates the optimal STH for the cluster to serve at any given time.
//  - determines whether this node is eligible to participare in the election,
//    and leaves/joins the election as appropriate.
template <class Logged>
class ClusterStateController {
 public:
  ClusterStateController(util::Executor* executor,
                         const std::shared_ptr<libevent::Base>& base,
                         UrlFetcher* url_fetcher, Database* database,
                         ConsistentStore<Logged>* store,
                         MasterElection* election, ContinuousFetcher* fetcher);

  ~ClusterStateController();

  // Updates *this* node's ClusterNodeState to reflect the new STH available.
  void NewTreeHead(const ct::SignedTreeHead& sth);

  // Gets the current (if any) calculated serving STH for the cluster.
  // If there is such an STH then return true and |sth| is populated, returns
  // false otherwise.
  //
  // Note that this simply returns this node's interpretation of the optimum
  // serving STH, the current master/contents of the servingSTH file may
  // differ.
  //
  // Really only intended for testing.
  util::StatusOr<ct::SignedTreeHead> GetCalculatedServingSTH() const;

  void GetLocalNodeState(ct::ClusterNodeState* state) const;

  // Publishes this node's listening address in its ClusterNodeState, so that
  // other nodes can request entries from its database.
  void SetNodeHostPort(const std::string& host, const uint16_t port);

  void RefreshNodeState();

  bool NodeIsStale() const;

  // Returns a vector of the other nodes in the cluster which are able to serve
  // the cluster's current ServingSTH. Does not include this node in the
  // returned list regardless of its freshness.
  std::vector<ct::ClusterNodeState> GetFreshNodes() const;

 private:
  class ClusterPeer;

  // Updates the representation of *this* node's state in the consistent store.
  void PushLocalNodeState(const std::unique_lock<std::mutex>& lock);

  // Entry point for the watcher callback.
  // Called whenever a node changes its node state.
  void OnClusterStateUpdated(
      const std::vector<Update<ct::ClusterNodeState>>& updates);

  // Entry point for the config watcher callback.
  // Called whenever the ClusterConfig is changed.
  void OnClusterConfigUpdated(const Update<ct::ClusterConfig>& update);

  // Entry point for the config watcher callback.
  // Called whenever the ClusterConfig is changed.
  void OnServingSthUpdated(const Update<ct::SignedTreeHead>& update);

  // Calculates the STH which should be served by the cluster, given the
  // current state of the nodes.
  // If this node is the cluster master then the calculated serving STH is
  // pushed out to the consistent store.
  void CalculateServingSTH(const std::unique_lock<std::mutex>& lock);

  // Determines whether this node should be participating in the election based
  // on the current node's state.
  void DetermineElectionParticipation(
      const std::unique_lock<std::mutex>& lock);

  // Thread entry point for ServingSTH updater thread.
  void ClusterServingSTHUpdater();

  const std::shared_ptr<libevent::Base> base_;
  UrlFetcher* const url_fetcher_;         // Not owned by us
  Database* const database_;              // Not owned by us
  ConsistentStore<Logged>* const store_;  // Not owned by us
  MasterElection* const election_;        // Not owned by us
  ContinuousFetcher* const fetcher_;      // Not owned by us
  util::SyncTask watch_config_task_;
  util::SyncTask watch_node_states_task_;
  util::SyncTask watch_serving_sth_task_;
  ct::ClusterConfig cluster_config_;

  mutable std::mutex mutex_;  // covers the members below:
  ct::ClusterNodeState local_node_state_;
  std::map<std::string, const std::shared_ptr<ClusterPeer>> all_peers_;
  std::unique_ptr<ct::SignedTreeHead> calculated_serving_sth_;
  std::unique_ptr<ct::SignedTreeHead> actual_serving_sth_;
  bool exiting_;
  bool update_required_;
  std::condition_variable update_required_cv_;
  std::thread cluster_serving_sth_update_thread_;

  friend class ClusterStateControllerTest;

  DISALLOW_COPY_AND_ASSIGN(ClusterStateController);
};


}  // namespace cert_trans

#endif  // CERT_TRANS_LOG_CLUSTER_STATE_CONTROLLER_H__
