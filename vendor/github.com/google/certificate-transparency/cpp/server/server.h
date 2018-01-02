#ifndef CERT_TRANS_SERVER_SERVER_H_
#define CERT_TRANS_SERVER_SERVER_H_

#include <stdint.h>
#include <memory>
#include <string>
#include <thread>

#include "base/macros.h"
#include "log/strict_consistent_store.h"
#include "monitoring/gauge.h"
#include "util/libevent_wrapper.h"
#include "util/masterelection.h"
#include "util/sync_task.h"

class Frontend;
class LogVerifier;

namespace cert_trans {

template <class Logged>
class ClusterStateController;
// TODO(pphaneuf): Not needed?
template <class Logged>
class ConsistentStore;
class ContinuousFetcher;
class Database;
class EtcdClient;
class GCMExporter;
class LogLookup;
class LogSigner;
class LoggedEntry;
class Proxy;
class ThreadPool;
class UrlFetcher;

// Size of latest locally generated STH.
Gauge<>* latest_local_tree_size_gauge();


class Server {
 public:
  static void StaticInit();

  // Doesn't take ownership of anything.
  Server(const std::shared_ptr<libevent::Base>& event_base,
         ThreadPool* internal_pool, ThreadPool* http_pool, Database* db,
         EtcdClient* etcd_client, UrlFetcher* url_fetcher,
         const LogVerifier* log_verifier);
  ~Server();

  bool IsMaster() const;
  MasterElection* election();
  ConsistentStore<LoggedEntry>* consistent_store();
  ClusterStateController<LoggedEntry>* cluster_state_controller();
  LogLookup* log_lookup();
  ContinuousFetcher* continuous_fetcher();
  Proxy* proxy();
  libevent::HttpServer* http_server();

  void Initialise(bool is_mirror);
  void WaitForReplication() const;
  void Run();

 private:
  const std::shared_ptr<libevent::Base> event_base_;
  std::unique_ptr<libevent::EventPumpThread> event_pump_;
  libevent::HttpServer http_server_;
  Database* const db_;
  const LogVerifier* const log_verifier_;
  const std::string node_id_;
  UrlFetcher* const url_fetcher_;
  EtcdClient* const etcd_client_;
  MasterElection election_;
  ThreadPool* const internal_pool_;
  util::SyncTask server_task_;
  StrictConsistentStore<LoggedEntry> consistent_store_;
  const std::unique_ptr<Frontend> frontend_;
  std::unique_ptr<LogLookup> log_lookup_;
  std::unique_ptr<ClusterStateController<LoggedEntry>> cluster_controller_;
  std::unique_ptr<ContinuousFetcher> fetcher_;
  ThreadPool* const http_pool_;
  std::unique_ptr<Proxy> proxy_;
  std::unique_ptr<std::thread> node_refresh_thread_;
  std::unique_ptr<GCMExporter> gcm_exporter_;

  DISALLOW_COPY_AND_ASSIGN(Server);
};


}  // namespace cert_trans

#endif  // CERT_TRANS_SERVER_SERVER_H_
