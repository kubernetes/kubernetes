#include "server/server.h"

#include <gflags/gflags.h>
#include <chrono>
#include <csignal>
#include <functional>

#include "log/cluster_state_controller.h"
#include "log/etcd_consistent_store.h"
#include "log/frontend.h"
#include "log/log_lookup.h"
#include "log/log_verifier.h"
#include "monitoring/gcm/exporter.h"
#include "monitoring/monitoring.h"
#include "server/metrics.h"
#include "server/proxy.h"
#include "util/thread_pool.h"
#include "util/uuid.h"

using std::bind;
using std::chrono::seconds;
using std::chrono::steady_clock;
using std::placeholders::_1;
using std::shared_ptr;
using std::signal;
using std::string;
using std::this_thread::sleep_for;
using std::thread;

// These flags are DEFINEd in server_helper to keep the validation logic
// related to server startup options in one place.
DECLARE_string(server);
DECLARE_int32(port);
DECLARE_string(etcd_root);

DEFINE_int32(node_state_refresh_seconds, 10,
             "How often to refresh the ClusterNodeState entry for this node.");
DEFINE_int32(watchdog_seconds, 120,
             "How many seconds without successfully refreshing this node's "
             "before firing the watchdog timer.");
DEFINE_bool(watchdog_timeout_is_fatal, true,
            "Exit if the watchdog timer fires.");

namespace cert_trans {


Gauge<>* latest_local_tree_size_gauge() {
  static Gauge<>* const gauge(
      Gauge<>::New("latest_local_tree_size",
                   "Size of latest locally generated STH."));

  return gauge;
}


namespace {


void RefreshNodeState(ClusterStateController<LoggedEntry>* controller,
                      util::Task* task) {
  CHECK_NOTNULL(task);
  const steady_clock::duration period(
      (seconds(FLAGS_node_state_refresh_seconds)));
  steady_clock::time_point target_run_time(steady_clock::now());

  while (true) {
    if (task->CancelRequested()) {
      task->Return(util::Status::CANCELLED);
      alarm(0);
    }
    // If we haven't managed to refresh our state file in a timely fashion,
    // then send us a SIGALRM:
    alarm(FLAGS_watchdog_seconds);

    controller->RefreshNodeState();

    const steady_clock::time_point now(steady_clock::now());
    while (target_run_time <= now) {
      target_run_time += period;
    }
    sleep_for(target_run_time - now);
  }
}


void WatchdogTimeout(int) {
  if (FLAGS_watchdog_timeout_is_fatal) {
    LOG(FATAL) << "Watchdog timed out, killing process.";
  } else {
    LOG(INFO) << "Watchdog timeout out, ignoring.";
  }
}


string GetNodeId(Database* db) {
  string node_id;
  if (db->NodeId(&node_id) != Database::LOOKUP_OK) {
    node_id = UUID4();
    LOG(INFO) << "Initializing Node DB with UUID: " << node_id;
    db->InitializeNode(node_id);
  } else {
    LOG(INFO) << "Found DB with Node UUID: " << node_id;
  }
  return node_id;
}


}  // namespace


// static
void Server::StaticInit() {
  CHECK_NE(SIG_ERR, signal(SIGALRM, &WatchdogTimeout));
}


Server::Server(const shared_ptr<libevent::Base>& event_base,
               ThreadPool* internal_pool, ThreadPool* http_pool, Database* db,
               EtcdClient* etcd_client, UrlFetcher* url_fetcher,
               const LogVerifier* log_verifier)
    : event_base_(event_base),
      event_pump_(new libevent::EventPumpThread(event_base_)),
      http_server_(*event_base_),
      db_(CHECK_NOTNULL(db)),
      log_verifier_(CHECK_NOTNULL(log_verifier)),
      node_id_(GetNodeId(db_)),
      url_fetcher_(CHECK_NOTNULL(url_fetcher)),
      etcd_client_(CHECK_NOTNULL(etcd_client)),
      election_(event_base_, etcd_client_, FLAGS_etcd_root + "/election",
                node_id_),
      internal_pool_(CHECK_NOTNULL(internal_pool)),
      server_task_(internal_pool_),
      consistent_store_(&election_,
                        new EtcdConsistentStore<LoggedEntry>(
                            event_base_.get(), internal_pool_, etcd_client_,
                            &election_, FLAGS_etcd_root, node_id_)),
      http_pool_(CHECK_NOTNULL(http_pool)) {
  CHECK_LT(0, FLAGS_port);

  if (FLAGS_monitoring == kPrometheus) {
    http_server_.AddHandler("/metrics", ExportPrometheusMetrics);
  } else if (FLAGS_monitoring == kGcm) {
    gcm_exporter_.reset(
        new GCMExporter(FLAGS_server, url_fetcher_, internal_pool_));
  } else {
    LOG(FATAL) << "Please set --monitoring to one of the supported values.";
  }

  http_server_.Bind(nullptr, FLAGS_port);
  election_.StartElection();
}


Server::~Server() {
  server_task_.Cancel();
  node_refresh_thread_->join();
  server_task_.Wait();
}


bool Server::IsMaster() const {
  return election_.IsMaster();
}


MasterElection* Server::election() {
  return &election_;
}


ConsistentStore<LoggedEntry>* Server::consistent_store() {
  return &consistent_store_;
}


ClusterStateController<LoggedEntry>* Server::cluster_state_controller() {
  return cluster_controller_.get();
}


LogLookup* Server::log_lookup() {
  return log_lookup_.get();
}


ContinuousFetcher* Server::continuous_fetcher() {
  return fetcher_.get();
}


Proxy* Server::proxy() {
  return proxy_.get();
}


libevent::HttpServer* Server::http_server() {
  return &http_server_;
}


void Server::WaitForReplication() const {
  // If we're joining an existing cluster, this node needs to get its database
  // up-to-date with the serving_sth before we can do anything, so we'll wait
  // here for that:
  util::StatusOr<ct::SignedTreeHead> serving_sth(
      consistent_store_.GetServingSTH());
  if (serving_sth.ok()) {
    while (db_->TreeSize() < serving_sth.ValueOrDie().tree_size()) {
      LOG(WARNING) << "Waiting for local database to catch up to serving_sth ("
                   << db_->TreeSize() << " of "
                   << serving_sth.ValueOrDie().tree_size() << ")";
      sleep(1);
    }
  }
}


void Server::Initialise(bool is_mirror) {
  fetcher_.reset(ContinuousFetcher::New(event_base_.get(), internal_pool_, db_,
                                        log_verifier_, !is_mirror)
                     .release());

  log_lookup_.reset(new LogLookup(db_));

  cluster_controller_.reset(new ClusterStateController<LoggedEntry>(
      internal_pool_, event_base_, url_fetcher_, db_, &consistent_store_,
      &election_, fetcher_.get()));

  // Publish this node's hostname:port info
  cluster_controller_->SetNodeHostPort(FLAGS_server, FLAGS_port);
  {
    ct::SignedTreeHead db_sth;
    if (db_->LatestTreeHead(&db_sth) == Database::LOOKUP_OK) {
      const LogVerifier::LogVerifyResult sth_verify_result(
          log_verifier_->VerifySignedTreeHead(db_sth));
      if (sth_verify_result != LogVerifier::VERIFY_OK) {
        LOG(FATAL) << "STH retrieved from DB did not verify: "
                   << LogVerifier::VerifyResultString(sth_verify_result);
      }
      cluster_controller_->NewTreeHead(db_sth);
    }
  }

  node_refresh_thread_.reset(new thread(&RefreshNodeState,
                                        cluster_controller_.get(),
                                        server_task_.task()));

  proxy_.reset(
      new Proxy(event_base_.get(),
                bind(&ClusterStateController<LoggedEntry>::GetFreshNodes,
                     cluster_controller_.get()),
                url_fetcher_, http_pool_));
}


void Server::Run() {
  // Ding the temporary event pump because we're about to enter the event loop
  event_pump_.reset();
  event_base_->Dispatch();
}


}  // namespace cert_trans
