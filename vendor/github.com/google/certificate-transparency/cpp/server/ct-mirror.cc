/* -*- indent-tabs-mode: nil -*- */

#include <event2/buffer.h>
#include <event2/thread.h>
#include <gflags/gflags.h>
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <signal.h>
#include <unistd.h>
#include <chrono>
#include <csignal>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "client/async_log_client.h"
#include "config.h"
#include "fetcher/continuous_fetcher.h"
#include "fetcher/peer_group.h"
#include "fetcher/remote_peer.h"
#include "log/cluster_state_controller.h"
#include "log/ct_extensions.h"
#include "log/database.h"
#include "log/etcd_consistent_store.h"
#include "log/log_lookup.h"
#include "log/strict_consistent_store.h"
#include "merkletree/compact_merkle_tree.h"
#include "merkletree/merkle_verifier.h"
#include "monitoring/latency.h"
#include "monitoring/monitoring.h"
#include "monitoring/registry.h"
#include "proto/cert_serializer.h"
#include "server/certificate_handler.h"
#include "server/json_output.h"
#include "server/metrics.h"
#include "server/proxy.h"
#include "server/server.h"
#include "server/server_helper.h"
#include "util/etcd.h"
#include "util/init.h"
#include "util/libevent_wrapper.h"
#include "util/masterelection.h"
#include "util/periodic_closure.h"
#include "util/read_key.h"
#include "util/status.h"
#include "util/thread_pool.h"
#include "util/util.h"
#include "util/uuid.h"

DEFINE_int32(log_stats_frequency_seconds, 3600,
             "Interval for logging summary statistics. Approximate: the "
             "server will log statistics if in the beginning of its select "
             "loop, at least this period has elapsed since the last log time. "
             "Must be greater than 0.");
DEFINE_int32(target_poll_frequency_seconds, 10,
             "How often should the target log be polled for updates.");
DEFINE_int32(num_http_server_threads, 16,
             "Number of threads for servicing the incoming HTTP requests.");
DEFINE_string(target_log_uri, "http://ct.googleapis.com/pilot",
              "URI of the log to mirror.");
DEFINE_string(
    target_public_key, "",
    "PEM-encoded server public key file of the log we're mirroring.");
DEFINE_int32(local_sth_update_frequency_seconds, 30,
             "Number of seconds between local checks for updated tree data.");

namespace libevent = cert_trans::libevent;

using cert_trans::AsyncLogClient;
using cert_trans::CertificateHttpHandler;
using cert_trans::ClusterStateController;
using cert_trans::ConsistentStore;
using cert_trans::ContinuousFetcher;
using cert_trans::Counter;
using cert_trans::Database;
using cert_trans::EtcdClient;
using cert_trans::EtcdConsistentStore;
using cert_trans::Gauge;
using cert_trans::HttpHandler;
using cert_trans::Latency;
using cert_trans::LogLookup;
using cert_trans::LoggedEntry;
using cert_trans::MasterElection;
using cert_trans::PeriodicClosure;
using cert_trans::Proxy;
using cert_trans::ReadPublicKey;
using cert_trans::RemotePeer;
using cert_trans::ScopedLatency;
using cert_trans::Server;
using cert_trans::StalenessTracker;
using cert_trans::StrictConsistentStore;
using cert_trans::ThreadPool;
using cert_trans::Update;
using cert_trans::UrlFetcher;
using ct::ClusterNodeState;
using ct::SignedTreeHead;
using google::RegisterFlagValidator;
using std::bind;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::steady_clock;
using std::function;
using std::lock_guard;
using std::make_pair;
using std::make_shared;
using std::map;
using std::mutex;
using std::placeholders::_1;
using std::shared_ptr;
using std::string;
using std::thread;
using std::unique_ptr;
using util::HexString;
using util::StatusOr;
using util::SyncTask;
using util::Task;


namespace {


Gauge<>* latest_local_tree_size_gauge =
    Gauge<>::New("latest_local_tree_size",
                 "Size of latest locally available STH.");

Counter<>* inconsistent_sths_received =
    Counter<>::New("inconsistent_sths_received",
                   "Number of STHs received from the mirror target whose root "
                   "hash does not match the locally built tree.");


// Basic sanity checks on flag values.
static bool ValidateRead(const char* flagname, const string& path) {
  if (access(path.c_str(), R_OK) != 0) {
    std::cout << "Cannot access " << flagname << " at " << path << std::endl;
    return false;
  }
  return true;
}

static const bool pubkey_dummy =
    RegisterFlagValidator(&FLAGS_target_public_key, &ValidateRead);

static bool ValidateIsPositive(const char* flagname, int value) {
  if (value <= 0) {
    std::cout << flagname << " must be greater than 0" << std::endl;
    return false;
  }
  return true;
}

static const bool follow_dummy =
    RegisterFlagValidator(&FLAGS_target_poll_frequency_seconds,
                          &ValidateIsPositive);
}  // namespace


void STHUpdater(Database* db,
                ClusterStateController<LoggedEntry>* cluster_state_controller,
                mutex* queue_mutex, map<int64_t, ct::SignedTreeHead>* queue,
                LogLookup* log_lookup, Task* task) {
  CHECK_NOTNULL(db);
  CHECK_NOTNULL(cluster_state_controller);
  CHECK_NOTNULL(queue_mutex);
  CHECK_NOTNULL(queue);
  CHECK_NOTNULL(task);
  CHECK_NOTNULL(log_lookup);

  while (true) {
    if (task->CancelRequested()) {
      task->Return(util::Status::CANCELLED);
    }

    const int64_t local_size(db->TreeSize());
    latest_local_tree_size_gauge->Set(local_size);

    // log_lookup doesn't yet have the data for the new STHs integrated (that
    // happens via a callback when the WriteTreeHead() method is called on the
    // DB), so we'll used a compact tree to pre-validate the STH roots.
    //
    // We'll start with one based on the current state of our serving tree and
    // update it to the STH sizes we're checking.
    unique_ptr<CompactMerkleTree> new_tree(
        log_lookup->GetCompactMerkleTree(new Sha256Hasher));

    {
      lock_guard<mutex> lock(*queue_mutex);
      unique_ptr<Database::Iterator> entries(
          db->ScanEntries(new_tree->LeafCount()));
      while (!queue->empty() &&
             queue->begin()->second.tree_size() <= local_size) {
        const SignedTreeHead next_sth(queue->begin()->second);
        queue->erase(queue->begin());

        // First, if necessary, catch our local compact tree up to the
        // candidate STH size:
        {
          LoggedEntry entry;
          CHECK_LE(next_sth.tree_size(), local_size);
          CHECK_GE(next_sth.tree_size(), 0);
          const uint64_t next_sth_tree_size(
              static_cast<uint64_t>(next_sth.tree_size()));
          while (new_tree->LeafCount() < next_sth_tree_size) {
            CHECK(entries->GetNextEntry(&entry));
            CHECK(entry.has_sequence_number());
            CHECK_GE(entry.sequence_number(), 0);
            const uint64_t entry_sequence_number(
                static_cast<uint64_t>(entry.sequence_number()));
            CHECK_EQ(new_tree->LeafCount(), entry_sequence_number);
            string serialized_leaf;
            CHECK(entry.SerializeForLeaf(&serialized_leaf));
            CHECK_EQ(entry_sequence_number + 1,
                     new_tree->AddLeaf(serialized_leaf));
          }
        }

        // If the candidate STH is historical, use the RootAtSnapshot() from
        // our serving tree, otherwise use the root we just calculated with our
        // compact tree.
        const string local_root_at_snapshot(
            next_sth.tree_size() > log_lookup->GetSTH().tree_size()
                ? new_tree->CurrentRoot()
                : log_lookup->RootAtSnapshot(next_sth.tree_size()));

        if (next_sth.sha256_root_hash() != local_root_at_snapshot) {
          LOG(WARNING) << "Received STH:\n" << next_sth.DebugString()
                       << " whose root:\n"
                       << HexString(next_sth.sha256_root_hash())
                       << "\ndoes not match that of local tree at "
                       << "corresponding snapshot:\n"
                       << HexString(local_root_at_snapshot);
          inconsistent_sths_received->Increment();
          // TODO(alcutter): We should probably write these bad STHs out to a
          // separate DB table for later analysis.
          continue;
        }
        LOG(INFO) << "Can serve new STH of size " << next_sth.tree_size()
                  << " locally";
        cluster_state_controller->NewTreeHead(next_sth);
      }
    }

    std::this_thread::sleep_for(
        seconds(FLAGS_local_sth_update_frequency_seconds));
  }
}


int main(int argc, char* argv[]) {
  // Ignore various signals whilst we start up.
  signal(SIGHUP, SIG_IGN);
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  util::InitCT(&argc, &argv);
  ConfigureSerializerForV1CT();

  Server::StaticInit();

  cert_trans::EnsureValidatorsRegistered();
  const unique_ptr<Database> db(cert_trans::ProvideDatabase());
  CHECK(db) << "No database instance created, check flag settings";

  const bool stand_alone_mode(cert_trans::IsStandalone(false));
  const shared_ptr<libevent::Base> event_base(make_shared<libevent::Base>());
  ThreadPool internal_pool(8);
  UrlFetcher url_fetcher(event_base.get(), &internal_pool);

  const unique_ptr<EtcdClient> etcd_client(
      cert_trans::ProvideEtcdClient(event_base.get(), &internal_pool,
                                    &url_fetcher));

  CHECK(!FLAGS_target_public_key.empty());
  const StatusOr<EVP_PKEY*> pubkey(ReadPublicKey(FLAGS_target_public_key));
  CHECK(pubkey.ok()) << "Failed to read target log's public key file: "
                     << pubkey.status();

  const LogVerifier log_verifier(new LogSigVerifier(pubkey.ValueOrDie()),
                                 new MerkleVerifier(new Sha256Hasher));

  ThreadPool http_pool(FLAGS_num_http_server_threads);

  Server server(event_base, &internal_pool, &http_pool, db.get(),
                etcd_client.get(), &url_fetcher, &log_verifier);
  server.Initialise(true /* is_mirror */);

  unique_ptr<StalenessTracker> staleness_tracker(
      new StalenessTracker(server.cluster_state_controller(), &internal_pool,
                           event_base.get()));

  CertificateHttpHandler handler(server.log_lookup(), db.get(),
                                 server.cluster_state_controller(),
                                 nullptr /* checker */, nullptr /* Frontend */,
                                 &internal_pool, event_base.get(),
                                 staleness_tracker.get());

  // Connect the handler, proxy and server together
  handler.SetProxy(server.proxy());
  handler.Add(server.http_server());

  if (stand_alone_mode) {
    // Set up a simple single-node mirror environment for testing.
    //
    // Put a sensible single-node config into FakeEtcd. For a real clustered
    // log
    // we'd expect a ClusterConfig already to be present within etcd as part of
    // the provisioning of the log.
    //
    // TODO(alcutter): Note that we're currently broken wrt to restarting the
    // log server when there's data in the log.  It's a temporary thing though,
    // so fear ye not.
    ct::ClusterConfig config;
    config.set_minimum_serving_nodes(1);
    config.set_minimum_serving_fraction(1);
    LOG(INFO) << "Setting default single-node ClusterConfig:\n"
              << config.DebugString();
    server.consistent_store()->SetClusterConfig(config);

    // Since we're a single node cluster, we'll settle that we're the
    // master here, so that we can populate the initial STH
    // (StrictConsistentStore won't allow us to do so unless we're master.)
    server.election()->StartElection();
    server.election()->WaitToBecomeMaster();
  }

  CHECK(!FLAGS_target_log_uri.empty());

  ThreadPool pool(16);
  SyncTask fetcher_task(&pool);

  mutex queue_mutex;
  map<int64_t, ct::SignedTreeHead> queue;

  const function<void(const ct::SignedTreeHead&)> new_sth(
      [&queue_mutex, &queue](const ct::SignedTreeHead& sth) {
        lock_guard<mutex> lock(queue_mutex);
        const auto it(queue.find(sth.tree_size()));
        if (it != queue.end() && sth.timestamp() < it->second.timestamp()) {
          LOG(WARNING) << "Received older STH:\nHad:\n"
                       << it->second.DebugString() << "\nGot:\n"
                       << sth.DebugString();
          return;
        }
        queue.insert(make_pair(sth.tree_size(), sth));
      });

  const shared_ptr<RemotePeer> peer(make_shared<RemotePeer>(
      unique_ptr<AsyncLogClient>(
          new AsyncLogClient(&pool, &url_fetcher, FLAGS_target_log_uri)),
      unique_ptr<LogVerifier>(
          new LogVerifier(new LogSigVerifier(pubkey.ValueOrDie()),
                          new MerkleVerifier(new Sha256Hasher))),
      new_sth, fetcher_task.task()->AddChild(
                   [](Task*) { LOG(INFO) << "RemotePeer exited."; })));

  server.continuous_fetcher()->AddPeer("target", peer);

  server.WaitForReplication();

  thread sth_updater(&STHUpdater, db.get(), server.cluster_state_controller(),
                     &queue_mutex, &queue, server.log_lookup(),
                     fetcher_task.task()->AddChild(
                         [](Task*) { LOG(INFO) << "STHUpdater exited."; }));

  server.Run();

  fetcher_task.task()->Return();
  fetcher_task.Wait();
  sth_updater.join();

  return 0;
}
