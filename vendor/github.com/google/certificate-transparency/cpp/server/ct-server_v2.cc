/* -*- indent-tabs-mode: nil -*- */

#include <gflags/gflags.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>
#include <string>

#include "config.h"
#include "log/cert_checker.h"
#include "log/cert_submission_handler.h"
#include "log/cluster_state_controller.h"
#include "log/etcd_consistent_store.h"
#include "log/frontend.h"
#include "log/frontend_signer.h"
#include "log/log_lookup.h"
#include "log/log_signer.h"
#include "log/log_verifier.h"
#include "log/strict_consistent_store.h"
#include "log/tree_signer.h"
#include "merkletree/merkle_verifier.h"
#include "proto/cert_serializer.h"
#include "server/certificate_handler_v2.h"
#include "server/log_processes.h"
#include "server/server.h"
#include "server/server_helper.h"
#include "server/staleness_tracker.h"
#include "util/etcd.h"
#include "util/init.h"
#include "util/libevent_wrapper.h"
#include "util/read_key.h"
#include "util/status.h"
#include "util/uuid.h"

DEFINE_string(key, "", "PEM-encoded server private key file");
DEFINE_string(trusted_cert_file, "",
              "File for trusted CA certificates, in concatenated PEM format");
DEFINE_double(guard_window_seconds, 60,
              "Unsequenced entries newer than this "
              "number of seconds will not be sequenced.");
DEFINE_int32(num_http_server_threads, 16,
             "Number of threads for servicing the incoming HTTP requests.");
// TODO(mhs): Remove this flag when V2 is complete
DEFINE_bool(i_know_v2_is_not_finished_yet, false,
            "Set this to allow V2 server startup as the functionality "
            "is not complete yet");

namespace libevent = cert_trans::libevent;

using cert_trans::CertChecker;
using cert_trans::CertificateHttpHandlerV2;
using cert_trans::CleanUpEntries;
using cert_trans::ClusterStateController;
using cert_trans::ConsistentStore;
using cert_trans::Database;
using cert_trans::EtcdClient;
using cert_trans::EtcdConsistentStore;
using cert_trans::LoggedEntry;
using cert_trans::ReadPrivateKey;
using cert_trans::SequenceEntries;
using cert_trans::Server;
using cert_trans::SignMerkleTree;
using cert_trans::StalenessTracker;
using cert_trans::ThreadPool;
using cert_trans::TreeSigner;
using cert_trans::UrlFetcher;
using ct::ClusterNodeState;
using ct::SignedTreeHead;
using google::RegisterFlagValidator;
using std::bind;
using std::function;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::thread;
using std::unique_ptr;


namespace {

// Basic sanity checks on flag values.
static bool ValidateRead(const char* flagname, const string& path) {
  if (access(path.c_str(), R_OK) != 0) {
    std::cout << "Cannot access " << flagname << " at " << path << std::endl;
    return false;
  }
  return true;
}

static const bool key_dummy = RegisterFlagValidator(&FLAGS_key, &ValidateRead);

static const bool cert_dummy =
    RegisterFlagValidator(&FLAGS_trusted_cert_file, &ValidateRead);

}  // namespace


int main(int argc, char* argv[]) {
  // Ignore various signals whilst we start up.
  signal(SIGHUP, SIG_IGN);
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  ConfigureSerializerForV2CT();
  util::InitCT(&argc, &argv);

  if (!FLAGS_i_know_v2_is_not_finished_yet) {
    LOG(FATAL)
        << "You must set --i_know_v2_is_not_finished_yet to run a v2 server";
  }

  Server::StaticInit();

  util::StatusOr<EVP_PKEY*> pkey(ReadPrivateKey(FLAGS_key));
  CHECK_EQ(pkey.status(), util::Status::OK);
  LogSigner log_signer(pkey.ValueOrDie());

  CertChecker checker;
  CHECK(checker.LoadTrustedCertificates(FLAGS_trusted_cert_file))
      << "Could not load CA certs from " << FLAGS_trusted_cert_file;

  cert_trans::EnsureValidatorsRegistered();
  const unique_ptr<Database> db(cert_trans::ProvideDatabase());
  CHECK(db) << "No database instance created, check flag settings";

  shared_ptr<libevent::Base> event_base(make_shared<libevent::Base>());
  ThreadPool internal_pool(8);
  UrlFetcher url_fetcher(event_base.get(), &internal_pool);

  const bool stand_alone_mode(cert_trans::IsStandalone(true));
  LOG(INFO) << "Running in "
            << (stand_alone_mode ? "STAND-ALONE" : "CLUSTERED") << " mode.";

  unique_ptr<EtcdClient> etcd_client(
      cert_trans::ProvideEtcdClient(event_base.get(), &internal_pool,
                                    &url_fetcher));

  const LogVerifier log_verifier(new LogSigVerifier(pkey.ValueOrDie()),
                                 new MerkleVerifier(new Sha256Hasher));

  ThreadPool http_pool(FLAGS_num_http_server_threads);

  Server server(event_base, &internal_pool, &http_pool, db.get(),
                etcd_client.get(), &url_fetcher, &log_verifier);
  server.Initialise(false /* is_mirror */);

  Frontend frontend(
      new FrontendSigner(db.get(), server.consistent_store(), &log_signer));
  unique_ptr<StalenessTracker> staleness_tracker(
      new StalenessTracker(server.cluster_state_controller(), &internal_pool,
                           event_base.get()));
  CertificateHttpHandlerV2 handler(server.log_lookup(), db.get(),
                                   server.cluster_state_controller(), &checker,
                                   &frontend, &internal_pool, event_base.get(),
                                   staleness_tracker.get());

  // Connect the handler, proxy and server together
  handler.SetProxy(server.proxy());
  handler.Add(server.http_server());

  TreeSigner<LoggedEntry> tree_signer(
      std::chrono::duration<double>(FLAGS_guard_window_seconds), db.get(),
      server.log_lookup()->GetCompactMerkleTree(new Sha256Hasher),
      server.consistent_store(), &log_signer);

  if (stand_alone_mode) {
    // Set up a simple single-node environment.
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

    {
      EtcdClient::Response resp;
      util::SyncTask task(event_base.get());
      etcd_client->Create("/root/sequence_mapping", "", &resp, task.task());
      task.Wait();
      CHECK_EQ(util::Status::OK, task.status());
    }

    // Do an initial signing run to get the initial STH, again this is
    // temporary until we re-populate FakeEtcd from the DB.
    CHECK_EQ(tree_signer.UpdateTree(), TreeSigner<LoggedEntry>::OK);

    // Need to boot-strap the Serving STH too because we consider it an error
    // if it's not set, which in turn causes us to not attempt to become
    // master:
    server.consistent_store()->SetServingSTH(tree_signer.LatestSTH());
  }

  server.WaitForReplication();

  // TODO(pphaneuf): We should be remaining in an "unhealthy state"
  // (either not accepting any requests, or returning some internal
  // server error) until we have an STH to serve.
  const function<bool()> is_master(bind(&Server::IsMaster, &server));
  thread sequencer(&SequenceEntries, &tree_signer, is_master);
  thread cleanup(&CleanUpEntries, server.consistent_store(), is_master);
  thread signer(&SignMerkleTree, &tree_signer, server.consistent_store(),
                server.cluster_state_controller());

  server.Run();

  return 0;
}
