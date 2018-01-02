#include "server/log_processes.h"

#include <gflags/gflags.h>
#include <iostream>

#include "monitoring/latency.h"
#include "monitoring/monitoring.h"
#include "server/metrics.h"

DEFINE_int32(tree_signing_frequency_seconds, 600,
             "How often should we issue a new signed tree head. Approximate: "
             "the signer process will kick off if in the beginning of the "
             "server select loop, at least this period has elapsed since the "
             "last signing. Set this well below the MMD to ensure we sign in "
             "a timely manner. Must be greater than 0.");
DEFINE_int32(sequencing_frequency_seconds, 10,
             "How often should new entries be sequenced. The sequencing runs "
             "in parallel with the tree signing and cleanup.");
DEFINE_int32(cleanup_frequency_seconds, 10,
             "How often should new entries be cleanedup. The cleanup runs in "
             "in parallel with the tree signing and sequencing.");

using cert_trans::Counter;
using cert_trans::Gauge;
using cert_trans::Latency;
using google::RegisterFlagValidator;
using ct::SignedTreeHead;
using std::function;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::steady_clock;

namespace {

Gauge<>* latest_local_tree_size_gauge =
    Gauge<>::New("latest_local_tree_size",
                 "Size of latest locally generated STH.");

Counter<bool>* sequencer_total_runs = Counter<bool>::New(
    "sequencer_total_runs", "successful",
    "Total number of sequencer runs broken out by success.");

Latency<milliseconds> sequencer_sequence_latency_ms(
    "sequencer_sequence_latency_ms",
    "Total time spent sequencing entries by sequencer");

Counter<bool>* signer_total_runs =
    Counter<bool>::New("signer_total_runs", "successful",
                       "Total number of signer runs broken out by success.");

Latency<milliseconds> signer_run_latency_ms("signer_run_latency_ms",
                                            "Total runtime of signer");

static bool ValidateIsPositive(const char* flagname, int value) {
  if (value <= 0) {
    std::cout << flagname << " must be greater than 0" << std::endl;
    return false;
  }
  return true;
}

static const bool sign_dummy =
    RegisterFlagValidator(&FLAGS_tree_signing_frequency_seconds,
                          &ValidateIsPositive);
}

namespace cert_trans {

void SignMerkleTree(TreeSigner<LoggedEntry>* tree_signer,
                    ConsistentStore<LoggedEntry>* store,
                    ClusterStateController<LoggedEntry>* controller) {
  CHECK_NOTNULL(tree_signer);
  CHECK_NOTNULL(store);
  CHECK_NOTNULL(controller);
  const steady_clock::duration period(
      (seconds(FLAGS_tree_signing_frequency_seconds)));
  steady_clock::time_point target_run_time(steady_clock::now());

  while (true) {
    {
      ScopedLatency signer_run_latency(
          signer_run_latency_ms.GetScopedLatency());
      const TreeSigner<LoggedEntry>::UpdateResult result(
          tree_signer->UpdateTree());
      switch (result) {
        case TreeSigner<LoggedEntry>::OK: {
          const SignedTreeHead latest_sth(tree_signer->LatestSTH());
          latest_local_tree_size_gauge->Set(latest_sth.tree_size());
          controller->NewTreeHead(latest_sth);
          signer_total_runs->Increment(true /* successful */);
          break;
        }
        case TreeSigner<LoggedEntry>::INSUFFICIENT_DATA:
          LOG(INFO) << "Can't update tree because we don't have all the "
                    << "entries locally, will try again later.";
          signer_total_runs->Increment(false /* successful */);
          break;
        default:
          LOG(FATAL) << "Error updating tree: " << result;
      }
    }

    const steady_clock::time_point now(steady_clock::now());
    while (target_run_time <= now) {
      target_run_time += period;
    }
    std::this_thread::sleep_for(target_run_time - now);
  }
}

void CleanUpEntries(ConsistentStore<LoggedEntry>* store,
                    const function<bool()>& is_master) {
  CHECK_NOTNULL(store);
  CHECK(is_master);
  const steady_clock::duration period(
      (seconds(FLAGS_cleanup_frequency_seconds)));
  steady_clock::time_point target_run_time(steady_clock::now());

  while (true) {
    if (is_master()) {
      // Keep cleaning up until there's no more work to do.
      // This should help to keep the etcd contents size down during heavy
      // load.
      while (true) {
        const util::StatusOr<int64_t> num_cleaned(store->CleanupOldEntries());
        if (!num_cleaned.ok()) {
          LOG(WARNING) << "Problem cleaning up old entries: "
                       << num_cleaned.status();
          break;
        }
        if (num_cleaned.ValueOrDie() == 0) {
          break;
        }
      }
    }

    const steady_clock::time_point now(steady_clock::now());
    while (target_run_time <= now) {
      target_run_time += period;
    }

    std::this_thread::sleep_for(target_run_time - now);
  }
}


void SequenceEntries(TreeSigner<LoggedEntry>* tree_signer,
                     const function<bool()>& is_master) {
  CHECK_NOTNULL(tree_signer);
  CHECK(is_master);
  const steady_clock::duration period(
      (seconds(FLAGS_sequencing_frequency_seconds)));
  steady_clock::time_point target_run_time(steady_clock::now());

  while (true) {
    if (is_master()) {
      const ScopedLatency sequencer_sequence_latency(
          sequencer_sequence_latency_ms.GetScopedLatency());
      util::Status status(tree_signer->SequenceNewEntries());
      if (!status.ok()) {
        LOG(WARNING) << "Problem sequencing new entries: " << status;
      }
      sequencer_total_runs->Increment(status.ok());
    }

    const steady_clock::time_point now(steady_clock::now());
    while (target_run_time <= now) {
      target_run_time += period;
    }

    std::this_thread::sleep_for(target_run_time - now);
  }
}

}  // namespace cert_trans
