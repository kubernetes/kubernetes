#include "fetcher/fetcher.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <memory>
#include <mutex>

#include "base/macros.h"
#include "log/log_verifier.h"
#include "monitoring/monitoring.h"

using cert_trans::AsyncLogClient;
using cert_trans::LoggedEntry;
using cert_trans::PeerGroup;
using std::bind;
using std::lock_guard;
using std::move;
using std::mutex;
using std::placeholders::_1;
using std::string;
using std::to_string;
using std::unique_lock;
using std::unique_ptr;
using std::vector;
using util::Status;
using util::Task;
using util::TaskHold;

DEFINE_int32(fetcher_concurrent_fetches, 2,
             "number of concurrent fetch requests");
DEFINE_int32(fetcher_batch_size, 1000,
             "maximum number of entries to fetch per request");

namespace cert_trans {

Counter<string>* num_invalid_entries_fetched =
    Counter<string>::New("num_invalid_entries_fetched", "reason",
                         "Number of invalid entries fetched from remote peers "
                         "broken down by reason.");


namespace {


struct Range {
  enum State {
    HAVE,
    FETCHING,
    WANT,
  };

  Range(State state, int64_t size, unique_ptr<Range> next = nullptr)
      : state_(state), size_(size), next_(move(next)) {
    CHECK(state_ == HAVE || state_ == FETCHING || state_ == WANT);
    CHECK_GT(size_, 0);
  };

  State state_;
  int64_t size_;
  unique_ptr<Range> next_;
};


struct FetchState {
  FetchState(Database* db, unique_ptr<PeerGroup> peer_group,
             const LogVerifier* log_verifier, Task* task);

  void WalkEntries();
  void FetchRange(const unique_lock<mutex>& lock, Range* current,
                  int64_t index, Task* range_task);
  void WriteToDatabase(int64_t index, Range* range,
                       const vector<AsyncLogClient::Entry>* retval,
                       Task* range_task, Task* fetch_task);

  Database* const db_;
  const unique_ptr<PeerGroup> peer_group_;
  const LogVerifier* const log_verifier_;
  Task* const task_;

  mutex lock_;
  int64_t start_;
  unique_ptr<Range> entries_;

 private:
  DISALLOW_COPY_AND_ASSIGN(FetchState);
};


FetchState::FetchState(Database* db, unique_ptr<PeerGroup> peer_group,
                       const LogVerifier* log_verifier, Task* task)
    : db_(CHECK_NOTNULL(db)),
      peer_group_(move(peer_group)),
      log_verifier_(CHECK_NOTNULL(log_verifier)),
      task_(CHECK_NOTNULL(task)),
      start_(db_->TreeSize()) {
  // TODO(pphaneuf): Might be better to get that as a parameter?
  const int64_t remote_tree_size(peer_group_->TreeSize());
  CHECK_GE(start_, 0);

  // Nothing to do...
  if (remote_tree_size <= start_) {
    VLOG(1) << "nothing to do: we have " << start_ << " entries, remote has "
            << remote_tree_size;
    task_->Return();
    return;
  }

  entries_.reset(new Range(Range::WANT, remote_tree_size - start_));

  WalkEntries();
}


// This is called either when starting the fetching, or when fetching
// a range completed. In that both cases, there's a hold on our task,
// so it shouldn't go away from under us.
void FetchState::WalkEntries() {
  if (!task_->IsActive()) {
    // We've already stopped, for one reason or another, no point
    // getting anything started.
    return;
  }

  if (task_->CancelRequested()) {
    task_->Return(Status::CANCELLED);
    return;
  }

  unique_lock<mutex> lock(lock_);

  // Prune fetched and unavailable sequences at the beginning.
  const int64_t remote_tree_size(peer_group_->TreeSize());
  while (entries_ &&
         (entries_->state_ == Range::HAVE ||
          (entries_->state_ == Range::WANT && remote_tree_size < start_))) {
    VLOG(1) << "pruning " << entries_->size_ << " at offset " << start_;
    start_ += entries_->size_;
    entries_ = move(entries_->next_);
  }

  // Are we done?
  if (!entries_) {
    task_->Return();
    return;
  }

  int64_t index(start_);
  int num_fetch(0);
  for (Range *current = entries_.get(); current;
       index += current->size_, current = current->next_.get()) {
    // Coalesce with the next Range, if possible.
    if (current->state_ != Range::FETCHING) {
      while (current->next_ && current->next_->state_ == current->state_) {
        current->size_ += current->next_->size_;
        current->next_ = move(current->next_->next_);
      }
    }

    switch (current->state_) {
      case Range::HAVE:
        VLOG(2) << "at offset " << index << ", we have " << current->size_
                << " entries";
        break;

      case Range::FETCHING:
        VLOG(2) << "at offset " << index << ", fetching " << current->size_
                << " entries";
        ++num_fetch;
        break;

      case Range::WANT:
        VLOG(2) << "at offset " << index << ", we want " << current->size_
                << " entries";

        // Do not start a fetch if we think our peer group does not
        // have it.
        if (index >= remote_tree_size) {
          break;
        }

        // If the range is bigger than the maximum batch size, split it.
        if (current->size_ > FLAGS_fetcher_batch_size) {
          current->next_.reset(
              new Range(Range::WANT, current->size_ - FLAGS_fetcher_batch_size,
                        move(current->next_)));
          current->size_ = FLAGS_fetcher_batch_size;
        }

        FetchRange(lock, current, index,
                   task_->AddChild(bind(&FetchState::WalkEntries, this)));
        ++num_fetch;

        break;
    }

    if (num_fetch >= FLAGS_fetcher_concurrent_fetches ||
        index >= remote_tree_size) {
      break;
    }
  }
}


void FetchState::FetchRange(const unique_lock<mutex>& lock, Range* current,
                            int64_t index, Task* range_task) {
  CHECK(lock.owns_lock());
  const int64_t end_index(index + current->size_ - 1);
  VLOG(1) << "fetching from offset " << index << " to " << end_index;

  vector<AsyncLogClient::Entry>* const retval(
      new vector<AsyncLogClient::Entry>);
  range_task->DeleteWhenDone(retval);

  current->state_ = Range::FETCHING;

  peer_group_->FetchEntries(index, end_index, retval,
                            range_task->AddChild(
                                bind(&FetchState::WriteToDatabase, this, index,
                                     current, retval, range_task, _1)));
}


void FetchState::WriteToDatabase(int64_t index, Range* range,
                                 const vector<AsyncLogClient::Entry>* retval,
                                 Task* range_task, Task* fetch_task) {
  if (!fetch_task->status().ok()) {
    LOG(INFO) << "error fetching entries at index " << index << ": "
              << fetch_task->status();
    lock_guard<mutex> lock(lock_);
    range->state_ = Range::WANT;
    range_task->Return(fetch_task->status());
    return;
  }

  CHECK_GT(retval->size(), static_cast<size_t>(0));

  VLOG(1) << "received " << retval->size() << " entries at offset " << index;
  int64_t processed(0);
  for (const auto& entry : *retval) {
    LoggedEntry cert;
    if (!cert.CopyFromClientLogEntry(entry)) {
      LOG(WARNING) << "could not convert entry to a LoggedEntry";
      num_invalid_entries_fetched->Increment("format");
      break;
    }
    if (entry.sct) {
      *cert.mutable_sct() = *entry.sct;
      // If we have the full SCT (because this LogEntry came from another
      // internal node which supports our private "give me the SCT too"
      // option), then verify that the signature is good.
      const LogVerifier::LogVerifyResult verify_result(
          log_verifier_->VerifySignedCertificateTimestamp(
              cert.contents().entry(), cert.sct()));
      VLOG(1) << "SCT verify entry #" << index << ": "
              << LogVerifier::VerifyResultString(verify_result);
      if (verify_result != LogVerifier::VERIFY_OK) {
        num_invalid_entries_fetched->Increment("sct_verify_failed");
        const string msg("Failed to verify SCT signature for entry# " +
                         to_string(index) + " : " +
                         LogVerifier::VerifyResultString(verify_result));
        LOG(WARNING) << msg;
        task_->Return(Status(util::error::FAILED_PRECONDITION, msg));
        return;
      }
    }
    cert.set_sequence_number(index++);
    if (db_->CreateSequencedEntry(cert) == Database::OK) {
      ++processed;
    } else {
      LOG(WARNING) << "could not insert entry into the database:\n"
                   << cert.DebugString();
      break;
    }
  }

  {
    lock_guard<mutex> lock(lock_);
    // TODO(pphaneuf): If we have problems fetching entries, to what
    // point should we retry? Or should we just return on the task
    // with an error?
    if (processed > 0) {
      // If we don't receive everything, split up the range.
      if (range->size_ > processed) {
        range->next_.reset(new Range(Range::WANT, range->size_ - processed,
                                     move(range->next_)));
        range->size_ = processed;
      }

      range->state_ = Range::HAVE;
    } else {
      range->state_ = Range::WANT;
    }
  }

  if (static_cast<uint64_t>(processed) < retval->size()) {
    // We couldn't insert everything that we received into the
    // database, this is fairly serious, return an error for the
    // overall operation and let the higher level deal with it.
    task_->Return(Status(util::error::INTERNAL,
                         "could not write some entries to the database"));
  }

  range_task->Return();
}


}  // namespace


void FetchLogEntries(Database* db, unique_ptr<PeerGroup> peer_group,
                     const LogVerifier* log_verifier, Task* task) {
  TaskHold hold(task);
  task->DeleteWhenDone(
      new FetchState(db, move(peer_group), log_verifier, task));
}


}  // namespace cert_trans
