#include "fetcher/continuous_fetcher.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "fetcher/fetcher.h"
#include "fetcher/peer_group.h"
#include "log/log_verifier.h"

using std::bind;
using std::chrono::seconds;
using std::lock_guard;
using std::map;
using std::move;
using std::mutex;
using std::placeholders::_1;
using std::shared_ptr;
using std::string;
using std::unique_lock;
using std::unique_ptr;
using util::Executor;
using util::Task;

DEFINE_int32(delay_between_fetches_seconds, 30, "delay between fetches");

namespace cert_trans {

namespace {


class ContinuousFetcherImpl : public ContinuousFetcher {
 public:
  ContinuousFetcherImpl(libevent::Base* base, Executor* executor, Database* db,
                        const LogVerifier* log_verifier, bool fetch_scts);

  void AddPeer(const string& node_id, const shared_ptr<Peer>& peer) override;
  void RemovePeer(const string& node_id) override;

 private:
  void StartFetch(const unique_lock<mutex>& lock);
  void FetchDone(Task* task);
  void FetchDelayDone(Task* task);

  libevent::Base* const base_;
  Executor* const executor_;
  Database* const db_;
  const LogVerifier* const log_verifier_;
  const bool fetch_scts_;

  mutex lock_;
  map<string, shared_ptr<Peer>> peers_;

  bool restart_fetch_;
  unique_ptr<Task> fetch_task_;

  DISALLOW_COPY_AND_ASSIGN(ContinuousFetcherImpl);
};


ContinuousFetcherImpl::ContinuousFetcherImpl(
    libevent::Base* base, Executor* executor, Database* db,
    const LogVerifier* const log_verifier, bool fetch_scts)
    : base_(CHECK_NOTNULL(base)),
      executor_(CHECK_NOTNULL(executor)),
      db_(CHECK_NOTNULL(db)),
      log_verifier_(CHECK_NOTNULL(log_verifier)),
      fetch_scts_(fetch_scts),
      restart_fetch_(false) {
}


void ContinuousFetcherImpl::AddPeer(const string& node_id,
                                    const shared_ptr<Peer>& peer) {
  unique_lock<mutex> lock(lock_);

  // TODO(pphaneuf): Allow updating the peer, as this is currently
  // simplest, for the case where a peer might change host:port.
  const auto it(peers_.find(node_id));
  if (it != peers_.end()) {
    it->second = peer;
  } else {
    CHECK(peers_.emplace(node_id, peer).second);
  }

  if (fetch_task_) {
    restart_fetch_ = true;
    fetch_task_->Cancel();
  } else {
    StartFetch(lock);
  }
}


void ContinuousFetcherImpl::RemovePeer(const string& node_id) {
  lock_guard<mutex> lock(lock_);

  // TODO(pphaneuf): In tests, we rig more than cluster state
  // controllers to the same continuous fetcher instance, so additions
  // and removals can be duplicated. Tolerate this for now, but only
  // restart the fetching process if there was an actual removal.
  if (peers_.erase(node_id) > 0 && fetch_task_) {
    restart_fetch_ = true;
    fetch_task_->Cancel();
  }
}


void ContinuousFetcherImpl::StartFetch(const unique_lock<mutex>& lock) {
  CHECK(lock.owns_lock());
  CHECK(!fetch_task_);

  restart_fetch_ = false;

  unique_ptr<PeerGroup> peer_group(new PeerGroup(fetch_scts_));
  for (const auto& peer : peers_) {
    peer_group->Add(peer.second);
  }

  fetch_task_.reset(
      new Task(bind(&ContinuousFetcherImpl::FetchDone, this, _1), executor_));

  VLOG(1) << "starting fetch with tree size: " << peer_group->TreeSize();
  FetchLogEntries(db_, move(peer_group), log_verifier_, fetch_task_.get());
}


void ContinuousFetcherImpl::FetchDone(Task* task) {
  if (!task->status().ok()) {
    LOG(WARNING) << "error while fetching: " << task->status();
  }

  lock_guard<mutex> lock(lock_);
  fetch_task_.reset();

  if (restart_fetch_) {
    executor_->Add(
        bind(&ContinuousFetcherImpl::FetchDelayDone, this, nullptr));
  } else {
    base_->Delay(seconds(FLAGS_delay_between_fetches_seconds),
                 new Task(bind(&ContinuousFetcherImpl::FetchDelayDone, this,
                               _1),
                          executor_));
  }
}


void ContinuousFetcherImpl::FetchDelayDone(Task* task) {
  // "task" can be null, if we're restarting a fetch.
  if (task) {
    CHECK_EQ(util::Status::OK, task->status());
    delete task;
  }

  unique_lock<mutex> lock(lock_);
  if (!fetch_task_) {
    StartFetch(lock);
  }
}


}  // namespace


// static
unique_ptr<ContinuousFetcher> ContinuousFetcher::New(
    libevent::Base* base, Executor* executor, Database* db,
    const LogVerifier* log_verifier, bool fetch_scts) {
  return unique_ptr<ContinuousFetcher>(
      new ContinuousFetcherImpl(base, executor, db, log_verifier, fetch_scts));
}


}  // namespace cert_trans
