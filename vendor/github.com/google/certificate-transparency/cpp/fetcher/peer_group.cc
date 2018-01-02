#include "fetcher/peer_group.h"

#include <glog/logging.h>

using std::lock_guard;
using std::max;
using std::mutex;
using std::placeholders::_1;
using std::shared_ptr;
using std::vector;
using util::Status;
using util::Task;

namespace cert_trans {

namespace {


void GetEntriesDone(AsyncLogClient::Status client_status,
                    const vector<AsyncLogClient::Entry>* entries, Task* task) {
  Status status;

  switch (client_status) {
    case AsyncLogClient::OK:
      break;

    default:
      // TODO(pphaneuf): Improve this a bit? Or wouldn't it be nice if
      // AsyncLogClient gave us a util::Status in the first place? ;-)
      status = util::Status::UNKNOWN;
  }

  if (status.ok() && entries->empty()) {
    // This should never happen.
    status =
        Status(util::error::INTERNAL, "log server did not return any entries");
  }

  task->Return(status);
}


}  // namespace


PeerGroup::PeerGroup(bool fetch_scts) : fetch_scts_(fetch_scts) {
}


void PeerGroup::Add(const shared_ptr<Peer>& peer) {
  lock_guard<mutex> lock(lock_);

  CHECK(peers_.emplace(peer, PeerState()).second);
}


int64_t PeerGroup::TreeSize() const {
  lock_guard<mutex> lock(lock_);

  int64_t tree_size(-1);
  for (const auto& peer : peers_) {
    tree_size = max(tree_size, peer.first->TreeSize());
  }

  return tree_size;
}


void PeerGroup::FetchEntries(int64_t start_index, int64_t end_index,
                             vector<AsyncLogClient::Entry>* entries,
                             Task* task) {
  CHECK_GE(start_index, 0);
  CHECK_GE(end_index, start_index);

  const shared_ptr<Peer> peer(PickPeer(end_index + 1));
  if (!peer) {
    task->Return(Status(util::error::UNAVAILABLE,
                        "requested entries not available in the peer group"));
    return;
  }

  // TODO(pphaneuf): Handle the case where we have no peer more cleanly.
  if (fetch_scts_) {
    peer->client().GetEntriesAndSCTs(start_index, end_index,
                                     CHECK_NOTNULL(entries),
                                     bind(GetEntriesDone, _1, entries, task));
  } else {
    peer->client().GetEntries(start_index, end_index, CHECK_NOTNULL(entries),
                              bind(GetEntriesDone, _1, entries, task));
  }
}


shared_ptr<Peer> PeerGroup::PickPeer(const int64_t needed_size) const {
  lock_guard<mutex> lock(lock_);

  // TODO(pphaneuf): We should pick peers a bit more cleverly, to
  // spread the load somewhat.
  int64_t group_tree_size(-1);
  vector<shared_ptr<Peer>> capable_peers;
  for (const auto& peer : peers_) {
    const int64_t tree_size(peer.first->TreeSize());
    group_tree_size = max(group_tree_size, tree_size);
    if (tree_size >= needed_size) {
      capable_peers.push_back(peer.first);
    }
  }

  if (!capable_peers.empty()) {
    return capable_peers[std::rand() % capable_peers.size()];
  }

  LOG(INFO) << "requested a peer with " << needed_size
            << " entries but the peer group only has " << group_tree_size
            << " entries";

  return nullptr;
}


}  // namespace cert_trans
