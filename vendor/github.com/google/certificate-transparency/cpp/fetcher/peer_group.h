#ifndef CERT_TRANS_FETCHER_PEER_GROUP_H_
#define CERT_TRANS_FETCHER_PEER_GROUP_H_

#include <stdint.h>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "base/macros.h"
#include "client/async_log_client.h"
#include "fetcher/peer.h"
#include "util/task.h"

namespace cert_trans {


// A PeerGroup is a set of peers used for a fetch operation, providing
// a slightly higher level abstraction for fetching entries. Fetch
// errors will be retried, and unhealthy peers will be dropped (so the
// available tree size can get smaller).
// TODO(pphaneuf): Make that last sentence true!
class PeerGroup {
 public:
  explicit PeerGroup(bool fetch_scts_);

  // Adding a peer twice is not allowed.
  void Add(const std::shared_ptr<Peer>& peer);

  // Returns the highest tree size of the peer group.
  int64_t TreeSize() const;

  void FetchEntries(int64_t start_offset, int64_t end_offset,
                    std::vector<AsyncLogClient::Entry>* entries,
                    util::Task* task);

 private:
  struct PeerState {
    // TODO(pphaneuf): Keep a count of errors here, to prune away
    // unhealthy peers.
  };

  std::shared_ptr<Peer> PickPeer(const int64_t needed_size) const;

  mutable std::mutex lock_;
  const bool fetch_scts_;
  std::map<std::shared_ptr<Peer>, PeerState> peers_;

  DISALLOW_COPY_AND_ASSIGN(PeerGroup);
};


}  // namespace cert_trans

#endif  // CERT_TRANS_FETCHER_PEER_GROUP_H_
