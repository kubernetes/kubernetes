#include "fetcher/peer.h"

#include <glog/logging.h>

using std::unique_ptr;

namespace cert_trans {


Peer::Peer(unique_ptr<AsyncLogClient> client) : client_(move(client)) {
  CHECK_NOTNULL(client_.get());
}


}  // namespace cert_trans
