#ifndef CERT_TRANS_FETCHER_PEER_H_
#define CERT_TRANS_FETCHER_PEER_H_

#include <memory>

#include "base/macros.h"
#include "client/async_log_client.h"

namespace cert_trans {


class Peer {
 public:
  Peer(std::unique_ptr<AsyncLogClient> client);
  virtual ~Peer() {
  }

  AsyncLogClient& client() {
    return *client_;
  }

  // Returns -1 if we do not know yet.
  virtual int64_t TreeSize() const = 0;

 protected:
  const std::unique_ptr<AsyncLogClient> client_;

 private:
  DISALLOW_COPY_AND_ASSIGN(Peer);
};


}  // namespace cert_trans


#endif  // CERT_TRANS_FETCHER_PEER_H_
