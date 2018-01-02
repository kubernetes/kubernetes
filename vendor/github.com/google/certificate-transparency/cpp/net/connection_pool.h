#ifndef CERT_TRANS_NET_CONNECTION_POOL_H_
#define CERT_TRANS_NET_CONNECTION_POOL_H_

#include <openssl/ssl.h>
#include <stdint.h>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "base/macros.h"
#include "net/url.h"
#include "util/libevent_wrapper.h"

namespace cert_trans {


// Status code for when something went wrong with the connection.
const int kUnknownErrorStatus = 0;

// Status code for when there was an error with the SSL negotiation.
const int kSSLErrorStatus = 1;

// Status code for when the connection timed-out.
const int kTimeout = 2;


namespace internal {


struct evhtp_connection_deleter {
  void operator()(evhtp_connection_t* con) const {
    evhtp_connection_free(con);
  }
};


typedef std::pair<std::string, uint16_t> HostPortPair;
class EvConnection;


class ConnectionPool {
 public:
  class Connection {
   public:
    Connection(const std::shared_ptr<EvConnection>& evc);

    evhtp_connection_t* connection() const;

    const HostPortPair& other_end() const;

    bool GetErrored() const;

   private:
    std::shared_ptr<EvConnection> connection_;
    friend class ConnectionPool;

    DISALLOW_COPY_AND_ASSIGN(Connection);
  };

  ConnectionPool(libevent::Base* base);

  std::unique_ptr<Connection> Get(const URL& url);
  void Put(std::unique_ptr<Connection> conn);

 private:
  typedef std::pair<std::chrono::system_clock::time_point,
                    std::unique_ptr<Connection>> TimestampedConnection;

  static void RemoveDeadConnectionsFromDeque(
      const std::unique_lock<std::mutex>& lock,
      std::deque<TimestampedConnection>* deque);

  void Cleanup();

  libevent::Base* const base_;

  std::mutex lock_;
  // We get and put connections from the back of the deque, and when
  // there are too many, we prune them from the front (LIFO).
  std::map<HostPortPair, std::deque<TimestampedConnection>> conns_;
  bool cleanup_scheduled_;

  std::unique_ptr<evhtp_ssl_ctx_t, void (*)(evhtp_ssl_ctx_t*)> ssl_ctx_;

  DISALLOW_COPY_AND_ASSIGN(ConnectionPool);
};


}  // namespace internal
}  // namespace cert_trans

#endif  // CERT_TRANS_NET_CONNECTION_POOL_H_
