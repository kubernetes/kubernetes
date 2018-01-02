#ifndef CERT_TRANS_SERVER_PROXY_H_
#define CERT_TRANS_SERVER_PROXY_H_

#include <functional>
#include <vector>

#include "base/macros.h"
#include "net/url_fetcher.h"


struct evhttp_request;

namespace ct {
class ClusterNodeState;
}  // namespace ct

namespace util {
class Executor;
class Status;
class Task;
}  // namespace util


namespace cert_trans {
namespace libevent {
class Base;
}  // namespace libevent


// Visible for testing
void FilterHeaders(UrlFetcher::Headers* headers);


class Proxy {
 public:
  typedef std::function<std::vector<ct::ClusterNodeState>()>
      GetFreshNodesFunction;
  Proxy(libevent::Base* base, const GetFreshNodesFunction& get_fresh_nodes,
        UrlFetcher* fetcher, util::Executor* executor);

  virtual ~Proxy() = default;

  virtual void ProxyRequest(evhttp_request* req) const;

 private:
  libevent::Base* const base_;
  const GetFreshNodesFunction get_fresh_nodes_;
  UrlFetcher* const fetcher_;
  util::Executor* const executor_;

  DISALLOW_COPY_AND_ASSIGN(Proxy);
};


}  // namespace cert_trans

#endif  // CERT_TRANS_SERVER_PROXY_H_
