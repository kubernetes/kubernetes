#include "server/proxy.h"

#include <event2/buffer.h>
#include <event2/http.h>
#include <event2/http_compat.h>
#include <event2/keyvalq_struct.h>
#include <glog/logging.h>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "monitoring/monitoring.h"
#include "proto/ct.pb.h"
#include "server/json_output.h"
#include "util/libevent_wrapper.h"


using ct::ClusterNodeState;
using std::bind;
using std::getline;
using std::make_pair;
using std::pair;
using std::placeholders::_1;
using std::rand;
using std::string;
using std::stringstream;
using std::unique_ptr;
using std::unordered_set;
using std::vector;
using util::Executor;
using util::Task;

namespace cert_trans {
namespace {


static Counter<string>* total_proxied_requests(
    Counter<string>::New("total_proxied_requests", "path",
                         "Number of proxied API requests by path."));
static Counter<string, int>* total_proxied_responses(
    Counter<string, int>::New("total_proxied_responses", "path", "status_code",
                              "Number of proxied API requests by path "
                              "and status code."));


void ProxyRequestDone(libevent::Base* base, evhttp_request* request,
                      const string& path, UrlFetcher::Response* response,
                      Task* task) {
  CHECK_NOTNULL(request);
  CHECK_NOTNULL(task);
  unique_ptr<UrlFetcher::Response> response_deleter(CHECK_NOTNULL(response));

  total_proxied_requests->Increment(path);
  total_proxied_responses->Increment(path, response->status_code);

  if (!task->status().ok()) {
    return SendJsonError(base, request, HTTP_INTERNAL,
                         "Proxied request failed.");
  }

  // TODO(alcutter): Consider retrying the proxied request some number of times
  // in the case where the request fails.
  FilterHeaders(&response->headers);
  for (auto it(response->headers.begin()); it != response->headers.end();
       ++it) {
    CHECK_EQ(evhttp_add_header(evhttp_request_get_output_headers(request),
                               it->first.c_str(), it->second.c_str()),
             0);
  }
  CHECK_GT(evbuffer_add_printf(evhttp_request_get_output_buffer(request), "%s",
                               response->body.c_str()),
           -1);

  const int response_code(response->status_code);
  base->Add([request, response_code]() {
    evhttp_send_reply(request, response_code, /*reason*/ NULL,
                      /*databuf*/ NULL);
  });
}


}  // namespace


// Filters out any headers which should not be proxied on.
// See http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.10
void FilterHeaders(UrlFetcher::Headers* headers) {
  const auto connection_range(headers->equal_range("Connection"));

  unordered_set<string> connection_headers;
  for (auto it(connection_range.first); it != connection_range.second; ++it) {
    stringstream ss(it->second);
    string token;
    while (getline(ss, token, ',')) {
      token.erase(remove(token.begin(), token.end(), ' '), token.end());
      if (!token.empty()) {
        connection_headers.insert(token);
      }
    }
  }

  headers->erase(connection_range.first, connection_range.second);

  for (const string& header : connection_headers) {
    const auto range(headers->equal_range(header));
    if (range.first != headers->end()) {
      headers->erase(range.first, range.second);
    }
  }
}


Proxy::Proxy(libevent::Base* base,
             const GetFreshNodesFunction& get_fresh_nodes, UrlFetcher* fetcher,
             Executor* executor)
    : base_(CHECK_NOTNULL(base)),
      get_fresh_nodes_(get_fresh_nodes),
      fetcher_(CHECK_NOTNULL(fetcher)),
      executor_(CHECK_NOTNULL(executor)) {
  CHECK(get_fresh_nodes_);
}


void Proxy::ProxyRequest(evhttp_request* req) const {
  CHECK_NOTNULL(req);

  const vector<ClusterNodeState> fresh_nodes(get_fresh_nodes_());
  if (fresh_nodes.empty()) {
    return SendJsonError(base_, req, HTTP_SERVUNAVAIL,
                         "No node able to serve request.");
  }
  const ClusterNodeState& target(fresh_nodes[rand() % fresh_nodes.size()]);

  URL url(evhttp_request_uri(req));
  url.SetProtocol("http");
  url.SetHost(target.hostname());
  url.SetPort(target.log_port());

  UrlFetcher::Request fetcher_req(url);

  switch (evhttp_request_get_command(req)) {
    case EVHTTP_REQ_DELETE:
      fetcher_req.verb = UrlFetcher::Verb::DELETE;
      break;
    case EVHTTP_REQ_GET:
      fetcher_req.verb = UrlFetcher::Verb::GET;
      break;
    case EVHTTP_REQ_POST:
      fetcher_req.verb = UrlFetcher::Verb::POST;
      break;
    case EVHTTP_REQ_PUT:
      fetcher_req.verb = UrlFetcher::Verb::PUT;
      break;
    default:
      return SendJsonError(base_, req, HTTP_BADMETHOD,
                           "Bad method requested.");
      break;
  }

  for (evkeyval* ptr = evhttp_request_get_input_headers(req)->tqh_first; ptr;
       ptr = ptr->next.tqe_next) {
    fetcher_req.headers.insert(make_pair(ptr->key, ptr->value));
  }
  FilterHeaders(&fetcher_req.headers);
  if (fetcher_req.verb == UrlFetcher::Verb::PUT ||
      fetcher_req.verb == UrlFetcher::Verb::POST) {
    const size_t body_length(
        evbuffer_get_length(evhttp_request_get_input_buffer(req)));
    string body(reinterpret_cast<const char*>(evbuffer_pullup(
                    evhttp_request_get_input_buffer(req), body_length)),
                body_length);
    CHECK_EQ(0, evbuffer_drain(evhttp_request_get_input_buffer(req),
                               body_length));
    fetcher_req.body.swap(body);
  }
  VLOG(1) << "Proxying request to " << url.Host() << ":" << url.Port()
          << url.PathQuery();
  UrlFetcher::Response* resp(new UrlFetcher::Response);
  fetcher_->Fetch(fetcher_req, resp, new Task(bind(&ProxyRequestDone, base_,
                                                   req, url.Path(), resp, _1),
                                              executor_));
}


}  // namespace cert_trans
