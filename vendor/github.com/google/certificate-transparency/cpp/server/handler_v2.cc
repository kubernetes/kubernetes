#include "server/handler_v2.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <algorithm>
#include <string>
#include <vector>

#include "log/cert.h"
#include "log/cert_checker.h"
#include "log/cluster_state_controller.h"
#include "log/log_lookup.h"
#include "log/logged_entry.h"
#include "monitoring/latency.h"
#include "monitoring/monitoring.h"
#include "server/json_output.h"
#include "server/proxy.h"
#include "util/json_wrapper.h"
#include "util/thread_pool.h"

namespace libevent = cert_trans::libevent;

using cert_trans::Counter;
using cert_trans::HttpHandlerV2;
using cert_trans::Latency;
using cert_trans::LoggedEntry;
using cert_trans::Proxy;
using cert_trans::ScopedLatency;
using ct::ShortMerkleAuditProof;
using ct::SignedCertificateTimestamp;
using ct::SignedTreeHead;
using std::bind;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::lock_guard;
using std::make_shared;
using std::multimap;
using std::min;
using std::mutex;
using std::placeholders::_1;
using std::string;
using std::unique_ptr;
using std::vector;

DEFINE_int32(max_leaf_entries_per_response, 1000,
             "maximum number of entries to put in the response of a "
             "get-entries request");

namespace {


static Latency<milliseconds, string> http_server_request_latency_ms(
    "total_http_server_request_latency_ms", "path",
    "Total request latency in ms broken down by path");


}  // namespace


HttpHandlerV2::HttpHandlerV2(
    LogLookup* log_lookup, const ReadOnlyDatabase* db,
    const ClusterStateController<LoggedEntry>* controller, ThreadPool* pool,
    libevent::Base* event_base, StalenessTracker* staleness_tracker)
    : log_lookup_(CHECK_NOTNULL(log_lookup)),
      db_(CHECK_NOTNULL(db)),
      controller_(CHECK_NOTNULL(controller)),
      proxy_(nullptr),
      pool_(CHECK_NOTNULL(pool)),
      event_base_(CHECK_NOTNULL(event_base)),
      staleness_tracker_(CHECK_NOTNULL(staleness_tracker)) {
}


HttpHandlerV2::~HttpHandlerV2() {
}


void StatsHandlerInterceptor(const string& path,
                             const libevent::HttpServer::HandlerCallback& cb,
                             evhttp_request* req) {
  ScopedLatency total_http_server_request_latency(
      http_server_request_latency_ms.GetScopedLatency(path));

  cb(req);
}


void HttpHandlerV2::AddEntryReply(
    evhttp_request* req, const util::Status& add_status,
    const SignedCertificateTimestamp& sct) const {
  return SendJsonError(event_base_, req, HTTP_NOTIMPLEMENTED,
                       "Not yet implemented.");
}

void HttpHandlerV2::ProxyInterceptor(
    const libevent::HttpServer::HandlerCallback& local_handler,
    evhttp_request* request) {
  VLOG(2) << "Running proxy interceptor...";
  // TODO(alcutter): We can be a bit smarter about when to proxy off
  // the request - being stale wrt to the current serving STH doesn't
  // automatically mean we're unable to answer this request.
  if (staleness_tracker_->IsNodeStale()) {
    // Can't do this on the libevent thread since it can block on the lock in
    // ClusterStatusController::GetFreshNodes().
    pool_->Add(bind(&Proxy::ProxyRequest, proxy_, request));
  } else {
    local_handler(request);
  }
}


void HttpHandlerV2::AddProxyWrappedHandler(
    libevent::HttpServer* server, const string& path,
    const libevent::HttpServer::HandlerCallback& local_handler) {
  const libevent::HttpServer::HandlerCallback stats_handler(
      bind(&StatsHandlerInterceptor, path, local_handler, _1));
  CHECK(server->AddHandler(path, bind(&HttpHandlerV2::ProxyInterceptor, this,
                                      stats_handler, _1)));
}


void HttpHandlerV2::Add(libevent::HttpServer* server) {
  CHECK_NOTNULL(server);
  // TODO(pphaneuf): An optional prefix might be nice?
  // TODO(pphaneuf): Find out which methods are CPU intensive enough
  // that they should be spun off to the thread pool.
  AddProxyWrappedHandler(server, "/ct/v2/get-entries",
                         bind(&HttpHandlerV2::GetEntries, this, _1));
  AddProxyWrappedHandler(server, "/ct/v2/get-proof-by-hash",
                         bind(&HttpHandlerV2::GetProof, this, _1));
  AddProxyWrappedHandler(server, "/ct/v2/get-sth",
                         bind(&HttpHandlerV2::GetSTH, this, _1));
  AddProxyWrappedHandler(server, "/ct/v2/get-sth-consistency",
                         bind(&HttpHandlerV2::GetConsistency, this, _1));

  // Now add any sub-class handlers.
  AddHandlers(server);
}


void HttpHandlerV2::SetProxy(Proxy* proxy) {
  LOG_IF(FATAL, proxy_) << "Attempting to re-add a Proxy.";
  proxy_ = CHECK_NOTNULL(proxy);
}


void HttpHandlerV2::GetEntries(evhttp_request* req) const {
  return SendJsonError(event_base_, req, HTTP_NOTIMPLEMENTED,
                       "Not yet implemented.");
}


void HttpHandlerV2::GetProof(evhttp_request* req) const {
  return SendJsonError(event_base_, req, HTTP_NOTIMPLEMENTED,
                       "Not yet implemented.");
}


void HttpHandlerV2::GetSTH(evhttp_request* req) const {
  return SendJsonError(event_base_, req, HTTP_NOTIMPLEMENTED,
                       "Not yet implemented.");
}


void HttpHandlerV2::GetConsistency(evhttp_request* req) const {
  return SendJsonError(event_base_, req, HTTP_NOTIMPLEMENTED,
                       "Not yet implemented.");
}


void HttpHandlerV2::BlockingGetEntries(evhttp_request* req, int64_t start,
                                       int64_t end, bool include_scts) const {
  return SendJsonError(event_base_, req, HTTP_NOTIMPLEMENTED,
                       "Not yet implemented.");
}
