#include "server/handler.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdint.h>
#include <stdlib.h>
#include <algorithm>
#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
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
using cert_trans::HttpHandler;
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


HttpHandler::HttpHandler(LogLookup* log_lookup, const ReadOnlyDatabase* db,
                         const ClusterStateController<LoggedEntry>* controller,
                         ThreadPool* pool, libevent::Base* event_base,
                         StalenessTracker* staleness_tracker)
    : log_lookup_(CHECK_NOTNULL(log_lookup)),
      db_(CHECK_NOTNULL(db)),
      controller_(CHECK_NOTNULL(controller)),
      proxy_(nullptr),
      pool_(CHECK_NOTNULL(pool)),
      event_base_(CHECK_NOTNULL(event_base)),
      staleness_tracker_(CHECK_NOTNULL(staleness_tracker)) {
}


HttpHandler::~HttpHandler() {
}


void StatsHandlerInterceptor(const string& path,
                             const libevent::HttpServer::HandlerCallback& cb,
                             evhttp_request* req) {
  ScopedLatency total_http_server_request_latency(
      http_server_request_latency_ms.GetScopedLatency(path));

  cb(req);
}


void HttpHandler::AddEntryReply(evhttp_request* req,
                                const util::Status& add_status,
                                const SignedCertificateTimestamp& sct) const {
  if (!add_status.ok() &&
      add_status.CanonicalCode() != util::error::ALREADY_EXISTS) {
    VLOG(1) << "error adding chain: " << add_status;
    const int response_code(add_status.CanonicalCode() ==
                                    util::error::RESOURCE_EXHAUSTED
                                ? HTTP_SERVUNAVAIL
                                : HTTP_BADREQUEST);
    return SendJsonError(event_base_, req, response_code,
                         add_status.error_message());
  }

  JsonObject json_reply;
  json_reply.Add("sct_version", static_cast<int64_t>(0));
  json_reply.AddBase64("id", sct.id().key_id());
  json_reply.Add("timestamp", sct.timestamp());
  json_reply.Add("extensions", "");
  json_reply.Add("signature", sct.signature());

  SendJsonReply(event_base_, req, HTTP_OK, json_reply);
}

void HttpHandler::ProxyInterceptor(
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


void HttpHandler::AddProxyWrappedHandler(
    libevent::HttpServer* server, const string& path,
    const libevent::HttpServer::HandlerCallback& local_handler) {
  const libevent::HttpServer::HandlerCallback stats_handler(
      bind(&StatsHandlerInterceptor, path, local_handler, _1));
  CHECK(server->AddHandler(path, bind(&HttpHandler::ProxyInterceptor, this,
                                      stats_handler, _1)));
}


void HttpHandler::Add(libevent::HttpServer* server) {
  CHECK_NOTNULL(server);
  // TODO(pphaneuf): An optional prefix might be nice?
  // TODO(pphaneuf): Find out which methods are CPU intensive enough
  // that they should be spun off to the thread pool.
  AddProxyWrappedHandler(server, "/ct/v1/get-entries",
                         bind(&HttpHandler::GetEntries, this, _1));
  AddProxyWrappedHandler(server, "/ct/v1/get-proof-by-hash",
                         bind(&HttpHandler::GetProof, this, _1));
  AddProxyWrappedHandler(server, "/ct/v1/get-sth",
                         bind(&HttpHandler::GetSTH, this, _1));
  AddProxyWrappedHandler(server, "/ct/v1/get-sth-consistency",
                         bind(&HttpHandler::GetConsistency, this, _1));

  // Now add any sub-class handlers.
  AddHandlers(server);
}


void HttpHandler::SetProxy(Proxy* proxy) {
  LOG_IF(FATAL, proxy_) << "Attempting to re-add a Proxy.";
  proxy_ = CHECK_NOTNULL(proxy);
}


void HttpHandler::GetEntries(evhttp_request* req) const {
  if (evhttp_request_get_command(req) != EVHTTP_REQ_GET) {
    return SendJsonError(event_base_, req, HTTP_BADMETHOD,
                         "Method not allowed.");
  }

  const libevent::QueryParams query(libevent::ParseQuery(req));

  const int64_t start(libevent::GetIntParam(query, "start"));
  if (start < 0) {
    return SendJsonError(event_base_, req, HTTP_BADREQUEST,
                         "Missing or invalid \"start\" parameter.");
  }

  int64_t end(libevent::GetIntParam(query, "end"));
  if (end < start) {
    return SendJsonError(event_base_, req, HTTP_BADREQUEST,
                         "Missing or invalid \"end\" parameter.");
  }

  // Limit the number of entries returned in a single request.
  end = std::min(end, start + FLAGS_max_leaf_entries_per_response);

  // Sekrit parameter to indicate that SCTs should be included too.
  // This is non-standard, and is only used internally by other log nodes when
  // "following" nodes with more data.
  const bool include_scts(libevent::GetBoolParam(query, "include_scts"));

  BlockingGetEntries(req, start, end, include_scts);
}


void HttpHandler::GetProof(evhttp_request* req) const {
  if (evhttp_request_get_command(req) != EVHTTP_REQ_GET) {
    return SendJsonError(event_base_, req, HTTP_BADMETHOD,
                         "Method not allowed.");
  }

  const libevent::QueryParams query(libevent::ParseQuery(req));

  string b64_hash;
  if (!libevent::GetParam(query, "hash", &b64_hash)) {
    return SendJsonError(event_base_, req, HTTP_BADREQUEST,
                         "Missing or invalid \"hash\" parameter.");
  }

  const string hash(util::FromBase64(b64_hash.c_str()));
  if (hash.empty()) {
    return SendJsonError(event_base_, req, HTTP_BADREQUEST,
                         "Invalid \"hash\" parameter.");
  }

  const int64_t tree_size(libevent::GetIntParam(query, "tree_size"));
  if (tree_size < 0 ||
      static_cast<int64_t>(tree_size) > log_lookup_->GetSTH().tree_size()) {
    return SendJsonError(event_base_, req, HTTP_BADREQUEST,
                         "Missing or invalid \"tree_size\" parameter.");
  }

  ShortMerkleAuditProof proof;
  if (log_lookup_->AuditProof(hash, tree_size, &proof) != LogLookup::OK) {
    return SendJsonError(event_base_, req, HTTP_BADREQUEST,
                         "Couldn't find hash.");
  }

  JsonArray json_audit;
  for (int i = 0; i < proof.path_node_size(); ++i) {
    json_audit.AddBase64(proof.path_node(i));
  }

  JsonObject json_reply;
  json_reply.Add("leaf_index", proof.leaf_index());
  json_reply.Add("audit_path", json_audit);

  SendJsonReply(event_base_, req, HTTP_OK, json_reply);
}


void HttpHandler::GetSTH(evhttp_request* req) const {
  if (evhttp_request_get_command(req) != EVHTTP_REQ_GET) {
    return SendJsonError(event_base_, req, HTTP_BADMETHOD,
                         "Method not allowed.");
  }

  const SignedTreeHead& sth(log_lookup_->GetSTH());

  VLOG(2) << "SignedTreeHead:\n" << sth.DebugString();

  JsonObject json_reply;
  json_reply.Add("tree_size", sth.tree_size());
  json_reply.Add("timestamp", sth.timestamp());
  json_reply.AddBase64("sha256_root_hash", sth.sha256_root_hash());
  json_reply.Add("tree_head_signature", sth.signature());

  VLOG(2) << "GetSTH:\n" << json_reply.DebugString();

  SendJsonReply(event_base_, req, HTTP_OK, json_reply);
}


void HttpHandler::GetConsistency(evhttp_request* req) const {
  if (evhttp_request_get_command(req) != EVHTTP_REQ_GET) {
    return SendJsonError(event_base_, req, HTTP_BADMETHOD,
                         "Method not allowed.");
  }

  const libevent::QueryParams query(libevent::ParseQuery(req));

  const int64_t first(libevent::GetIntParam(query, "first"));
  if (first < 0) {
    return SendJsonError(event_base_, req, HTTP_BADREQUEST,
                         "Missing or invalid \"first\" parameter.");
  }

  const int64_t second(libevent::GetIntParam(query, "second"));
  if (second < first) {
    return SendJsonError(event_base_, req, HTTP_BADREQUEST,
                         "Missing or invalid \"second\" parameter.");
  }

  const vector<string> consistency(
      log_lookup_->ConsistencyProof(first, second));
  JsonArray json_cons;
  for (vector<string>::const_iterator it = consistency.begin();
       it != consistency.end(); ++it) {
    json_cons.AddBase64(*it);
  }

  JsonObject json_reply;
  json_reply.Add("consistency", json_cons);

  SendJsonReply(event_base_, req, HTTP_OK, json_reply);
}


void HttpHandler::BlockingGetEntries(evhttp_request* req, int64_t start,
                                     int64_t end, bool include_scts) const {
  JsonArray json_entries;
  auto it(db_->ScanEntries(start));
  for (int64_t i = start; i <= end; ++i) {
    LoggedEntry entry;

    if (!it->GetNextEntry(&entry) || entry.sequence_number() != i) {
      break;
    }

    string leaf_input;
    string extra_data;
    string sct_data;
    if (!entry.SerializeForLeaf(&leaf_input) ||
        !entry.SerializeExtraData(&extra_data) ||
        (include_scts &&
         Serializer::SerializeSCT(entry.sct(), &sct_data) !=
             SerializeResult::OK)) {
      LOG(WARNING) << "Failed to serialize entry @ " << i << ":\n"
                   << entry.DebugString();
      return SendJsonError(event_base_, req, HTTP_INTERNAL,
                           "Serialization failed.");
    }

    JsonObject json_entry;
    json_entry.AddBase64("leaf_input", leaf_input);
    json_entry.AddBase64("extra_data", extra_data);

    if (include_scts) {
      // This is non-standard, and currently only used by other SuperDuper log
      // nodes when "following" to fetch data from each other:
      json_entry.AddBase64("sct", sct_data);
    }

    json_entries.Add(&json_entry);
  }

  if (json_entries.Length() < 1) {
    return SendJsonError(event_base_, req, HTTP_BADREQUEST,
                         "Entry not found.");
  }

  JsonObject json_reply;
  json_reply.Add("entries", json_entries);

  SendJsonReply(event_base_, req, HTTP_OK, json_reply);
}
