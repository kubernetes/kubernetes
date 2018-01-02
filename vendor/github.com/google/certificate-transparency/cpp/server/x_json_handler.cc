#include "server/x_json_handler.h"

#include <functional>

#include "log/frontend.h"
#include "server/json_output.h"
#include "util/statusor.h"
#include "util/thread_pool.h"

namespace cert_trans {

using ct::LogEntry;
using ct::SignedCertificateTimestamp;
using ct::X_JSON_ENTRY;
using std::bind;
using std::make_shared;
using std::move;
using std::multimap;
using std::placeholders::_1;
using std::shared_ptr;
using std::string;
using util::Status;


namespace {


shared_ptr<JsonObject> ExtractJson(libevent::Base* base, evhttp_request* req) {
  CHECK_NOTNULL(base);
  CHECK_NOTNULL(req);
  if (evhttp_request_get_command(req) != EVHTTP_REQ_POST) {
    SendJsonError(base, req, HTTP_BADMETHOD, "Method not allowed.");
    return nullptr;
  }

  // TODO(pphaneuf): Should we check that Content-Type says
  // "application/json", as recommended by RFC4627?
  shared_ptr<JsonObject> json_body(
      make_shared<JsonObject>(evhttp_request_get_input_buffer(req)));
  if (!json_body->Ok() || !json_body->IsType(json_type_object)) {
    SendJsonError(base, req, HTTP_BADREQUEST,
                  "Unable to parse provided JSON.");
    return nullptr;
  }

  VLOG(2) << "ExtractJson:\n" << json_body->DebugString();
  return json_body;
}


}  // namespace


XJsonHttpHandler::XJsonHttpHandler(
    LogLookup* log_lookup, const ReadOnlyDatabase* db,
    const ClusterStateController<LoggedEntry>* controller, Frontend* frontend,
    ThreadPool* pool, libevent::Base* event_base,
    StalenessTracker* staleness_tracker)
    : HttpHandler(log_lookup, db, controller, pool, event_base,
                  staleness_tracker),
      frontend_(frontend) {
}


void XJsonHttpHandler::AddHandlers(libevent::HttpServer* server) {
  if (frontend_) {
    // Proxy the add-* calls too, technically we could serve them, but a
    // more up-to-date node will have a better chance of handling dupes
    // correctly, rather than bloating the tree.
    AddProxyWrappedHandler(server, "/ct/v1/add-json",
                           bind(&XJsonHttpHandler::AddJson, this, _1));
  }
}


void XJsonHttpHandler::AddJson(evhttp_request* req) {
  shared_ptr<JsonObject> json(ExtractJson(event_base_, req));
  if (!json) {
    return;
  }

  pool_->Add(bind(&XJsonHttpHandler::BlockingAddJson, this, req, json));
}


void XJsonHttpHandler::BlockingAddJson(evhttp_request* req,
                                       shared_ptr<JsonObject> json) const {
  SignedCertificateTimestamp sct;

  LogEntry entry;
  // do this here for now
  entry.set_type(X_JSON_ENTRY);
  entry.mutable_x_json_entry()->set_json(json->ToString());

  AddEntryReply(req, CHECK_NOTNULL(frontend_)
                         ->QueueProcessedEntry(Status::OK, entry, &sct),
                sct);
}


}  // namespace cert_trans
