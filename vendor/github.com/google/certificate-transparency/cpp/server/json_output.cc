#include "server/json_output.h"

#include <glog/logging.h>
#include <string>

#include "monitoring/latency.h"
#include "monitoring/monitoring.h"
#include "util/json_wrapper.h"
#include "util/libevent_wrapper.h"

using std::string;

namespace cert_trans {
namespace {


static Counter<string>* total_http_server_requests(
    Counter<string>::New("total_http_server_requests", "path",
                         "Total number of HTTP requests received for a given "
                         "path."));
static Counter<string, int>* total_http_server_response_codes(
    Counter<string, int>::New("total_http_server_response_codes", "path",
                              "response_code",
                              "Total number of responses sent with a given "
                              "HTTP response code for a given path."));

static const char kJsonContentType[] = "application/json; charset=utf-8";


string LogRequest(evhttp_request* req, int http_status, int resp_body_length) {
  evhttp_connection* conn = evhttp_request_get_connection(req);
  char* peer_addr;
  ev_uint16_t peer_port;
  evhttp_connection_get_peer(conn, &peer_addr, &peer_port);

  string http_verb;
  switch (evhttp_request_get_command(req)) {
    case EVHTTP_REQ_DELETE:
      http_verb = "DELETE";
      break;
    case EVHTTP_REQ_GET:
      http_verb = "GET";
      break;
    case EVHTTP_REQ_HEAD:
      http_verb = "HEAD";
      break;
    case EVHTTP_REQ_POST:
      http_verb = "POST";
      break;
    case EVHTTP_REQ_PUT:
      http_verb = "PUT";
      break;
    default:
      http_verb = "UNKNOWN";
      break;
  }

  const string path(evhttp_uri_get_path(evhttp_request_get_evhttp_uri(req)));
  total_http_server_requests->Increment(path);
  total_http_server_response_codes->Increment(path, http_status);

  const string uri(evhttp_request_get_uri(req));
  return string(peer_addr) + " \"" + http_verb + " " + uri + "\" " +
         std::to_string(http_status) + " " + std::to_string(resp_body_length);
}


}  // namespace


void SendJsonReply(libevent::Base* base, evhttp_request* req, int http_status,
                   const JsonObject& json) {
  CHECK_NOTNULL(base);
  CHECK_NOTNULL(req);
  CHECK_EQ(evhttp_add_header(evhttp_request_get_output_headers(req),
                             "Content-Type", kJsonContentType),
           0);
  if (http_status == HTTP_SERVUNAVAIL) {
    CHECK_EQ(evhttp_add_header(evhttp_request_get_output_headers(req),
                               "Retry-After", "10"),
             0);
  }
  const string resp_body(json.ToString());
  CHECK_GT(evbuffer_add_printf(evhttp_request_get_output_buffer(req), "%s",
                               resp_body.c_str()),
           0);

  const string logstr(LogRequest(req, http_status, resp_body.size()));
  const auto send_reply([req, http_status, logstr]() {
    evhttp_send_reply(req, http_status, /*reason*/ NULL, /*databuf*/ NULL);

    VLOG(1) << logstr;
  });

  if (!libevent::Base::OnEventThread()) {
    base->Add(send_reply);
  } else {
    send_reply();
  }
}


void SendJsonError(libevent::Base* base, evhttp_request* req, int http_status,
                   const string& error_msg) {
  JsonObject json_reply;
  json_reply.Add("error_message", error_msg);
  json_reply.AddBoolean("success", false);

  SendJsonReply(base, req, http_status, json_reply);
}


}  // namespace cert_trans
