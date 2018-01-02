#ifndef CERT_TRANS_SERVER_JSON_OUTPUT_H_
#define CERT_TRANS_SERVER_JSON_OUTPUT_H_

#include <string>

struct evhttp_request;
class JsonObject;

namespace cert_trans {
namespace libevent {
class Base;
}  // namespace libevent


void SendJsonReply(libevent::Base* base, evhttp_request* req, int http_status,
                   const JsonObject& json);


void SendJsonError(libevent::Base* base, evhttp_request* req, int http_status,
                   const std::string& error_msg);


}  // namespace cert_trans


#endif  // CERT_TRANS_SERVER_JSON_OUTPUT_H_
