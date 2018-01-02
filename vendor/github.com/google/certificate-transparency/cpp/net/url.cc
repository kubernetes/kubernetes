#include "net/url.h"

#include <event2/http.h>
#include <glog/logging.h>
#include <memory>

using std::string;
using std::unique_ptr;

namespace cert_trans {

namespace {


void StringFromCharPtr(string* out, const char* in) {
  if (in) {
    *out = in;
  }
}


}  // namespace


URL::URL(const string& url) {
  VLOG(1) << "parsing URL: " << url;

  unique_ptr<evhttp_uri, void (*)(evhttp_uri*)> uri(
      evhttp_uri_parse(url.c_str()), &evhttp_uri_free);

  if (!uri) {
    LOG(FATAL) << "URL invalid: " << url;
  }

  const int port(evhttp_uri_get_port(uri.get()));
  port_ = port > 0 && port <= UINT16_MAX ? port : 0;

  StringFromCharPtr(&protocol_, evhttp_uri_get_scheme(uri.get()));
  StringFromCharPtr(&host_, evhttp_uri_get_host(uri.get()));
  StringFromCharPtr(&path_, evhttp_uri_get_path(uri.get()));
  StringFromCharPtr(&query_, evhttp_uri_get_query(uri.get()));
}


string URL::PathQuery() const {
  string retval(path_);

  if (!query_.empty()) {
    retval.append("?");
    retval.append(query_);
  }

  return retval;
}


std::ostream& operator<<(std::ostream& out, const URL& url) {
  out << url.Protocol() << "://" << url.Host();
  if (url.Port() > 0) {
    out << ":" << url.Port();
  }
  out << url.PathQuery();
  return out;
}


}  // namespace cert_trans
