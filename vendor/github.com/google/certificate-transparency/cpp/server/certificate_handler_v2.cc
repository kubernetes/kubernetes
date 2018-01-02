#include <functional>

#include "log/frontend.h"
#include "server/certificate_handler_v2.h"
#include "server/json_output.h"
#include "util/json_wrapper.h"
#include "util/status.h"
#include "util/thread_pool.h"

namespace cert_trans {

using ct::LogEntry;
using ct::SignedCertificateTimestamp;
using std::bind;
using std::make_shared;
using std::multimap;
using std::placeholders::_1;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using util::Status;


CertificateHttpHandlerV2::CertificateHttpHandlerV2(
    LogLookup* log_lookup, const ReadOnlyDatabase* db,
    const ClusterStateController<LoggedEntry>* controller,
    const CertChecker* cert_checker, Frontend* frontend, ThreadPool* pool,
    libevent::Base* event_base, StalenessTracker* staleness_tracker)
    : HttpHandlerV2(log_lookup, db, controller, pool, event_base,
                    staleness_tracker),
      cert_checker_(CHECK_NOTNULL(cert_checker)),
      submission_handler_(cert_checker_),
      frontend_(frontend) {
}


void CertificateHttpHandlerV2::AddHandlers(libevent::HttpServer* server) {
  // TODO(alcutter): Support this for mirrors too
  if (cert_checker_) {
    // Don't really need to proxy this one, but may as well just to keep
    // everything tidy:
    AddProxyWrappedHandler(server, "/ct/v2/get-roots",
                           bind(&CertificateHttpHandlerV2::GetRoots, this,
                                _1));
  }
  if (frontend_) {
    // Proxy the add-* calls too, technically we could serve them, but a
    // more up-to-date node will have a better chance of handling dupes
    // correctly, rather than bloating the tree.
    AddProxyWrappedHandler(server, "/ct/v2/add-chain",
                           bind(&CertificateHttpHandlerV2::AddChain, this,
                                _1));
    AddProxyWrappedHandler(server, "/ct/v2/add-pre-chain",
                           bind(&CertificateHttpHandlerV2::AddPreChain, this,
                                _1));
  }
}


void CertificateHttpHandlerV2::GetRoots(evhttp_request* req) const {
  return SendJsonError(event_base_, req, HTTP_NOTIMPLEMENTED,
                       "Not yet implemented.");
}


void CertificateHttpHandlerV2::AddChain(evhttp_request* req) {
  return SendJsonError(event_base_, req, HTTP_NOTIMPLEMENTED,
                       "Not yet implemented.");
}


void CertificateHttpHandlerV2::AddPreChain(evhttp_request* req) {
  return SendJsonError(event_base_, req, HTTP_NOTIMPLEMENTED,
                       "Not yet implemented.");
}


void CertificateHttpHandlerV2::BlockingAddChain(
    evhttp_request* req, const shared_ptr<CertChain>& chain) const {
  return SendJsonError(event_base_, req, HTTP_NOTIMPLEMENTED,
                       "Not yet implemented.");
}


void CertificateHttpHandlerV2::BlockingAddPreChain(
    evhttp_request* req, const shared_ptr<PreCertChain>& chain) const {
  return SendJsonError(event_base_, req, HTTP_NOTIMPLEMENTED,
                       "Not yet implemented.");
}


}  // namespace cert_trans
