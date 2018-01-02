#ifndef CERT_TRANS_SERVER_CERTIFICATE_HANDLER_H_
#define CERT_TRANS_SERVER_CERTIFICATE_HANDLER_H_

#include "log/cert_submission_handler.h"
#include "log/database.h"
#include "log/logged_entry.h"
#include "server/handler.h"
#include "server/staleness_tracker.h"

namespace cert_trans {


class CertificateHttpHandler : public HttpHandler {
 public:
  // Does not take ownership of its parameters, which must outlive
  // this instance. The |frontend| and |cert_checker| parameters can be NULL,
  // in which case this server will not accept "add-chain" and "add-pre-chain"
  // requests.
  CertificateHttpHandler(LogLookup* log_lookup, const ReadOnlyDatabase* db,
                         const ClusterStateController<LoggedEntry>* controller,
                         const CertChecker* cert_checker, Frontend* frontend,
                         ThreadPool* pool, libevent::Base* event_base,
                         StalenessTracker* staleness_tracker);

  ~CertificateHttpHandler() = default;

 protected:
  void AddHandlers(libevent::HttpServer* server) override;

 private:
  const CertChecker* const cert_checker_;
  const std::unique_ptr<CertSubmissionHandler> submission_handler_;
  Frontend* const frontend_;

  void GetRoots(evhttp_request* req) const;
  void AddChain(evhttp_request* req);
  void AddPreChain(evhttp_request* req);

  void BlockingAddChain(evhttp_request* req,
                        const std::shared_ptr<CertChain>& chain) const;
  void BlockingAddPreChain(evhttp_request* req,
                           const std::shared_ptr<PreCertChain>& chain) const;

  DISALLOW_COPY_AND_ASSIGN(CertificateHttpHandler);
};


}  // namespace cert_trans


#endif  // CERT_TRANS_SERVER_CERTIFICATE_HANDLER_H_
