#ifndef CERT_TRANS_SERVER_HANDLER_V2_H_
#define CERT_TRANS_SERVER_HANDLER_V2_H_

#include <stdint.h>
#include <memory>
#include <mutex>
#include <string>

#include "proto/ct.pb.h"
#include "server/staleness_tracker.h"
#include "util/libevent_wrapper.h"
#include "util/sync_task.h"
#include "util/task.h"

class Frontend;

namespace cert_trans {

class CertChain;
class CertChecker;
template <class T>
class ClusterStateController;
class LogLookup;
class LoggedEntry;
class PreCertChain;
class Proxy;
class ReadOnlyDatabase;
class ThreadPool;


class HttpHandlerV2 {
 public:
  // Does not take ownership of its parameters, which must outlive
  // this instance.
  HttpHandlerV2(LogLookup* log_lookup, const ReadOnlyDatabase* db,
                const ClusterStateController<LoggedEntry>* controller,
                ThreadPool* pool, libevent::Base* event_base,
                StalenessTracker* staleness_tracker);
  virtual ~HttpHandlerV2();

  void Add(libevent::HttpServer* server);

  void SetProxy(Proxy* proxy);

 protected:
  // Implemented by subclasses which want to add their own extra http handlers.
  virtual void AddHandlers(libevent::HttpServer* server) = 0;

  void AddEntryReply(evhttp_request* req, const util::Status& add_status,
                     const ct::SignedCertificateTimestamp& sct) const;

  void ProxyInterceptor(
      const libevent::HttpServer::HandlerCallback& local_handler,
      evhttp_request* request);

  void AddProxyWrappedHandler(
      libevent::HttpServer* server, const std::string& path,
      const libevent::HttpServer::HandlerCallback& local_handler);

  void GetEntries(evhttp_request* req) const;
  void GetProof(evhttp_request* req) const;
  void GetSTH(evhttp_request* req) const;
  void GetConsistency(evhttp_request* req) const;

  void BlockingGetEntries(evhttp_request* req, int64_t start, int64_t end,
                          bool include_scts) const;

  LogLookup* const log_lookup_;
  const ReadOnlyDatabase* const db_;
  const ClusterStateController<LoggedEntry>* const controller_;
  Proxy* proxy_;
  ThreadPool* const pool_;
  libevent::Base* const event_base_;
  StalenessTracker* const staleness_tracker_;

  DISALLOW_COPY_AND_ASSIGN(HttpHandlerV2);
};


}  // namespace cert_trans

#endif  // CERT_TRANS_SERVER_HANDLER_V2_H_
