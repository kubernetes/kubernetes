#ifndef CERT_TRANS_UTIL_LIBEVENT_WRAPPER_H_
#define CERT_TRANS_UTIL_LIBEVENT_WRAPPER_H_

#include <event2/dns.h>
#include <event2/event.h>
#include <atomic>
#include <chrono>
// TODO(alcutter): Use evhtp for the HttpServer too.
#include <event2/http.h>
#include <evhtp.h>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "base/macros.h"
#include "util/executor.h"
#include "util/task.h"

namespace cert_trans {
namespace libevent {


class Event;


class Base : public util::Executor {
 public:
  class Resolver {
   public:
    virtual std::string Resolve(const std::string& host) = 0;
  };

  static bool OnEventThread();
  static void CheckNotOnEventThread();

  Base();
  Base(std::unique_ptr<Resolver> resolver);
  ~Base();

  // Arranges to run the closure on the main loop.
  void Add(const std::function<void()>& cb) override;

  void Delay(const std::chrono::duration<double>& delay,
             util::Task* task) override;

  void Dispatch();
  void DispatchOnce();
  void LoopExit();

  event* EventNew(evutil_socket_t& sock, short events, Event* event) const;
  evhttp* HttpNew() const;
  evdns_base* GetDns();
  evhtp_connection_t* HttpConnectionNew(const std::string& host,
                                        unsigned short port);
  evhtp_connection_t* HttpsConnectionNew(const std::string& host,
                                         unsigned short port,
                                         SSL_CTX* ssl_ctx);

 private:
  static void RunClosures(evutil_socket_t sock, short flag, void* userdata);

  const std::unique_ptr<event_base, void (*)(event_base*)> base_;
  std::mutex dispatch_lock_;

  std::mutex dns_lock_;
  // "dns_" should be after base_, so that it gets destroyed first.
  std::unique_ptr<evdns_base, void (*)(evdns_base*)> dns_;

  std::mutex closures_lock_;
  // "wake_closures_" should be after base_, so that it gets destroyed
  // first.
  const std::unique_ptr<event, void (*)(event*)> wake_closures_;
  std::vector<std::function<void()>> closures_;
  std::unique_ptr<Resolver> resolver_;

  DISALLOW_COPY_AND_ASSIGN(Base);
};


class Event {
 public:
  typedef std::function<void(evutil_socket_t, short)> Callback;

  Event(const Base& base, evutil_socket_t sock, short events,
        const Callback& cb);
  ~Event();

  void Add(const std::chrono::duration<double>& timeout) const;
  // Note that this is only public so |Base| can use it.
  static void Dispatch(evutil_socket_t sock, short events, void* userdata);

 private:
  const Callback cb_;
  event* const ev_;

  DISALLOW_COPY_AND_ASSIGN(Event);
};


class HttpServer {
 public:
  typedef std::function<void(evhttp_request*)> HandlerCallback;

  explicit HttpServer(const Base& base);
  ~HttpServer();

  void Bind(const char* address, ev_uint16_t port);

  // Returns false if there was an error adding the handler.
  bool AddHandler(const std::string& path, const HandlerCallback& cb);

 private:
  struct Handler;

  static void HandleRequest(evhttp_request* req, void* userdata);

  evhttp* const http_;
  // Could have been a vector<Handler>, but it is important that
  // pointers to entries remain valid.
  std::vector<Handler*> handlers_;

  DISALLOW_COPY_AND_ASSIGN(HttpServer);
};

typedef std::multimap<std::string, std::string> QueryParams;

QueryParams ParseQuery(evhttp_request* req);

bool GetParam(const QueryParams& query, const std::string& param,
              std::string* value);

int64_t GetIntParam(const QueryParams& query, const std::string& param);

bool GetBoolParam(const QueryParams& query, const std::string& param);


class EventPumpThread {
 public:
  EventPumpThread(const std::shared_ptr<Base>& base);
  ~EventPumpThread();

 private:
  void Pump();

  const std::shared_ptr<Base> base_;
  std::thread pump_thread_;

  DISALLOW_COPY_AND_ASSIGN(EventPumpThread);
};


}  // namespace libevent
}  // namespace cert_trans

#endif  // CERT_TRANS_UTIL_LIBEVENT_WRAPPER_H_
