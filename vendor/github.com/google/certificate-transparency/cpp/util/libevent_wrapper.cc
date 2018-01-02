#include "config.h"
#include "util/libevent_wrapper.h"


#include <arpa/inet.h>
#ifdef HAVE_ARPA_NAMESER_H
#include <arpa/nameser.h> /* DNS HEADER struct */
#endif
#include <event2/keyvalq_struct.h>
#include <event2/thread.h>
#include <evhtp.h>
#include <glog/logging.h>
#include <math.h>
#include <climits>
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#include <netdb.h>
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h> /* inet_ functions / structs */
#endif
#include <resolv.h>
#include <sys/socket.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#include <signal.h>

using std::bind;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
using std::function;
using std::lock_guard;
using std::make_pair;
using std::multimap;
using std::mutex;
using std::placeholders::_1;
using std::recursive_mutex;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;
using util::TaskHold;

namespace {

void FreeEvDns(evdns_base* dns) {
  if (dns) {
    evdns_base_free(dns, true);
  }
}


static void Handler_ExitLoop(evutil_socket_t, short, void* base) {
  event_base_loopexit((event_base*)base, NULL);
}


void SetExitLoopHandler(event_base* base, int signum) {
  struct event* signal_event;
  // TODO(pphaneuf): this should be free'd
  signal_event = evsignal_new(base, signum, Handler_ExitLoop, base);
  CHECK_NOTNULL(signal_event);
  CHECK_GE(event_add(signal_event, NULL), 0);
}


void DelayCancel(event* timer, util::Task* task) {
  event_del(timer);
  task->Return(util::Status::CANCELLED);
}


void DelayDispatch(evutil_socket_t, short, void* userdata) {
  static_cast<util::Task*>(CHECK_NOTNULL(userdata))->Return();
}


#ifdef HAVE_THREAD_LOCAL
thread_local bool on_event_thread = false;
#elif HAVE___THREAD
__thread bool on_event_thread = false;
#else
#error No suitable thread local storage available
#endif


}  // namespace

namespace cert_trans {
namespace libevent {


struct HttpServer::Handler {
  Handler(const string& _path, const HandlerCallback& _cb)
      : path(_path), cb(_cb) {
  }

  const string path;
  const HandlerCallback cb;
};


class ResolverImpl : public Base::Resolver {
 public:
  string Resolve(const string& host) override {
    struct addrinfo* info;
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    // It seems evhtp doesn't support IPv6 addresses because it uses
    // inet_addr() to parse the passed in "stringified" address.
    // Restrict to IPv4 addresses for now:
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    const int resolved(getaddrinfo(host.c_str(), AF_UNSPEC, &hints, &info));
    if (resolved != 0) {
      LOG(WARNING) << "Failed to resolve HTTPS hostname " << host << ": "
                   << gai_strerror(resolved);
      return nullptr;
    }

    struct addrinfo* res(info);
    void* addr(nullptr);
    while (res && !addr) {
      switch (res->ai_family) {
        case AF_INET:
          addr =
              &reinterpret_cast<struct sockaddr_in*>(res->ai_addr)->sin_addr;
          break;
        case AF_INET6:
          // Just in case one day evhtp uses inet_pton()
          addr =
              &reinterpret_cast<struct sockaddr_in6*>(res->ai_addr)->sin6_addr;
          break;
        default:
          res = res->ai_next;
          break;
      }
    }

    if (!addr) {
      LOG(WARNING) << "Got no usable address for " << host;
      return nullptr;
    }

    char addr_str[INET6_ADDRSTRLEN];
    inet_ntop(res->ai_family, addr, addr_str, INET6_ADDRSTRLEN);
    freeaddrinfo(info);
    return string(addr_str);
  }
};


Base::Base() : Base(unique_ptr<Resolver>(new ResolverImpl)) {
}


Base::Base(unique_ptr<Resolver> resolver)
    : base_(CHECK_NOTNULL(event_base_new()), event_base_free),
      dns_(nullptr, FreeEvDns),
      wake_closures_(event_new(base_.get(), -1, 0, &Base::RunClosures, this),
                     &event_free),
      resolver_(std::move(resolver)) {
  evthread_make_base_notifiable(base_.get());

  // So much stuff breaks if there's not a Dns client around to keep the
  // event loop doing stuff that we may as well just have one from the get go.
  GetDns();
}


Base::~Base() {
}


// static
bool Base::OnEventThread() {
  return on_event_thread;
}


// static
void Base::CheckNotOnEventThread() {
  CHECK_EQ(false, OnEventThread());
}


void Base::Add(const function<void()>& cb) {
  lock_guard<mutex> lock(closures_lock_);
  closures_.push_back(cb);
  event_active(wake_closures_.get(), 0, 0);
}


void Base::Delay(const duration<double>& delay, util::Task* task) {
  // If the delay is zero (or less?), what the heck, we're done!
  if (delay <= delay.zero()) {
    task->Return();
    return;
  }

  // Make sure nothing "bad" happens while we're still setting up our
  // callbacks.
  TaskHold hold(task);

  event* timer(CHECK_NOTNULL(evtimer_new(base_.get(), &DelayDispatch, task)));

  // Ensure that the cancellation callback is run on this libevent::Base, to
  // avoid races during cancellation.

  // Cancellation callbacks are always called before the task enters
  // the DONE state (and "timer" is freed), and "event_del" is
  // thread-safe, so it does not matter on which thread "DelayCancel"
  // is called on.
  task->WhenCancelled(bind(DelayCancel, timer, task));

  task->CleanupWhenDone(bind(event_free, timer));

  timeval tv;
  const seconds sec(duration_cast<seconds>(delay));
  tv.tv_sec = sec.count();
  tv.tv_usec = duration_cast<microseconds>(delay - sec).count();

  CHECK_EQ(evtimer_add(timer, &tv), 0);
}


void Base::Dispatch() {
  SetExitLoopHandler(base_.get(), SIGHUP);
  SetExitLoopHandler(base_.get(), SIGINT);
  SetExitLoopHandler(base_.get(), SIGTERM);

  // There should /never/ be more than 1 thread trying to call Dispatch(), so
  // we should expect to always own the lock here.
  CHECK(dispatch_lock_.try_lock());
  LOG_IF(WARNING, on_event_thread)
      << "Huh?, Are you calling Dispatch() from a libevent thread?";
  const bool old_on_event_thread(on_event_thread);
  on_event_thread = true;
  CHECK_EQ(event_base_dispatch(base_.get()), 0);
  on_event_thread = old_on_event_thread;
  dispatch_lock_.unlock();
}


void Base::DispatchOnce() {
  // Only one thread can be running a dispatch loop at a time
  lock_guard<mutex> lock(dispatch_lock_);
  LOG_IF(WARNING, on_event_thread)
      << "Huh?, Are you calling Dispatch() from a libevent thread?";
  const bool old_on_event_thread(on_event_thread);
  on_event_thread = true;
  CHECK_EQ(event_base_loop(base_.get(), EVLOOP_ONCE), 0);
  on_event_thread = old_on_event_thread;
}


void Base::LoopExit() {
  event_base_loopexit(base_.get(), nullptr);
}


event* Base::EventNew(evutil_socket_t& sock, short events,
                      Event* event) const {
  return CHECK_NOTNULL(
      event_new(base_.get(), sock, events, &Event::Dispatch, event));
}


evhttp* Base::HttpNew() const {
  return CHECK_NOTNULL(evhttp_new(base_.get()));
}


evdns_base* Base::GetDns() {
  lock_guard<mutex> lock(dns_lock_);

  if (!dns_) {
    dns_.reset(CHECK_NOTNULL(evdns_base_new(base_.get(), 1)));
  }

  return dns_.get();
}


evhtp_connection_t* Base::HttpConnectionNew(const string& host,
                                            unsigned short port) {
  return CHECK_NOTNULL(
      evhtp_connection_new_dns(base_.get(), GetDns(), host.c_str(), port));
}


evhtp_connection_t* Base::HttpsConnectionNew(const string& host,
                                             unsigned short port,
                                             SSL_CTX* ssl_ctx) {
  CHECK_NOTNULL(ssl_ctx);

  // TODO(alcutter): remove this all temporary name resolution stuff when this
  // PR is merged: https://github.com/ellzey/libevhtp/pull/163
  const string addr_str(resolver_->Resolve(host));
  VLOG(1) << "Got addr: " << addr_str << ":" << port;
  evhtp_connection_t* ret(CHECK_NOTNULL(
      evhtp_connection_ssl_new(base_.get(), addr_str.c_str(), port, ssl_ctx)));
  return ret;
}


void Base::RunClosures(evutil_socket_t, short, void* userdata) {
  Base* self(static_cast<Base*>(CHECK_NOTNULL(userdata)));

  vector<function<void()>> closures;
  {
    lock_guard<mutex> lock(self->closures_lock_);
    closures.swap(self->closures_);
  }

  for (const auto& closure : closures) {
    closure();
  }
}


Event::Event(const Base& base, evutil_socket_t sock, short events,
             const Callback& cb)
    : cb_(cb), ev_(base.EventNew(sock, events, this)) {
}


Event::~Event() {
  event_free(ev_);
}


void Event::Add(const duration<double>& timeout) const {
  timeval tv;
  timeval* tvp(NULL);

  if (timeout != duration<double>::zero()) {
    const seconds sec(duration_cast<seconds>(timeout));
    tv.tv_sec = sec.count();
    tv.tv_usec = duration_cast<microseconds>(timeout - sec).count();
    tvp = &tv;
  }

  CHECK_EQ(event_add(ev_, tvp), 0);
}


void Event::Dispatch(evutil_socket_t sock, short events, void* userdata) {
  static_cast<Event*>(userdata)->cb_(sock, events);
}


HttpServer::HttpServer(const Base& base) : http_(base.HttpNew()) {
}


HttpServer::~HttpServer() {
  evhttp_free(http_);
  for (vector<Handler*>::iterator it = handlers_.begin();
       it != handlers_.end(); ++it) {
    delete *it;
  }
}


void HttpServer::Bind(const char* address, ev_uint16_t port) {
  CHECK_EQ(evhttp_bind_socket(http_, address, port), 0);
}


bool HttpServer::AddHandler(const string& path, const HandlerCallback& cb) {
  Handler* handler(new Handler(path, cb));
  handlers_.push_back(handler);

  return evhttp_set_cb(http_, path.c_str(), &HandleRequest, handler) == 0;
}


void HttpServer::HandleRequest(evhttp_request* req, void* userdata) {
  static_cast<Handler*>(userdata)->cb(req);
}


QueryParams ParseQuery(evhttp_request* req) {
  evkeyvalq keyval;
  QueryParams retval;

  // We return an empty result in case of a parsing error.
  if (evhttp_parse_query_str(evhttp_uri_get_query(
                                 evhttp_request_get_evhttp_uri(req)),
                             &keyval) == 0) {
    for (evkeyval* i = keyval.tqh_first; i; i = i->next.tqe_next) {
      retval.insert(make_pair(i->key, i->value));
    }
  }

  return retval;
}


bool GetParam(const QueryParams& query, const string& param, string* value) {
  CHECK_NOTNULL(value);

  auto it = query.find(param);
  if (it == query.end()) {
    return false;
  }

  const string possible_value(it->second);
  ++it;

  // Flag duplicate query parameters as invalid.
  const bool retval(it == query.end() || it->first != param);
  if (retval) {
    *value = possible_value;
  }

  return retval;
}


// Returns -1 on error, and on success too if the parameter contains
// -1 (so it's advised to only use it when expecting unsigned
// parameters).
int64_t GetIntParam(const QueryParams& query, const string& param) {
  int retval(-1);
  string value;
  if (GetParam(query, param, &value)) {
    errno = 0;
    const long num(strtol(value.c_str(), /*endptr*/ NULL, 10));
    // Detect strtol() errors or overflow/underflow when casting to
    // retval's type clips the value. We do the following by doing it,
    // and checking that they're still equal afterward (this will
    // still work if we change retval's type later on).
    retval = num;
    if (errno || static_cast<long>(retval) != num) {
      VLOG(1) << "over/underflow getting \"" << param << "\": " << retval
              << ", " << num << " (" << strerror(errno) << ")";
      retval = -1;
    }
  }

  return retval;
}


bool GetBoolParam(const QueryParams& query, const string& param) {
  string value;
  if (GetParam(query, param, &value)) {
    return (value == "true");
  } else {
    return false;
  }
}


EventPumpThread::EventPumpThread(const shared_ptr<Base>& base)
    : base_(base), pump_thread_(bind(&EventPumpThread::Pump, this)) {
}


EventPumpThread::~EventPumpThread() {
  base_->LoopExit();
  pump_thread_.join();
}


void EventPumpThread::Pump() {
  base_->Dispatch();
}


}  // namespace libevent
}  // namespace cert_trans
