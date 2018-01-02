#include "net/connection_pool.h"

#include <event2/event.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>

#include "monitoring/monitoring.h"
#include "util/openssl_util.h"

extern "C" {
#include "third_party/curl/hostcheck.h"
#include "third_party/isec_partners/openssl_hostname_validation.h"
}  // extern "C"


using std::bind;
using std::chrono::duration_cast;
using std::chrono::seconds;
using std::chrono::system_clock;
using std::lock_guard;
using std::make_pair;
using std::map;
using std::move;
using std::mutex;
using std::pair;
using std::placeholders::_1;
using std::placeholders::_2;
using std::string;
using std::to_string;
using std::unique_lock;
using std::unique_ptr;
using std::shared_ptr;
using util::ClearOpenSSLErrors;
using util::DumpOpenSSLErrorStack;

DEFINE_int32(
    connection_read_timeout_seconds, 60,
    "Connection read timeout in seconds, only applies while willing to read.");
DEFINE_int32(connection_write_timeout_seconds, 60,
             "Connection write timeout in seconds, only applies while willing "
             "to write.");
DEFINE_int32(
    connection_pool_max_unused_age_seconds, 60 * 5,
    "When there are more than --url_fetcher_max_conn_per_host_port "
    "connections per host:port pair, any unused for at least this long will "
    "be removed.");
DEFINE_string(trusted_root_certs, "",
              "Location of trusted CA root certs for outgoing SSL "
              "connections.");
DEFINE_int32(url_fetcher_max_conn_per_host_port, 4,
             "maximum number of URL fetcher connections per host:port");

DEFINE_string(tls_client_minimum_protocol, "tlsv12",
              "Minimum acceptable TLS "
              "version protocol (tlsv1, tlsv11, tlsv12)");

DEFINE_string(
    tls_client_ciphers,
    "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:"
    "ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:"
    "DHE-RSA-AES128-GCM-SHA256:DHE-DSS-AES128-GCM-SHA256:kEDH+AESGCM:"
    "ECDHE-RSA-AES128-SHA256:ECDHE-ECDSA-AES128-SHA256:ECDHE-RSA-AES128-SHA:"
    "ECDHE-ECDSA-AES128-SHA:ECDHE-RSA-AES256-SHA384:ECDHE-ECDSA-AES256-SHA384:"
    "ECDHE-RSA-AES256-SHA:ECDHE-ECDSA-AES256-SHA:DHE-RSA-AES128-SHA256:"
    "DHE-RSA-AES128-SHA:DHE-DSS-AES128-SHA256:DHE-RSA-AES256-SHA256:"
    "DHE-DSS-AES256-SHA:DHE-RSA-AES256-SHA:"
    "!aNULL:!eNULL:!EXPORT:!DES:!RC4:!3DES:!MD5:!PSK",
    "List of ciphers the client will accept, default is the Mozilla 'Modern "
    "compatibility' recommended list.");


namespace cert_trans {
namespace internal {


const int kZeroMillis = 0;


static Gauge<string>* connections_per_host_port(
    Gauge<string>::New("connections_per_host_port", "host_port",
                       "Number of cached connections port host:port"));


namespace {


int GetSSLConnectionIndex() {
  static const int ssl_connection_index(
      SSL_get_ex_new_index(0, nullptr, nullptr, nullptr, nullptr));
  return ssl_connection_index;
}


string HostPortString(const HostPortPair& pair) {
  return pair.first + ":" + to_string(pair.second);
}


}  // namespace


// This class wraps the evhtp_connection_t* and associated data which need to
// hang around for at least the lifetime that structure.
class EvConnection {
 public:
  // Called by OpenSSL to verify the hostname presented in the server cert.
  static int SSLVerifyCallback(int preverify_ok, X509_STORE_CTX* x509_ctx);

  // Called by libevhtp when it detects some kind of error with the connection.
  static evhtp_res ConnectionErrorHook(evhtp_connection_t* conn,
                                       evhtp_error_flags errtype, void* arg);

  // Called by libevhtp once it's finished with the connection and is about to
  // delete it.
  static evhtp_res ConnectionFinishedHook(evhtp_connection_t* conn, void* arg);

  EvConnection(evhtp_connection_t* conn, HostPortPair&& other_end)
      : ev_conn_(CHECK_NOTNULL(conn)),
        other_end_(move(other_end)),
        errored_(false) {
    if (ev_conn_->ssl) {
      SSL_set_ex_data(ev_conn_->ssl, GetSSLConnectionIndex(),
                      static_cast<void*>(this));
      SSL_set_tlsext_host_name(ev_conn_->ssl, other_end_.first.c_str());
    }
  }

  evhtp_connection_t* connection() const {
    return ev_conn_;
  }

  const HostPortPair& other_end() const {
    return other_end_;
  }

  void SetErrored() {
    lock_guard<mutex> lock(lock_);
    errored_ = true;
  }

  bool GetErrored() const {
    lock_guard<mutex> lock(lock_);
    return errored_;
  }

 private:
  // We never really own this, evhtp does, as it likes to remind us.
  evhtp_connection_t* ev_conn_;
  const HostPortPair other_end_;

  mutable std::mutex lock_;
  bool errored_;
};


// static
int EvConnection::SSLVerifyCallback(const int preverify_ok,
                                    X509_STORE_CTX* x509_ctx) {
  CHECK_NOTNULL(x509_ctx);
  X509* const server_cert(
      CHECK_NOTNULL(X509_STORE_CTX_get_current_cert(x509_ctx)));

  if (preverify_ok == 0) {
    const int err(X509_STORE_CTX_get_error(x509_ctx));
    char buf[256];
    X509_NAME_oneline(X509_get_subject_name(server_cert), buf, 256);

    LOG(WARNING) << "OpenSSL failed to verify cert for " << buf << ": "
                 << X509_verify_cert_error_string(err);
    return preverify_ok;
  }

  // Only do extra checks (i.e. hostname matching) for the end-entity cert.
  const int depth(X509_STORE_CTX_get_error_depth(x509_ctx));
  if (depth > 0) {
    return preverify_ok;
  }

  const SSL* const ssl(static_cast<SSL*>(CHECK_NOTNULL(
      X509_STORE_CTX_get_ex_data(x509_ctx,
                                 SSL_get_ex_data_X509_STORE_CTX_idx()))));
  const EvConnection* const connection(
      CHECK_NOTNULL(static_cast<const EvConnection*>(
          SSL_get_ex_data(ssl, GetSSLConnectionIndex()))));

  const HostnameValidationResult hostname_valid(
      validate_hostname(connection->other_end_.first.c_str(), server_cert));
  if (hostname_valid != MatchFound) {
    string error;
    switch (hostname_valid) {
      case MatchFound:
        LOG(FATAL) << "Shouldn't get here.";
        break;
      case MatchNotFound:
        error = "certificate doesn't match hostname";
        break;
      case NoSANPresent:
        // I don't think we should ever see this, should be handled inside
        // validate_hostname()
        error = "no SAN present";
        break;
      case MalformedCertificate:
        error = "certificate is malformed";
        break;
      case Error:
        error = "unknown error";
        break;
    }
    if (connection->ev_conn_->request) {
      connection->ev_conn_->request->status = kSSLErrorStatus;
    }
    LOG_EVERY_N(WARNING, 100)
        << "Failed to validate SSL certificate: " << error << " : "
        << DumpOpenSSLErrorStack();
    ClearOpenSSLErrors();
    return 0;
  }
  return 1;
}


ConnectionPool::Connection::Connection(const shared_ptr<EvConnection>& conn)
    : connection_(conn) {
}


evhtp_connection_t* ConnectionPool::Connection::connection() const {
  return connection_->connection();
}


bool ConnectionPool::Connection::GetErrored() const {
  return connection_->GetErrored();
}


const HostPortPair& ConnectionPool::Connection::other_end() const {
  return connection_->other_end();
}


namespace {

// No SSLv2 or SSLv3, thanks.
const int kSslOpMinVersionTls1 = SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3;
const int kSslOpMinVersionTls11 = kSslOpMinVersionTls1 | SSL_OP_NO_TLSv1;
const int kSslOpMinVersionTls12 = kSslOpMinVersionTls11 | SSL_OP_NO_TLSv1_1;


// Create an SSL_CTX which permits the TLS versions specified by flags.
// SSLv2 and SSLv3 are never supported.
SSL_CTX* CreateSSLCTXFromFlags() {
  SSL_CTX* ctx(SSL_CTX_new(SSLv23_client_method()));
  CHECK_NOTNULL(ctx);

  LOG(WARNING) << "Setting client minimum TLS protocol to "
               << FLAGS_tls_client_minimum_protocol;

  if (FLAGS_tls_client_minimum_protocol == "tlsv1") {
    SSL_CTX_set_options(ctx, kSslOpMinVersionTls1);
  } else if (FLAGS_tls_client_minimum_protocol == "tlsv11") {
    SSL_CTX_set_options(ctx, kSslOpMinVersionTls11);
  } else if (FLAGS_tls_client_minimum_protocol == "tlsv12") {
    SSL_CTX_set_options(ctx, kSslOpMinVersionTls12);
  } else {
    LOG(FATAL) << "Unrecognised TLS version: "
               << FLAGS_tls_client_minimum_protocol;
  }

  CHECK_EQ(1, SSL_CTX_set_cipher_list(ctx, FLAGS_tls_client_ciphers.c_str()));

  return ctx;
}


}  // namespace


ConnectionPool::ConnectionPool(libevent::Base* base)
    : base_(CHECK_NOTNULL(base)),
      cleanup_scheduled_(false),
      ssl_ctx_(CreateSSLCTXFromFlags(), SSL_CTX_free) {
  CHECK(ssl_ctx_) << "could not build SSL context: "
                  << DumpOpenSSLErrorStack();

  // Try to load trusted root certificates.
  if (FLAGS_trusted_root_certs == "") {
    LOG(INFO) << "Loading openssl default trusted root certificates...";
    if (SSL_CTX_set_default_verify_paths(ssl_ctx_.get()) != 1) {
      DumpOpenSSLErrorStack();
      LOG(FATAL) << "Couldn't load openssl default trusted root certificates.";
    }
  } else {
    LOG(INFO) << "Loading trusted root certificates from: "
              << FLAGS_trusted_root_certs << " ...";
    if (SSL_CTX_load_verify_locations(ssl_ctx_.get(),
                                      FLAGS_trusted_root_certs.c_str(),
                                      nullptr) != 1) {
      DumpOpenSSLErrorStack();
      LOG(FATAL) << "Couldn't load trusted root certificates: "
                 << FLAGS_trusted_root_certs;
    }
  }

  SSL_CTX_set_verify(ssl_ctx_.get(), SSL_VERIFY_PEER,
                     EvConnection::SSLVerifyCallback);
}


namespace {


string ErrorFlagDescription(const evhtp_error_flags flags) {
  return string(flags & BEV_EVENT_READING ? " READING" : "") +
         (flags & BEV_EVENT_WRITING ? " WRITING" : "") +
         (flags & BEV_EVENT_EOF ? " EOF" : "") +
         (flags & BEV_EVENT_ERROR ? " ERROR" : "") +
         (flags & BEV_EVENT_TIMEOUT ? " TIMEOUT" : "") +
         (flags & BEV_EVENT_CONNECTED ? " CONNECTED" : "");
}


}  // namespace


// static
evhtp_res EvConnection::ConnectionErrorHook(evhtp_connection_t* conn,
                                            evhtp_error_flags flags,
                                            void* arg) {
  CHECK_NOTNULL(conn);
  CHECK_NOTNULL(arg);
  CHECK(libevent::Base::OnEventThread());
  EvConnection* const c(static_cast<EvConnection*>(arg));
  LOG(WARNING) << "Releasing errored connection to " << c->other_end_.first
               << ":" << c->other_end_.second;

  CHECK_EQ(conn, c->ev_conn_);
  c->SetErrored();

  // Need to let the client know their request has failed, seems evhtp doesn't
  // do that by default so we'll call the request done callback here.
  if (conn->request) {
    // If someone hasn't already modified the default status, set it to a
    // generic "something went wrong" value here:
    if (conn->request->status == 200) {
      bool log_error(true);
      if (flags == BEV_EVENT_EOF) {
        // not actually an error, we just read up to the EOF, probably the
        // server didn't send a Content-Length header.
        log_error = false;
      } else if (flags & BEV_EVENT_TIMEOUT) {
        conn->request->status = kTimeout;
      } else {
        conn->request->status = kUnknownErrorStatus;
      }
      if (log_error) {
        const string error_str(
            evutil_socket_error_to_string(evutil_socket_geterror(conn->sock)));
        LOG(WARNING) << "error flag (0x" << std::hex << static_cast<int>(flags)
                     << "):" << ErrorFlagDescription(flags) << " : "
                     << error_str;
      }
    } else {
      LOG(WARNING) << "status already set to " << conn->request->status;
    }
    // The callback is going to Put() the Connection back, which will release
    // the underlying connection, and then delete the Connection wrapper when
    // it's determined that it's bad.
    conn->request->cb(conn->request, conn->request->cbarg);
    conn->request = nullptr;
  }
  return EVHTP_RES_OK;
}


// static
evhtp_res EvConnection::ConnectionFinishedHook(evhtp_connection_t* conn,
                                               void* arg) {
  CHECK_NOTNULL(conn);
  CHECK_NOTNULL(arg);
  CHECK(libevent::Base::OnEventThread());
  // libevhtp has finished with this connection so we can release the
  // shared_ptr we had to keep it around until now.
  unique_ptr<shared_ptr<EvConnection>> const c(
      static_cast<shared_ptr<EvConnection>*>(arg));
  VLOG(1) << "Finished connection to " << (*c)->other_end_.first << ":"
          << (*c)->other_end_.second;
  // The underlying evhtp_connection_t is about to be freed, make sure nobody
  // can hurt themselves via a dangling pointer.
  (*c)->ev_conn_ = nullptr;
  return EVHTP_RES_OK;
}


// static
void ConnectionPool::RemoveDeadConnectionsFromDeque(
    const unique_lock<mutex>& lock,
    std::deque<ConnectionPool::TimestampedConnection>* deque) {
  CHECK(lock.owns_lock());
  CHECK(deque);

  // Do a sweep and remove any dead connections
  for (auto deque_it(deque->begin()); deque_it != deque->end();) {
    CHECK(deque_it->second);
    if (!deque_it->second->connection()) {
      VLOG(1) << "Removing dead connection to "
              << deque_it->second->other_end().first << ":"
              << deque_it->second->other_end().second;
      deque_it = deque->erase(deque_it);
      continue;
    }
    ++deque_it;
  }
}


unique_ptr<ConnectionPool::Connection> ConnectionPool::Get(const URL& url) {
  CHECK(url.Protocol() == "http" || url.Protocol() == "https");
  const uint16_t default_port(url.Protocol() == "https" ? 443 : 80);
  HostPortPair key(url.Host(), url.Port() != 0 ? url.Port() : default_port);
  unique_lock<mutex> lock(lock_);

  auto it(conns_.find(key));

  if (it != conns_.end() && !it->second.empty()) {
    RemoveDeadConnectionsFromDeque(lock, &it->second);
  }

  if (it == conns_.end() || it->second.empty()) {
    VLOG(1) << "new evhtp_connection for " << key.first << ":" << key.second;
    // This EvConnection has a slightly complicated lifetime; it needs to hang
    // around until libevhtp/libevent have entirely finished with the
    // evhtp_connection_t it references, and for at least as long as the life
    // of the Connection we return from this method.
    //
    // This is accomplished through the use of a couple of shared_ptrs;
    // this one, which goes inside the returned Connection object, and another
    // created further below which gets passed in to the
    // ConnectionFinishedHook.
    auto conn(std::make_shared<EvConnection>(
        url.Protocol() == "https"
            ? base_->HttpsConnectionNew(key.first, key.second, ssl_ctx_.get())
            : base_->HttpConnectionNew(key.first, key.second),
        move(key)));
    unique_ptr<ConnectionPool::Connection> handle(new Connection(conn));
    struct timeval read_timeout = {FLAGS_connection_read_timeout_seconds,
                                   kZeroMillis};
    struct timeval write_timeout = {FLAGS_connection_write_timeout_seconds,
                                    kZeroMillis};
    evhtp_connection_set_timeouts(handle->connection(), &read_timeout,
                                  &write_timeout);
    evhtp_set_hook(&handle->connection()->hooks, evhtp_hook_on_conn_error,
                   reinterpret_cast<evhtp_hook>(
                       EvConnection::ConnectionErrorHook),
                   reinterpret_cast<void*>(conn.get()));
    evhtp_set_hook(
        &handle->connection()->hooks, evhtp_hook_on_connection_fini,
        reinterpret_cast<evhtp_hook>(EvConnection::ConnectionFinishedHook),
        // We'll hold on to another shared_ptr to the Connection
        // until evhtp tells us that it's finished with the cnxn.
        reinterpret_cast<void*>(new shared_ptr<EvConnection>(conn)));
    return handle;
  }

  VLOG(1) << "cached evhtp_connection for " << key.first << ":" << key.second;
  unique_ptr<ConnectionPool::Connection> retval(
      move(it->second.back().second));
  it->second.pop_back();
  CHECK_NOTNULL(retval->connection());

  return retval;
}


void ConnectionPool::Put(unique_ptr<ConnectionPool::Connection> handle) {
  if (!handle) {
    VLOG(1) << "returned null Connection";
    return;
  }

  if (!handle->connection()) {
    VLOG(1) << "returned dead Connection";
    handle.reset();
    return;
  }

  if (handle->GetErrored()) {
    VLOG(1) << "returned errored Connection";
    handle.reset();
    return;
  }

  const HostPortPair& key(handle->other_end());
  VLOG(1) << "returned Connection for " << key.first << ":" << key.second;
  lock_guard<mutex> lock(lock_);
  auto& entry(conns_[key]);

  CHECK_GE(FLAGS_url_fetcher_max_conn_per_host_port, 0);
  entry.emplace_back(make_pair(system_clock::now(), move(handle)));
  const string hostport(HostPortString(key));
  VLOG(1) << "ConnectionPool for " << hostport << " size : " << entry.size();
  connections_per_host_port->Set(hostport, entry.size());
  if (!cleanup_scheduled_ &&
      entry.size() >
          static_cast<uint>(FLAGS_url_fetcher_max_conn_per_host_port)) {
    cleanup_scheduled_ = true;
    base_->Add(bind(&ConnectionPool::Cleanup, this));
  }
}


void ConnectionPool::Cleanup() {
  unique_lock<mutex> lock(lock_);
  cleanup_scheduled_ = false;
  const system_clock::time_point cutoff(
      system_clock::now() -
      seconds(FLAGS_connection_pool_max_unused_age_seconds));

  // conns_ is a std::map<HostPortPair, std::deque<TimestampedConnection>>
  for (auto& entry : conns_) {
    RemoveDeadConnectionsFromDeque(lock, &entry.second);
    while (entry.second.front().first < cutoff &&
           entry.second.size() >
               static_cast<uint>(FLAGS_url_fetcher_max_conn_per_host_port)) {
      entry.second.pop_front();
    }
    const string hostport(HostPortString(entry.first));
    VLOG(1) << "ConnectionPool for " << hostport
            << " size : " << entry.second.size();
    connections_per_host_port->Set(hostport, entry.second.size());
  }
}


}  // namespace internal
}  // namespace cert_trans
