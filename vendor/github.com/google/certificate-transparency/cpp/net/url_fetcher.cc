#include "net/url_fetcher.h"

#include <event2/buffer.h>
#include <event2/keyvalq_struct.h>
#include <evhtp.h>
#include <glog/logging.h>
#include <htparse.h>

#include "net/connection_pool.h"
#include "util/thread_pool.h"

using cert_trans::internal::ConnectionPool;
using std::bind;
using std::endl;
using std::make_pair;
using std::move;
using std::ostream;
using std::string;
using std::to_string;
using std::unique_ptr;
using util::Status;
using util::Task;
using util::TaskHold;

namespace cert_trans {


struct UrlFetcher::Impl {
  Impl(libevent::Base* base, ThreadPool* thread_pool)
      : base_(CHECK_NOTNULL(base)),
        thread_pool_(CHECK_NOTNULL(thread_pool)),
        pool_(base_) {
  }

  libevent::Base* const base_;
  ThreadPool* const thread_pool_;
  internal::ConnectionPool pool_;
};


namespace {


htp_method VerbToCmdType(UrlFetcher::Verb verb) {
  switch (verb) {
    case UrlFetcher::Verb::GET:
      return htp_method_GET;

    case UrlFetcher::Verb::POST:
      return htp_method_POST;

    case UrlFetcher::Verb::PUT:
      return htp_method_PUT;

    case UrlFetcher::Verb::DELETE:
      return htp_method_DELETE;
  }

  LOG(FATAL) << "unknown UrlFetcher::Verb: " << static_cast<int>(verb);
}


struct State {
  State(libevent::Base* base, ConnectionPool* pool,
        const UrlFetcher::Request& request, UrlFetcher::Response* response,
        Task* task);

  ~State() {
    CHECK(!conn_) << "request state object still had a connection at cleanup?";
  }

  void MakeRequest();

  // The following methods must only be called on the libevent
  // dispatch thread.
  void RunRequest();
  void RequestDone(evhtp_request_t* req);

  libevent::Base* const base_;
  ConnectionPool* const pool_;
  const UrlFetcher::Request request_;
  UrlFetcher::Response* const response_;
  Task* const task_;

  unique_ptr<ConnectionPool::Connection> conn_;
};


void RequestCallback(evhtp_request_t* req, void* userdata) {
  static_cast<State*>(CHECK_NOTNULL(userdata))->RequestDone(req);
}


UrlFetcher::Request NormaliseRequest(UrlFetcher::Request req) {
  if (req.url.Path().empty()) {
    req.url.SetPath("/");
  }

  if (req.headers.find("Host") == req.headers.end()) {
    req.headers.insert(make_pair("Host", req.url.Host()));
  }

  return req;
}


State::State(libevent::Base* base, ConnectionPool* pool,
             const UrlFetcher::Request& request,
             UrlFetcher::Response* response, Task* task)
    : base_(CHECK_NOTNULL(base)),
      pool_(CHECK_NOTNULL(pool)),
      request_(NormaliseRequest(request)),
      response_(CHECK_NOTNULL(response)),
      task_(CHECK_NOTNULL(task)) {
  if (request_.url.Protocol() != "http" &&
      request_.url.Protocol() != "https") {
    VLOG(1) << "unsupported protocol: " << request_.url.Protocol();
    task_->Return(Status(util::error::INVALID_ARGUMENT,
                         "UrlFetcher: unsupported protocol: " +
                             request_.url.Protocol()));
    return;
  }
}


void State::MakeRequest() {
  CHECK(!libevent::Base::OnEventThread());
  conn_ = pool_->Get(request_.url);
  base_->Add(bind(&State::RunRequest, this));
}


void State::RunRequest() {
  CHECK(libevent::Base::OnEventThread());
  evhtp_request_t* const http_req(
      CHECK_NOTNULL(evhtp_request_new(&RequestCallback, this)));
  if (!request_.body.empty() &&
      request_.headers.find("Content-Length") == request_.headers.end()) {
    evhtp_headers_add_header(
        http_req->headers_out,
        evhtp_header_new("Content-Length",
                         to_string(request_.body.size()).c_str(), 1, 1));
  }
  for (const auto& header : request_.headers) {
    evhtp_headers_add_header(http_req->headers_out,
                             evhtp_header_new(header.first.c_str(),
                                              header.second.c_str(), 1, 1));
  }

  if (!conn_->connection() || conn_->GetErrored()) {
    conn_.reset();
    task_->Return(Status(util::error::UNAVAILABLE, "connection failed."));
    return;
  }

  const htp_method verb(VerbToCmdType(request_.verb));
  VLOG(1) << "evhtp_make_request(" << conn_.get()->connection() << ", "
          << http_req << ", " << verb << ", \"" << request_.url.PathQuery()
          << "\")";
  VLOG(2) << request_;
  if (evhtp_make_request(conn_->connection(), http_req, verb,
                         request_.url.PathQuery().c_str()) != 0) {
    VLOG(1) << "evhtp_make_request error";
    // Put back the connection, RequestDone is not going to get
    // called.
    pool_->Put(move(conn_));
    task_->Return(Status(util::error::INTERNAL, "evhtp_make_request error"));
    return;
  }

  // evhtp_make_request doesn't know anything about the body, so we send it
  // outselves here:
  if (!request_.body.empty()) {
    if (evbuffer_add_reference(bufferevent_get_output(
                                   conn_->connection()->bev),
                               request_.body.data(), request_.body.size(),
                               nullptr, nullptr) != 0) {
      VLOG(1) << "error when adding the request body";
      task_->Return(
          Status(util::error::INTERNAL, "could not set the request body"));
      return;
    }
  }
}


struct evhtp_request_deleter {
  void operator()(evhtp_request_t* r) const {
    evhtp_request_free(r);
  }
};


void State::RequestDone(evhtp_request_t* req) {
  CHECK(libevent::Base::OnEventThread());
  CHECK(conn_);
  this->pool_->Put(move(conn_));
  unique_ptr<evhtp_request_t, evhtp_request_deleter> req_deleter(req);

  if (!req) {
    // TODO(pphaneuf): The dreaded null request... These are fairly
    // fatal things, like protocol parse errors, but could also be a
    // connection timeout. I think we should do retries in this case,
    // with a deadline of our own? At least, then, it would be easier
    // to distinguish between an obscure error, or a more common
    // timeout.
    VLOG(1) << "RequestCallback received a null request";
    task_->Return(Status::UNKNOWN);
    return;
  }

  response_->status_code = req->status;
  if (response_->status_code < 100) {
    util::Status status;
    switch (response_->status_code) {
      case kTimeout:
        status =
            Status(util::error::DEADLINE_EXCEEDED, "connection timed out");
        break;
      case kSSLErrorStatus:
        // There was a problem communicating with the remote host.
        status = Status(util::error::UNAVAILABLE, "SSL connection failed");
        break;
      case kUnknownErrorStatus:
        status = Status(util::error::UNAVAILABLE, "connection failed");
        break;
      default:
        LOG(WARNING) << "Unknown status code encountered: "
                     << response_->status_code;
        status =
            Status(util::error::UNKNOWN, "unknown status code encountered");
    }
    task_->Return(status);
    return;
  }

  response_->headers.clear();
  for (evhtp_kv_s* ptr = req->headers_in->tqh_first; ptr;
       ptr = ptr->next.tqe_next) {
    response_->headers.insert(make_pair(ptr->key, ptr->val));
  }

  const size_t body_length(evbuffer_get_length(req->buffer_in));
  string body(reinterpret_cast<const char*>(
                  evbuffer_pullup(req->buffer_in, body_length)),
              body_length);
  response_->body.swap(body);

  VLOG(2) << *response_;

  task_->Return();
}


}  // namespace


// Needs to be defined where Impl is also defined.
UrlFetcher::UrlFetcher() {
}


UrlFetcher::UrlFetcher(libevent::Base* base, ThreadPool* thread_pool)
    : impl_(new Impl(CHECK_NOTNULL(base), CHECK_NOTNULL(thread_pool))) {
}


// Needs to be defined where Impl is also defined.
UrlFetcher::~UrlFetcher() {
}


void UrlFetcher::Fetch(const Request& req, Response* resp, Task* task) {
  TaskHold hold(task);

  State* const state(new State(impl_->base_, &impl_->pool_, req, resp, task));
  task->DeleteWhenDone(state);

  // Run State::MakeRequest() on the task's executor because it may
  // block doing DNS resolution etc.
  // TODO(alcutter): this can go back to being put straight on the event Base
  // once evhtp supports creating SSL connections to a DNS name.
  impl_->thread_pool_->Add(bind(&State::MakeRequest, state));
}


ostream& operator<<(ostream& output, const UrlFetcher::Response& resp) {
  output << "status_code: " << resp.status_code << endl << "headers {" << endl;
  for (const auto& header : resp.headers) {
    output << "  " << header.first << ": " << header.second << endl;
  }
  output << "}" << endl << "body: <<EOF" << endl << resp.body << "EOF" << endl;

  return output;
}


ostream& operator<<(ostream& output, const UrlFetcher::Request& req) {
  output << "verb: " << req.verb << endl
         << "url: " << req.url << endl
         << "headers: " << req.headers << endl
         << "body: <<EOF" << endl
         << req.body << "EOF" << endl;
  return output;
}


ostream& operator<<(ostream& os, const UrlFetcher::Verb& verb) {
  switch (verb) {
    case UrlFetcher::Verb::GET:
      os << "GET";
      break;
    case UrlFetcher::Verb::POST:
      os << "POST";
      break;
    case UrlFetcher::Verb::PUT:
      os << "PUT";
      break;
    case UrlFetcher::Verb::DELETE:
      os << "DELETE";
      break;
  }
  return os;
}


ostream& operator<<(ostream& os, const UrlFetcher::Headers& headers) {
  os << "{" << endl;
  for (const auto& h : headers) {
    os << "  " << h.first << ": " << h.second << endl;
  }
  return os << "}";
}


}  // namespace cert_trans
