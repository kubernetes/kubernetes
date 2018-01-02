#ifndef CERT_TRANS_NET_URL_FETCHER_H_
#define CERT_TRANS_NET_URL_FETCHER_H_

#include <chrono>
#include <map>
#include <memory>
#include <ostream>
#include <string>

#include "base/macros.h"
#include "net/url.h"
#include "util/compare.h"
#include "util/task.h"

namespace cert_trans {

namespace libevent {
class Base;
}

class ThreadPool;


class UrlFetcher {
 public:
  typedef std::multimap<std::string, std::string, ci_less<std::string>>
      Headers;

  enum class Verb {
    GET,
    POST,
    PUT,
    DELETE,
  };

  struct Request {
    Request() : verb(Verb::GET) {
    }
    Request(const URL& input_url) : verb(Verb::GET), url(input_url) {
    }

    Verb verb;
    URL url;
    Headers headers;
    std::string body;
  };

  struct Response {
    Response() : status_code(0) {
    }

    int status_code;
    Headers headers;
    std::string body;
  };

  UrlFetcher(libevent::Base* base, ThreadPool* thread_pool);
  virtual ~UrlFetcher();

  // If the status on the task is not OK, the response will be in an
  // undefined state. If it is OK, it only means that the transaction
  // with the remote server went correctly, you should still check
  // Response::status_code.
  virtual void Fetch(const Request& req, Response* resp, util::Task* task);

 protected:
  UrlFetcher();

 private:
  struct Impl;
  const std::unique_ptr<Impl> impl_;

  DISALLOW_COPY_AND_ASSIGN(UrlFetcher);
};


std::ostream& operator<<(std::ostream& output, const UrlFetcher::Request& req);
std::ostream& operator<<(std::ostream& output,
                         const UrlFetcher::Response& resp);
::std::ostream& operator<<(::std::ostream& os,
                           const UrlFetcher::Headers& headers);
::std::ostream& operator<<(::std::ostream& os, const UrlFetcher::Verb& verb);


}  // namespace cert_trans

#endif  // CERT_TRANS_NET_URL_FETCHER_H_
