#ifndef CERT_TRANS_NET_URL_H_
#define CERT_TRANS_NET_URL_H_

#include <stdint.h>
#include <ostream>
#include <string>

namespace cert_trans {


class URL {
 public:
  URL() : port_(0) {
  }

  explicit URL(const std::string& url);

  const std::string& Protocol() const {
    return protocol_;
  }

  const std::string& Host() const {
    return host_;
  }

  uint16_t Port() const {
    return port_;
  }

  const std::string& Path() const {
    return path_;
  }

  const std::string& Query() const {
    return query_;
  }

  std::string PathQuery() const;

  void SetProtocol(const std::string& protocol) {
    protocol_ = protocol;
  }

  void SetHost(const std::string& host) {
    host_ = host;
  }

  void SetPort(uint16_t port) {
    port_ = port;
  }

  void SetPath(const std::string& path) {
    path_ = path;
  }

  void SetQuery(const std::string& query) {
    query_ = query;
  }

  bool operator<(const URL& rhs) const {
    return protocol_ < rhs.protocol_ && host_ < rhs.host_ &&
           port_ < rhs.port_ && path_ < rhs.path_ && query_ < rhs.query_;
  }

  bool operator==(const URL& rhs) const {
    return protocol_ == rhs.protocol_ && host_ == rhs.host_ &&
           port_ == rhs.port_ && path_ == rhs.path_ && query_ == rhs.query_;
  }

 private:
  std::string protocol_;
  std::string host_;
  uint16_t port_;
  std::string path_;
  std::string query_;
};


// For testing and debugging.
std::ostream& operator<<(std::ostream& out, const URL& url);


}  // namespace cert_trans

#endif  // CERT_TRANS_NET_URL_H_
