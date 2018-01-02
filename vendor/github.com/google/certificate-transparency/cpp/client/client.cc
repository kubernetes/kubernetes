#include "client/client.h"

#include <arpa/inet.h>
#include <glog/logging.h>
#include <netinet/in.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/socket.h>
#include <unistd.h>

#ifdef __MACH__
// does not exist on MacOS
#define MSG_NOSIGNAL 0
#endif

using std::string;

Client::Client(const string& server, uint16_t port)
    : server_(server), port_(port), fd_(-1) {
}

Client::~Client() {
  Disconnect();
}

bool Client::Connect() {
  CHECK(!Connected());

  fd_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  PCHECK(fd_ >= 0) << "Socket creation failed";

  static struct sockaddr_in server_socket;
  memset(&server_socket, 0, sizeof server_socket);
  server_socket.sin_family = AF_INET;
  server_socket.sin_port = htons(port_);
  CHECK_EQ(1, inet_aton(server_.c_str(), &server_socket.sin_addr))
      << "Can't parse server address: " << server_;

  int ret =
      connect(fd_, (struct sockaddr*)&server_socket, sizeof server_socket);
  if (ret < 0) {
    Disconnect();
    PLOG(ERROR) << "Connection to " << server_ << ":" << port_ << " failed";
    return false;
  }
  LOG(INFO) << "Connected to " << server_ << ":" << port_;
  return true;
}

bool Client::Connected() const {
  return fd_ > 0;
}

void Client::Disconnect() {
  if (fd_ > 0) {
    close(fd_);
    LOG(INFO) << "Disconnected from " << server_ << ":" << port_;
    fd_ = -1;
  }
}

bool Client::Write(const string& data) {
  CHECK(Connected());
  int n = send(fd_, data.data(), data.length(), MSG_NOSIGNAL);
  if (n <= 0) {
    PCHECK(errno == EPIPE) << "Send failed";
    LOG(ERROR) << "Remote server closed the connection.";
    Disconnect();
    return false;
  }

  CHECK_EQ(data.length(), (unsigned)n);

  VLOG(1) << "wrote " << data.length() << " bytes";
  return true;
}

bool Client::Read(size_t length, string* result) {
  CHECK(Connected());
  char* buf = new char[length];
  for (size_t offset = 0; offset < length;) {
    int n = recv(fd_, buf + offset, length - offset, MSG_NOSIGNAL);
    if (n <= 0) {
      PCHECK(errno == EPIPE) << "Read failed";
      LOG(ERROR) << "Remote server closed the connection.";
      Disconnect();
      delete[] buf;
      return false;
    }

    offset += n;
  }
  result->assign(string(buf, length));
  delete[] buf;
  VLOG(1) << "read " << length << " bytes";
  return true;
}
