#ifndef CLIENT_H
#define CLIENT_H

#include <stdint.h>
#include <string>

#include "base/macros.h"

// Socket creation for client connections.
class Client {
 public:
  Client(const std::string& server, uint16_t port);

  ~Client();

  // Create a TCP socket and attempt to connect to server:port.
  // The Connect()-Disconnect() sequence can be called repeatedly.
  bool Connect();

  int fd() const {
    return fd_;
  }

  // The remote end may have closed the socket, in which
  // case this will still return true, but the next read/write
  // will fail and disconnect.
  bool Connected() const;

  void Disconnect();

  bool Write(const std::string& data);

  bool Read(size_t length, std::string* result);

 private:
  const std::string server_;
  const uint16_t port_;
  int fd_;

  DISALLOW_COPY_AND_ASSIGN(Client);
};
#endif
