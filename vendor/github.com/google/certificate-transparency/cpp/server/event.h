/* -*- mode: c++; indent-tabs-mode: nil -*- */
#ifndef CERT_TRANS_SERVER_EVENT_H_
#define CERT_TRANS_SERVER_EVENT_H_

#include <glog/logging.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <deque>
#include <string>

#include "base/macros.h"

class Services {
 public:
  // because time is expensive, for most tasks we can just use some
  // time sampled within this event handling loop. So, the main loop
  // needs to call SetRoughTime() appropriately.
  static time_t RoughTime() {
    if (rough_time_ == 0)
      rough_time_ = time(NULL);
    return rough_time_;
  }

  static void SetRoughTime() {
    rough_time_ = 0;
  }

  static bool InitServer(int* sock, int port, const char* ip, int type);

 private:
  // This class is only used as a namespace, it should never be
  // instantiated.
  // TODO(pphaneuf): Make this into normal functions in a namespace.
  Services();

  static time_t rough_time_;
};

class EventLoop;

class FD {
 public:
  enum CanDelete { DELETE, NO_DELETE };

  FD(EventLoop* loop, int fd, CanDelete deletable = DELETE);

  virtual ~FD() = default;

  virtual bool WantsWrite() const = 0;

  virtual void WriteIsAllowed() = 0;

  virtual bool WantsRead() const = 0;

  virtual void ReadIsAllowed() = 0;

  bool WantsErase() const {
    return wants_erase_;
  }

  void Close();

  int fd() const {
    return fd_;
  }

  bool CanDrop() const {
    return deletable_ == DELETE;
  }

  // Don't forget to call me if anything happens!
  void Activity() {
    last_activity_ = Services::RoughTime();
  }

  time_t LastActivity() const {
    return last_activity_;
  }

 protected:
  EventLoop* loop() const {
    return loop_;
  }

  bool WillAccept(int fd);

 private:
  int fd_;
  EventLoop* loop_;
  bool wants_erase_;
  CanDelete deletable_;
  time_t last_activity_;

  // Note that while you can set these low for test, they behave a
  // bit strangely when set low - for example, it is quite easy to
  // hit the limit even if the window is not 0. I'm guessing 1000
  // and 100 would be good numbers. Note EventLoop::kIdleTime below,
  // also.
  static const int kFDLimit = 1000;
  static const int kFDLimitWindow = 1;

  DISALLOW_COPY_AND_ASSIGN(FD);
};

class Listener : public FD {
 public:
  Listener(EventLoop* loop, int fd) : FD(loop, fd, NO_DELETE) {
  }

  bool WantsRead() const {
    return true;
  }

  void ReadIsAllowed();

  bool WantsWrite() const {
    return false;
  }

  void WriteIsAllowed();

  virtual void Accepted(int fd) = 0;
};

class RepeatedEvent {
 public:
  RepeatedEvent(time_t repeat_frequency_seconds)
      : frequency_(repeat_frequency_seconds),
        last_activity_(Services::RoughTime()) {
  }

  // The time when we should execute next.
  time_t Trigger() {
    return last_activity_ + frequency_;
  }

  virtual std::string Description() = 0;

  virtual void Execute() = 0;

  void Activity() {
    last_activity_ = Services::RoughTime();
  }

 private:
  time_t frequency_;
  time_t last_activity_;
};

class EventLoop {
 public:
  EventLoop() : go_(true) {
  }

  void Add(FD* fd) {
    fds_.push_back(fd);
  }

  void Add(RepeatedEvent* event) {
    events_.push_back(event);
  }

  // Returns remaining time until the next alarm.
  time_t ProcessRepeatedEvents();

  void OneLoop();

  void Forever();

  void MaybeDropOne();

  void Stop();

 private:
  bool EraseCheck(std::deque<FD*>::iterator* pfd);

  static void Set(int fd, fd_set* fdset, int* max);

  std::deque<FD*> fds_;
  std::vector<RepeatedEvent*> events_;
  // This should probably be set to 2 for anything but test (or 1 or 0).
  // 2: everything gets a chance to speak.
  // 1: sometimes the clock will tick before some get a chance to speak.
  // 0: maybe no-one ever gets a chance to speak.
  static const time_t kIdleTime = 20;

  bool go_;

  DISALLOW_COPY_AND_ASSIGN(EventLoop);
};

class Server : public FD {
 public:
  Server(EventLoop* loop, int fd) : FD(loop, fd) {
  }

  bool WantsRead() const {
    return true;
  }

  void ReadIsAllowed();

  // There are fresh bytes available in rbuffer.  It is the callee's
  // responsibility to remove consumed bytes from rbuffer. This will
  // NOT be called again until more data arrives from the network,
  // even if there are unconsumed bytes in rbuffer.
  virtual void BytesRead(std::string* rbuffer) = 0;

  bool WantsWrite() const {
    return !wbuffer_.empty();
  }

  void WriteIsAllowed();

  void Write(std::string str) {
    wbuffer_.append(str);
  }

 private:
  std::string rbuffer_;
  std::string wbuffer_;
};

class UDPServer : public FD {
 public:
  UDPServer(EventLoop* loop, int fd) : FD(loop, fd, NO_DELETE) {
  }

  bool WantsRead() const {
    return true;
  }

  void ReadIsAllowed();

  bool WantsWrite() const {
    return !write_queue_.empty();
  }

  void WriteIsAllowed();

  // A packet has been read. It will not be re-presented if you do not
  // process it now.
  virtual void PacketRead(const sockaddr_in& from, const char* buf,
                          size_t len) = 0;

  // Queue a packet for sending
  void QueuePacket(const sockaddr_in& to, const char* buf, size_t len);
  void QueuePacket(const sockaddr_in& to, const unsigned char* buf,
                   size_t len) {
    QueuePacket(to, reinterpret_cast<const char*>(buf), len);
  }

 private:
  struct WBuffer {
    sockaddr_in sa;
    std::string packet;
  };
  std::deque<WBuffer> write_queue_;
};

class UDPEchoServer : public UDPServer {
 public:
  UDPEchoServer(EventLoop* loop, int fd) : UDPServer(loop, fd) {
  }

  virtual void PacketRead(const sockaddr_in& from, const char* buf,
                          size_t len) {
    QueuePacket(from, buf, len);
  }
};

#endif  // CERT_TRANS_SERVER_EVENT_H_
