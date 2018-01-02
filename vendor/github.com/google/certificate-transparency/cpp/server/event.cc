/* -*- indent-tabs-mode: nil -*- */
#include "server/event.h"

#include <limits.h>
#include <openssl/evp.h>
#include <openssl/pem.h>

time_t Services::rough_time_;

FD::FD(EventLoop* loop, int fd, CanDelete deletable)
    : fd_(fd), loop_(loop), wants_erase_(false), deletable_(deletable) {
  DCHECK_GE(fd, 0);
  CHECK_LT((unsigned)fd, (unsigned)FD_SETSIZE);
  loop->Add(this);
  Activity();
}

void FD::Close() {
  DCHECK_EQ(deletable_, DELETE) << "Can't call Close() on a non-deletable FD";
  if (wants_erase_) {
    LOG(INFO) << "Attempting to close an already closed fd " << fd();
    return;
  }
  LOG(INFO) << "Closing fd " << fd() << std::endl;
  wants_erase_ = true;
  shutdown(fd(), SHUT_RDWR);
  close(fd());
}

bool FD::WillAccept(int fd) {
  if (fd >= kFDLimit - kFDLimitWindow)
    loop()->MaybeDropOne();
  return fd < kFDLimit;
}

void Listener::ReadIsAllowed() {
  int in = accept(fd(), NULL, NULL);
  CHECK_GE(in, 0);
  if (!WillAccept(in)) {
    static char sorry[] = "No free connections.\n";

    // we have to consume the result.
    ssize_t s = write(in, sorry, sizeof sorry);
    if (s != sizeof sorry)
      LOG(WARNING) << "Failed to write sorry correctly.";
    shutdown(in, SHUT_RDWR);
    close(in);
    return;
  }
  Accepted(in);
}

void Listener::WriteIsAllowed() {
  DLOG(FATAL) << "WriteIsAllowed() called on a read-only Listener.";
}

time_t EventLoop::ProcessRepeatedEvents() {
  if (events_.empty())
    return INT_MAX;
  Services::SetRoughTime();
  time_t now = Services::RoughTime();
  time_t earliest = INT_MAX;
  for (std::vector<RepeatedEvent*>::iterator it = events_.begin();
       it != events_.end(); ++it) {
    RepeatedEvent* event = *it;
    time_t trigger = event->Trigger();
    if (trigger <= now) {
      event->Execute();
      LOG(INFO) << "Executed " << event->Description() << " with a delay of "
                << difftime(now, trigger) << " seconds";
      event->Activity();
      trigger = event->Trigger();
      CHECK_GT(trigger, now);
    }
    earliest = std::min(earliest, trigger);
  }
  CHECK_GT(earliest, 0);
  return earliest - now;
}

void EventLoop::OneLoop() {
  time_t select_timeout = ProcessRepeatedEvents();
  // Do not schedule any repeated events between now and the next
  // select - they will get ignored until select returns.
  CHECK_GT(select_timeout, 0);

  fd_set readers, writers;
  int max = -1;

  memset(&readers, '\0', sizeof readers);
  memset(&writers, '\0', sizeof writers);
  for (std::deque<FD*>::const_iterator pfd = fds_.begin(); pfd != fds_.end();
       ++pfd) {
    FD* fd = *pfd;

    DCHECK(!fd->WantsErase());
    if (fd->WantsWrite())
      Set(fd->fd(), &writers, &max);
    if (fd->WantsRead())
      Set(fd->fd(), &readers, &max);
  }

  CHECK_GE(max, 0);

  struct timeval tv;
  tv.tv_sec = select_timeout;
  tv.tv_usec = 0;

  int r = select(max + 1, &readers, &writers, NULL, &tv);
  if (r == 0)
    return;

  CHECK_GT(r, 0);

  Services::SetRoughTime();
  int n = 0;
  for (std::deque<FD*>::iterator pfd = fds_.begin(); pfd != fds_.end();) {
    FD* fd = *pfd;

    if (EraseCheck(&pfd))
      continue;

    if (FD_ISSET(fd->fd(), &writers)) {
      DCHECK(fd->WantsWrite());
      fd->WriteIsAllowed();
      fd->Activity();
      ++n;
    }

    if (EraseCheck(&pfd))
      continue;

    if (FD_ISSET(fd->fd(), &readers)) {
      DCHECK(fd->WantsRead());
      fd->ReadIsAllowed();
      fd->Activity();
      ++n;
    }

    if (EraseCheck(&pfd))
      continue;

    ++pfd;
  }
  CHECK_LE(n, r);
}

void EventLoop::Stop() {
  go_ = false;
}

void EventLoop::Forever() {
  for (; go_;)
    OneLoop();
}

void EventLoop::MaybeDropOne() {
  std::deque<FD*>::iterator drop = fds_.end();
  time_t oldest = Services::RoughTime() - kIdleTime;

  for (std::deque<FD*>::iterator pfd = fds_.begin(); pfd != fds_.end();
       ++pfd) {
    FD* fd = *pfd;

    if (fd->CanDrop() && fd->LastActivity() < oldest) {
      oldest = fd->LastActivity();
      drop = pfd;
    }
  }
  if (drop != fds_.end())
    (*drop)->Close();
}

bool EventLoop::EraseCheck(std::deque<FD*>::iterator* pfd) {
  if ((**pfd)->WantsErase()) {
    delete **pfd;
    *pfd = fds_.erase(*pfd);
    return true;
  }
  return false;
}

// static
void EventLoop::Set(int fd, fd_set* fdset, int* max) {
  DCHECK_GE(fd, 0);
  CHECK_LT((unsigned)fd, (unsigned)FD_SETSIZE);
  FD_SET(fd, fdset);
  if (fd > *max)
    *max = fd;
}

void Server::ReadIsAllowed() {
  char buf[1024];

  ssize_t n = read(fd(), buf, sizeof buf);
  VLOG(1) << "read " << n << " bytes from " << fd();
  if (n <= 0) {
    Close();
    return;
  }
  rbuffer_.append(buf, (size_t)n);
  BytesRead(&rbuffer_);
}

void Server::WriteIsAllowed() {
  ssize_t n = write(fd(), wbuffer_.data(), wbuffer_.length());
  VLOG(1) << "wrote " << n << " bytes to " << fd();
  if (n <= 0) {
    Close();
    return;
  }
  wbuffer_.erase(0, n);
}

void UDPServer::ReadIsAllowed() {
  char buf[2048];
  struct sockaddr_in sa;
  socklen_t sa_len = sizeof sa;

  ssize_t in = recvfrom(fd(), buf, sizeof buf, 0, (sockaddr*)&sa, &sa_len);
  CHECK_GE(in, 1);
  CHECK_EQ(sa_len, sizeof sa);
  // LOG(INFO) << "UDP packet " << util::HexString(std::string(buf, in));
  PacketRead(sa, buf, in);
}

void UDPServer::WriteIsAllowed() {
  CHECK(!write_queue_.empty());
  WBuffer wbuf = write_queue_.front();
  write_queue_.pop_front();
  ssize_t out = sendto(fd(), wbuf.packet.data(), wbuf.packet.length(), 0,
                       (const sockaddr*)&wbuf.sa, sizeof wbuf.sa);
  CHECK_NE(out, -1);
  CHECK_EQ((size_t)out, wbuf.packet.length());
}

void UDPServer::QueuePacket(const sockaddr_in& to, const char* buf,
                            size_t len) {
  WBuffer wbuf;
  wbuf.sa = to;
  wbuf.packet = std::string(buf, len);
  write_queue_.push_back(wbuf);
}

bool Services::InitServer(int* sock, int port, const char* ip, int type) {
  bool ret = false;
  struct sockaddr_in server;
  int s = -1;

  memset(&server, 0, sizeof(server));
  server.sin_family = AF_INET;
  server.sin_port = htons((unsigned short)port);
  if (ip == NULL)
    server.sin_addr.s_addr = htonl(INADDR_ANY);
  else
    memcpy(&server.sin_addr.s_addr, ip, 4);

  if (type == SOCK_STREAM)
    s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  else /* type == SOCK_DGRAM */
    s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (s == -1)
    goto err;

  {
    int j = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &j, sizeof j);
  }

  if (bind(s, (struct sockaddr*)&server, sizeof(server)) == -1) {
    perror("bind");
    goto err;
  }
  /* Make it 128 for linux */
  if (type == SOCK_STREAM && listen(s, 128) == -1)
    goto err;
  *sock = s;
  ret = true;
err:
  if (!ret && s != -1) {
    shutdown(s, SHUT_RDWR);
    close(s);
  }
  return ret;
}
