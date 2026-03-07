# CHANGELOG

## v0.5.0

**This is the first release of package socket that only supports Go 1.21+.
Users on older versions of Go must use v0.4.1.**

- [Improvement]: drop support for older versions of Go.
- [New API]: add `socket.Conn` wrappers for various `Getsockopt` and
  `Setsockopt` system calls.

## v0.4.1

- [Bug Fix] [commit](https://github.com/mdlayher/socket/commit/2a14ceef4da279de1f957c5761fffcc6c87bbd3b):
  ensure `socket.Conn` can be used with non-socket file descriptors by handling
  `ENOTSOCK` in the constructor.

## v0.4.0

**This is the first release of package socket that only supports Go 1.18+.
Users on older versions of Go must use v0.3.0.**

- [Improvement]: drop support for older versions of Go so we can begin using
  modern versions of `x/sys` and other dependencies.

## v0.3.0

**This is the last release of package socket that supports Go 1.17 and below.**

- [New API/API change] [PR](https://github.com/mdlayher/socket/pull/8):
  numerous `socket.Conn` methods now support context cancelation. Future
  releases will continue adding support as needed.
  - New `ReadContext` and `WriteContext` methods.
  - `Connect`, `Recvfrom`, `Recvmsg`, `Sendmsg`, and `Sendto` methods now accept
    a context.
  - `Sendto` parameter order was also fixed to match the underlying syscall.

## v0.2.3

- [New API] [commit](https://github.com/mdlayher/socket/commit/a425d96e0f772c053164f8ce4c9c825380a98086):
  `socket.Conn` has new `Pidfd*` methods for wrapping the `pidfd_*(2)` family of
  system calls.

## v0.2.2

- [New API] [commit](https://github.com/mdlayher/socket/commit/a2429f1dfe8ec2586df5a09f50ead865276cd027):
  `socket.Conn` has new `IoctlKCM*` methods for wrapping `ioctl(2)` for `AF_KCM`
  operations.

## v0.2.1

- [New API] [commit](https://github.com/mdlayher/socket/commit/b18ddbe9caa0e34552b4409a3aa311cb460d2f99):
  `socket.Conn` has a new `SetsockoptPacketMreq` method for wrapping
  `setsockopt(2)` for `AF_PACKET` socket options.

## v0.2.0

- [New API] [commit](https://github.com/mdlayher/socket/commit/6e912a68523c45e5fd899239f4b46c402dd856da):
  `socket.FileConn` can be used to create a `socket.Conn` from an existing
  `os.File`, which may be provided by systemd socket activation or another
  external mechanism.
- [API change] [commit](https://github.com/mdlayher/socket/commit/66d61f565188c23fe02b24099ddc856d538bf1a7):
  `socket.Conn.Connect` now returns the `unix.Sockaddr` value provided by
  `getpeername(2)`, since we have to invoke that system call anyway to verify
  that a connection to a remote peer was successfully established.
- [Bug Fix] [commit](https://github.com/mdlayher/socket/commit/b60b2dbe0ac3caff2338446a150083bde8c5c19c):
  check the correct error from `unix.GetsockoptInt` in the `socket.Conn.Connect`
  method. Thanks @vcabbage!

## v0.1.2

- [Bug Fix]: `socket.Conn.Connect` now properly checks the `SO_ERROR` socket
  option value after calling `connect(2)` to verify whether or not a connection
  could successfully be established. This means that `Connect` should now report
  an error for an `AF_INET` TCP connection refused or `AF_VSOCK` connection
  reset by peer.
- [New API]: add `socket.Conn.Getpeername` for use in `Connect`, but also for
  use by external callers.

## v0.1.1

- [New API]: `socket.Conn` now has `CloseRead`, `CloseWrite`, and `Shutdown`
  methods.
- [Improvement]: internal rework to more robustly handle various errors.

## v0.1.0

- Initial unstable release. Most functionality has been developed and ported
from package [`netlink`](https://github.com/mdlayher/netlink).
