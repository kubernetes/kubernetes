package socket

import (
	"context"
	"errors"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

// Lock in an expected public interface for convenience.
var _ interface {
	io.ReadWriteCloser
	syscall.Conn
	SetDeadline(t time.Time) error
	SetReadDeadline(t time.Time) error
	SetWriteDeadline(t time.Time) error
} = &Conn{}

// A Conn is a low-level network connection which integrates with Go's runtime
// network poller to provide asynchronous I/O and deadline support.
//
// Many of a Conn's blocking methods support net.Conn deadlines as well as
// cancelation via context. Note that passing a context with a deadline set will
// override any of the previous deadlines set by calls to the SetDeadline family
// of methods.
type Conn struct {
	// Indicates whether or not Conn.Close has been called. Must be accessed
	// atomically. Atomics definitions must come first in the Conn struct.
	closed uint32

	// A unique name for the Conn which is also associated with derived file
	// descriptors such as those created by accept(2).
	name string

	// facts contains information we have determined about Conn to trigger
	// alternate behavior in certain functions.
	facts facts

	// Provides access to the underlying file registered with the runtime
	// network poller, and arbitrary raw I/O calls.
	fd *os.File
	rc syscall.RawConn
}

// facts contains facts about a Conn.
type facts struct {
	// isStream reports whether this is a streaming descriptor, as opposed to a
	// packet-based descriptor like a UDP socket.
	isStream bool

	// zeroReadIsEOF reports Whether a zero byte read indicates EOF. This is
	// false for a message based socket connection.
	zeroReadIsEOF bool
}

// A Config contains options for a Conn.
type Config struct {
	// NetNS specifies the Linux network namespace the Conn will operate in.
	// This option is unsupported on other operating systems.
	//
	// If set (non-zero), Conn will enter the specified network namespace and an
	// error will occur in Socket if the operation fails.
	//
	// If not set (zero), a best-effort attempt will be made to enter the
	// network namespace of the calling thread: this means that any changes made
	// to the calling thread's network namespace will also be reflected in Conn.
	// If this operation fails (due to lack of permissions or because network
	// namespaces are disabled by kernel configuration), Socket will not return
	// an error, and the Conn will operate in the default network namespace of
	// the process. This enables non-privileged use of Conn in applications
	// which do not require elevated privileges.
	//
	// Entering a network namespace is a privileged operation (root or
	// CAP_SYS_ADMIN are required), and most applications should leave this set
	// to 0.
	NetNS int
}

// High-level methods which provide convenience over raw system calls.

// Close closes the underlying file descriptor for the Conn, which also causes
// all in-flight I/O operations to immediately unblock and return errors. Any
// subsequent uses of Conn will result in EBADF.
func (c *Conn) Close() error {
	// The caller has expressed an intent to close the socket, so immediately
	// increment s.closed to force further calls to result in EBADF before also
	// closing the file descriptor to unblock any outstanding operations.
	//
	// Because other operations simply check for s.closed != 0, we will permit
	// double Close, which would increment s.closed beyond 1.
	if atomic.AddUint32(&c.closed, 1) != 1 {
		// Multiple Close calls.
		return nil
	}

	return os.NewSyscallError("close", c.fd.Close())
}

// CloseRead shuts down the reading side of the Conn. Most callers should just
// use Close.
func (c *Conn) CloseRead() error { return c.Shutdown(unix.SHUT_RD) }

// CloseWrite shuts down the writing side of the Conn. Most callers should just
// use Close.
func (c *Conn) CloseWrite() error { return c.Shutdown(unix.SHUT_WR) }

// Read reads directly from the underlying file descriptor.
func (c *Conn) Read(b []byte) (int, error) { return c.fd.Read(b) }

// ReadContext reads from the underlying file descriptor with added support for
// context cancelation.
func (c *Conn) ReadContext(ctx context.Context, b []byte) (int, error) {
	if c.facts.isStream && len(b) > maxRW {
		b = b[:maxRW]
	}

	n, err := readT(c, ctx, "read", func(fd int) (int, error) {
		return unix.Read(fd, b)
	})
	if n == 0 && err == nil && c.facts.zeroReadIsEOF {
		return 0, io.EOF
	}

	return n, os.NewSyscallError("read", err)
}

// Write writes directly to the underlying file descriptor.
func (c *Conn) Write(b []byte) (int, error) { return c.fd.Write(b) }

// WriteContext writes to the underlying file descriptor with added support for
// context cancelation.
func (c *Conn) WriteContext(ctx context.Context, b []byte) (int, error) {
	var (
		n, nn int
		err   error
	)

	doErr := c.write(ctx, "write", func(fd int) error {
		max := len(b)
		if c.facts.isStream && max-nn > maxRW {
			max = nn + maxRW
		}

		n, err = unix.Write(fd, b[nn:max])
		if n > 0 {
			nn += n
		}
		if nn == len(b) {
			return err
		}
		if n == 0 && err == nil {
			err = io.ErrUnexpectedEOF
			return nil
		}

		return err
	})
	if doErr != nil {
		return 0, doErr
	}

	return nn, os.NewSyscallError("write", err)
}

// SetDeadline sets both the read and write deadlines associated with the Conn.
func (c *Conn) SetDeadline(t time.Time) error { return c.fd.SetDeadline(t) }

// SetReadDeadline sets the read deadline associated with the Conn.
func (c *Conn) SetReadDeadline(t time.Time) error { return c.fd.SetReadDeadline(t) }

// SetWriteDeadline sets the write deadline associated with the Conn.
func (c *Conn) SetWriteDeadline(t time.Time) error { return c.fd.SetWriteDeadline(t) }

// ReadBuffer gets the size of the operating system's receive buffer associated
// with the Conn.
func (c *Conn) ReadBuffer() (int, error) {
	return c.GetsockoptInt(unix.SOL_SOCKET, unix.SO_RCVBUF)
}

// WriteBuffer gets the size of the operating system's transmit buffer
// associated with the Conn.
func (c *Conn) WriteBuffer() (int, error) {
	return c.GetsockoptInt(unix.SOL_SOCKET, unix.SO_SNDBUF)
}

// SetReadBuffer sets the size of the operating system's receive buffer
// associated with the Conn.
//
// When called with elevated privileges on Linux, the SO_RCVBUFFORCE option will
// be used to override operating system limits. Otherwise SO_RCVBUF is used
// (which obeys operating system limits).
func (c *Conn) SetReadBuffer(bytes int) error { return c.setReadBuffer(bytes) }

// SetWriteBuffer sets the size of the operating system's transmit buffer
// associated with the Conn.
//
// When called with elevated privileges on Linux, the SO_SNDBUFFORCE option will
// be used to override operating system limits. Otherwise SO_SNDBUF is used
// (which obeys operating system limits).
func (c *Conn) SetWriteBuffer(bytes int) error { return c.setWriteBuffer(bytes) }

// SyscallConn returns a raw network connection. This implements the
// syscall.Conn interface.
//
// SyscallConn is intended for advanced use cases, such as getting and setting
// arbitrary socket options using the socket's file descriptor. If possible,
// those operations should be performed using methods on Conn instead.
//
// Once invoked, it is the caller's responsibility to ensure that operations
// performed using Conn and the syscall.RawConn do not conflict with each other.
func (c *Conn) SyscallConn() (syscall.RawConn, error) {
	if atomic.LoadUint32(&c.closed) != 0 {
		return nil, os.NewSyscallError("syscallconn", unix.EBADF)
	}

	// TODO(mdlayher): mutex or similar to enforce syscall.RawConn contract of
	// FD remaining valid for duration of calls?
	return c.rc, nil
}

// Socket wraps the socket(2) system call to produce a Conn. domain, typ, and
// proto are passed directly to socket(2), and name should be a unique name for
// the socket type such as "netlink" or "vsock".
//
// The cfg parameter specifies optional configuration for the Conn. If nil, no
// additional configuration will be applied.
//
// If the operating system supports SOCK_CLOEXEC and SOCK_NONBLOCK, they are
// automatically applied to typ to mirror the standard library's socket flag
// behaviors.
func Socket(domain, typ, proto int, name string, cfg *Config) (*Conn, error) {
	if cfg == nil {
		cfg = &Config{}
	}

	if cfg.NetNS == 0 {
		// Non-Linux or no network namespace.
		return socket(domain, typ, proto, name)
	}

	// Linux only: create Conn in the specified network namespace.
	return withNetNS(cfg.NetNS, func() (*Conn, error) {
		return socket(domain, typ, proto, name)
	})
}

// socket is the internal, cross-platform entry point for socket(2).
func socket(domain, typ, proto int, name string) (*Conn, error) {
	var (
		fd  int
		err error
	)

	for {
		fd, err = unix.Socket(domain, typ|socketFlags, proto)
		switch {
		case err == nil:
			// Some OSes already set CLOEXEC with typ.
			if !flagCLOEXEC {
				unix.CloseOnExec(fd)
			}

			// No error, prepare the Conn.
			return New(fd, name)
		case !ready(err):
			// System call interrupted or not ready, try again.
			continue
		case err == unix.EINVAL, err == unix.EPROTONOSUPPORT:
			// On Linux, SOCK_NONBLOCK and SOCK_CLOEXEC were introduced in
			// 2.6.27. On FreeBSD, both flags were introduced in FreeBSD 10.
			// EINVAL and EPROTONOSUPPORT check for earlier versions of these
			// OSes respectively.
			//
			// Mirror what the standard library does when creating file
			// descriptors: avoid racing a fork/exec with the creation of new
			// file descriptors, so that child processes do not inherit socket
			// file descriptors unexpectedly.
			//
			// For a more thorough explanation, see similar work in the Go tree:
			// func sysSocket in net/sock_cloexec.go, as well as the detailed
			// comment in syscall/exec_unix.go.
			syscall.ForkLock.RLock()
			fd, err = unix.Socket(domain, typ, proto)
			if err != nil {
				syscall.ForkLock.RUnlock()
				return nil, os.NewSyscallError("socket", err)
			}
			unix.CloseOnExec(fd)
			syscall.ForkLock.RUnlock()

			return New(fd, name)
		default:
			// Unhandled error.
			return nil, os.NewSyscallError("socket", err)
		}
	}
}

// FileConn returns a copy of the network connection corresponding to the open
// file. It is the caller's responsibility to close the file when finished.
// Closing the Conn does not affect the File, and closing the File does not
// affect the Conn.
func FileConn(f *os.File, name string) (*Conn, error) {
	// First we'll try to do fctnl(2) with F_DUPFD_CLOEXEC because we can dup
	// the file descriptor and set the flag in one syscall.
	fd, err := unix.FcntlInt(f.Fd(), unix.F_DUPFD_CLOEXEC, 0)
	switch err {
	case nil:
		// OK, ready to set up non-blocking I/O.
		return New(fd, name)
	case unix.EINVAL:
		// The kernel rejected our fcntl(2), fall back to separate dup(2) and
		// setting close on exec.
		//
		// Mirror what the standard library does when creating file descriptors:
		// avoid racing a fork/exec with the creation of new file descriptors,
		// so that child processes do not inherit socket file descriptors
		// unexpectedly.
		syscall.ForkLock.RLock()
		fd, err := unix.Dup(fd)
		if err != nil {
			syscall.ForkLock.RUnlock()
			return nil, os.NewSyscallError("dup", err)
		}
		unix.CloseOnExec(fd)
		syscall.ForkLock.RUnlock()

		return New(fd, name)
	default:
		// Any other errors.
		return nil, os.NewSyscallError("fcntl", err)
	}
}

// New wraps an existing file descriptor to create a Conn. name should be a
// unique name for the socket type such as "netlink" or "vsock".
//
// Most callers should use Socket or FileConn to construct a Conn. New is
// intended for integrating with specific system calls which provide a file
// descriptor that supports asynchronous I/O. The file descriptor is immediately
// set to nonblocking mode and registered with Go's runtime network poller for
// future I/O operations.
//
// Unlike FileConn, New does not duplicate the existing file descriptor in any
// way. The returned Conn takes ownership of the underlying file descriptor.
func New(fd int, name string) (*Conn, error) {
	// All Conn I/O is nonblocking for integration with Go's runtime network
	// poller. Depending on the OS this might already be set but it can't hurt
	// to set it again.
	if err := unix.SetNonblock(fd, true); err != nil {
		return nil, os.NewSyscallError("setnonblock", err)
	}

	// os.NewFile registers the non-blocking file descriptor with the runtime
	// poller, which is then used for most subsequent operations except those
	// that require raw I/O via SyscallConn.
	//
	// See also: https://golang.org/pkg/os/#NewFile
	f := os.NewFile(uintptr(fd), name)
	rc, err := f.SyscallConn()
	if err != nil {
		return nil, err
	}

	c := &Conn{
		name: name,
		fd:   f,
		rc:   rc,
	}

	// Probe the file descriptor for socket settings.
	sotype, err := c.GetsockoptInt(unix.SOL_SOCKET, unix.SO_TYPE)
	switch {
	case err == nil:
		// File is a socket, check its properties.
		c.facts = facts{
			isStream:      sotype == unix.SOCK_STREAM,
			zeroReadIsEOF: sotype != unix.SOCK_DGRAM && sotype != unix.SOCK_RAW,
		}
	case errors.Is(err, unix.ENOTSOCK):
		// File is not a socket, treat it as a regular file.
		c.facts = facts{
			isStream:      true,
			zeroReadIsEOF: true,
		}
	default:
		return nil, err
	}

	return c, nil
}

// Low-level methods which provide raw system call access.

// Accept wraps accept(2) or accept4(2) depending on the operating system, but
// returns a Conn for the accepted connection rather than a raw file descriptor.
//
// If the operating system supports accept4(2) (which allows flags),
// SOCK_CLOEXEC and SOCK_NONBLOCK are automatically applied to flags to mirror
// the standard library's socket flag behaviors.
//
// If the operating system only supports accept(2) (which does not allow flags)
// and flags is not zero, an error will be returned.
//
// Accept obeys context cancelation and uses the deadline set on the context to
// cancel accepting the next connection. If a deadline is set on ctx, this
// deadline will override any previous deadlines set using SetDeadline or
// SetReadDeadline. Upon return, the read deadline is cleared.
func (c *Conn) Accept(ctx context.Context, flags int) (*Conn, unix.Sockaddr, error) {
	type ret struct {
		nfd int
		sa  unix.Sockaddr
	}

	r, err := readT(c, ctx, sysAccept, func(fd int) (ret, error) {
		// Either accept(2) or accept4(2) depending on the OS.
		nfd, sa, err := accept(fd, flags|socketFlags)
		return ret{nfd, sa}, err
	})
	if err != nil {
		// internal/poll, context error, or user function error.
		return nil, nil, err
	}

	// Successfully accepted a connection, wrap it in a Conn for use by the
	// caller.
	ac, err := New(r.nfd, c.name)
	if err != nil {
		return nil, nil, err
	}

	return ac, r.sa, nil
}

// Bind wraps bind(2).
func (c *Conn) Bind(sa unix.Sockaddr) error {
	return c.control("bind", func(fd int) error { return unix.Bind(fd, sa) })
}

// Connect wraps connect(2). In order to verify that the underlying socket is
// connected to a remote peer, Connect calls getpeername(2) and returns the
// unix.Sockaddr from that call.
//
// Connect obeys context cancelation and uses the deadline set on the context to
// cancel connecting to a remote peer. If a deadline is set on ctx, this
// deadline will override any previous deadlines set using SetDeadline or
// SetWriteDeadline. Upon return, the write deadline is cleared.
func (c *Conn) Connect(ctx context.Context, sa unix.Sockaddr) (unix.Sockaddr, error) {
	const op = "connect"

	// TODO(mdlayher): it would seem that trying to connect to unbound vsock
	// listeners by calling Connect multiple times results in ECONNRESET for the
	// first and nil error for subsequent calls. Do we need to memoize the
	// error? Check what the stdlib behavior is.

	var (
		// Track progress between invocations of the write closure. We don't
		// have an explicit WaitWrite call like internal/poll does, so we have
		// to wait until the runtime calls the closure again to indicate we can
		// write.
		progress uint32

		// Capture closure sockaddr and error.
		rsa unix.Sockaddr
		err error
	)

	doErr := c.write(ctx, op, func(fd int) error {
		if atomic.AddUint32(&progress, 1) == 1 {
			// First call: initiate connect.
			return unix.Connect(fd, sa)
		}

		// Subsequent calls: the runtime network poller indicates fd is
		// writable. Check for errno.
		errno, gerr := c.GetsockoptInt(unix.SOL_SOCKET, unix.SO_ERROR)
		if gerr != nil {
			return gerr
		}
		if errno != 0 {
			// Connection is still not ready or failed. If errno indicates
			// the socket is not ready, we will wait for the next write
			// event. Otherwise we propagate this errno back to the as a
			// permanent error.
			uerr := unix.Errno(errno)
			err = uerr
			return uerr
		}

		// According to internal/poll, it's possible for the runtime network
		// poller to spuriously wake us and return errno 0 for SO_ERROR.
		// Make sure we are actually connected to a peer.
		peer, err := c.Getpeername()
		if err != nil {
			// internal/poll unconditionally goes back to WaitWrite.
			// Synthesize an error that will do the same for us.
			return unix.EAGAIN
		}

		// Connection complete.
		rsa = peer
		return nil
	})
	if doErr != nil {
		// internal/poll or context error.
		return nil, doErr
	}

	if err == unix.EISCONN {
		// TODO(mdlayher): is this block obsolete with the addition of the
		// getsockopt SO_ERROR check above?
		//
		// EISCONN is reported if the socket is already established and should
		// not be treated as an error.
		//  - Darwin reports this for at least TCP sockets
		//  - Linux reports this for at least AF_VSOCK sockets
		return rsa, nil
	}

	return rsa, os.NewSyscallError(op, err)
}

// Getsockname wraps getsockname(2).
func (c *Conn) Getsockname() (unix.Sockaddr, error) {
	return controlT(c, "getsockname", unix.Getsockname)
}

// Getpeername wraps getpeername(2).
func (c *Conn) Getpeername() (unix.Sockaddr, error) {
	return controlT(c, "getpeername", unix.Getpeername)
}

// GetsockoptICMPv6Filter wraps getsockopt(2) for *unix.ICMPv6Filter values.
func (c *Conn) GetsockoptICMPv6Filter(level, opt int) (*unix.ICMPv6Filter, error) {
	return controlT(c, "getsockopt", func(fd int) (*unix.ICMPv6Filter, error) {
		return unix.GetsockoptICMPv6Filter(fd, level, opt)
	})
}

// GetsockoptInt wraps getsockopt(2) for integer values.
func (c *Conn) GetsockoptInt(level, opt int) (int, error) {
	return controlT(c, "getsockopt", func(fd int) (int, error) {
		return unix.GetsockoptInt(fd, level, opt)
	})
}

// GetsockoptString wraps getsockopt(2) for string values.
func (c *Conn) GetsockoptString(level, opt int) (string, error) {
	return controlT(c, "getsockopt", func(fd int) (string, error) {
		return unix.GetsockoptString(fd, level, opt)
	})
}

// Listen wraps listen(2).
func (c *Conn) Listen(n int) error {
	return c.control("listen", func(fd int) error { return unix.Listen(fd, n) })
}

// Recvmsg wraps recvmsg(2).
func (c *Conn) Recvmsg(ctx context.Context, p, oob []byte, flags int) (int, int, int, unix.Sockaddr, error) {
	type ret struct {
		n, oobn, recvflags int
		from               unix.Sockaddr
	}

	r, err := readT(c, ctx, "recvmsg", func(fd int) (ret, error) {
		n, oobn, recvflags, from, err := unix.Recvmsg(fd, p, oob, flags)
		return ret{n, oobn, recvflags, from}, err
	})
	if r.n == 0 && err == nil && c.facts.zeroReadIsEOF {
		return 0, 0, 0, nil, io.EOF
	}

	return r.n, r.oobn, r.recvflags, r.from, err
}

// Recvfrom wraps recvfrom(2).
func (c *Conn) Recvfrom(ctx context.Context, p []byte, flags int) (int, unix.Sockaddr, error) {
	type ret struct {
		n    int
		addr unix.Sockaddr
	}

	out, err := readT(c, ctx, "recvfrom", func(fd int) (ret, error) {
		n, addr, err := unix.Recvfrom(fd, p, flags)
		return ret{n, addr}, err
	})
	if out.n == 0 && err == nil && c.facts.zeroReadIsEOF {
		return 0, nil, io.EOF
	}

	return out.n, out.addr, err
}

// Sendmsg wraps sendmsg(2).
func (c *Conn) Sendmsg(ctx context.Context, p, oob []byte, to unix.Sockaddr, flags int) (int, error) {
	return writeT(c, ctx, "sendmsg", func(fd int) (int, error) {
		return unix.SendmsgN(fd, p, oob, to, flags)
	})
}

// Sendto wraps sendto(2).
func (c *Conn) Sendto(ctx context.Context, p []byte, flags int, to unix.Sockaddr) error {
	return c.write(ctx, "sendto", func(fd int) error {
		return unix.Sendto(fd, p, flags, to)
	})
}

// SetsockoptICMPv6Filter wraps setsockopt(2) for *unix.ICMPv6Filter values.
func (c *Conn) SetsockoptICMPv6Filter(level, opt int, filter *unix.ICMPv6Filter) error {
	return c.control("setsockopt", func(fd int) error {
		return unix.SetsockoptICMPv6Filter(fd, level, opt, filter)
	})
}

// SetsockoptInt wraps setsockopt(2) for integer values.
func (c *Conn) SetsockoptInt(level, opt, value int) error {
	return c.control("setsockopt", func(fd int) error {
		return unix.SetsockoptInt(fd, level, opt, value)
	})
}

// SetsockoptString wraps setsockopt(2) for string values.
func (c *Conn) SetsockoptString(level, opt int, value string) error {
	return c.control("setsockopt", func(fd int) error {
		return unix.SetsockoptString(fd, level, opt, value)
	})
}

// Shutdown wraps shutdown(2).
func (c *Conn) Shutdown(how int) error {
	return c.control("shutdown", func(fd int) error { return unix.Shutdown(fd, how) })
}

// Conn low-level read/write/control functions. These functions mirror the
// syscall.RawConn APIs but the input closures return errors rather than
// booleans.

// read wraps readT to execute a function and capture its error result. This is
// a convenience wrapper for functions which don't return any extra values.
func (c *Conn) read(ctx context.Context, op string, f func(fd int) error) error {
	_, err := readT(c, ctx, op, func(fd int) (struct{}, error) {
		return struct{}{}, f(fd)
	})
	return err
}

// write executes f, a write function, against the associated file descriptor.
// op is used to create an *os.SyscallError if the file descriptor is closed.
func (c *Conn) write(ctx context.Context, op string, f func(fd int) error) error {
	_, err := writeT(c, ctx, op, func(fd int) (struct{}, error) {
		return struct{}{}, f(fd)
	})
	return err
}

// readT executes c.rc.Read for op using the input function, returning a newly
// allocated result T.
func readT[T any](c *Conn, ctx context.Context, op string, f func(fd int) (T, error)) (T, error) {
	return rwT(c, rwContext[T]{
		Context: ctx,
		Type:    read,
		Op:      op,
		Do:      f,
	})
}

// writeT executes c.rc.Write for op using the input function, returning a newly
// allocated result T.
func writeT[T any](c *Conn, ctx context.Context, op string, f func(fd int) (T, error)) (T, error) {
	return rwT(c, rwContext[T]{
		Context: ctx,
		Type:    write,
		Op:      op,
		Do:      f,
	})
}

// readWrite indicates if an operation intends to read or write.
type readWrite bool

// Possible readWrite values.
const (
	read  readWrite = false
	write readWrite = true
)

// An rwContext provides arguments to rwT.
type rwContext[T any] struct {
	// The caller's context passed for cancelation.
	Context context.Context

	// The type of an operation: read or write.
	Type readWrite

	// The name of the operation used in errors.
	Op string

	// The actual function to perform.
	Do func(fd int) (T, error)
}

// rwT executes c.rc.Read or c.rc.Write (depending on the value of rw.Type) for
// rw.Op using the input function, returning a newly allocated result T.
//
// It obeys context cancelation and the rw.Context must not be nil.
func rwT[T any](c *Conn, rw rwContext[T]) (T, error) {
	if atomic.LoadUint32(&c.closed) != 0 {
		// If the file descriptor is already closed, do nothing.
		return *new(T), os.NewSyscallError(rw.Op, unix.EBADF)
	}

	if err := rw.Context.Err(); err != nil {
		// Early exit due to context cancel.
		return *new(T), os.NewSyscallError(rw.Op, err)
	}

	var (
		// The read or write function used to access the runtime network poller.
		poll func(func(uintptr) bool) error

		// The read or write function used to set the matching deadline.
		deadline func(time.Time) error
	)

	if rw.Type == write {
		poll = c.rc.Write
		deadline = c.SetWriteDeadline
	} else {
		poll = c.rc.Read
		deadline = c.SetReadDeadline
	}

	var (
		// Whether or not the context carried a deadline we are actively using
		// for cancelation.
		setDeadline bool

		// Signals for the cancelation watcher goroutine.
		wg    sync.WaitGroup
		doneC = make(chan struct{})

		// Atomic: reports whether we have to disarm the deadline.
		needDisarm atomic.Bool
	)

	// On cancel, clean up the watcher.
	defer func() {
		close(doneC)
		wg.Wait()
	}()

	if d, ok := rw.Context.Deadline(); ok {
		// The context has an explicit deadline. We will use it for cancelation
		// but disarm it after poll for the next call.
		if err := deadline(d); err != nil {
			return *new(T), err
		}
		setDeadline = true
		needDisarm.Store(true)
	} else {
		// The context does not have an explicit deadline. We have to watch for
		// cancelation so we can propagate that signal to immediately unblock
		// the runtime network poller.
		//
		// TODO(mdlayher): is it possible to detect a background context vs a
		// context with possible future cancel?
		wg.Add(1)
		go func() {
			defer wg.Done()

			select {
			case <-rw.Context.Done():
				// Cancel the operation. Make the caller disarm after poll
				// returns.
				needDisarm.Store(true)
				_ = deadline(time.Unix(0, 1))
			case <-doneC:
				// Nothing to do.
			}
		}()
	}

	var (
		t   T
		err error
	)

	pollErr := poll(func(fd uintptr) bool {
		t, err = rw.Do(int(fd))
		return ready(err)
	})

	if needDisarm.Load() {
		_ = deadline(time.Time{})
	}

	if pollErr != nil {
		if rw.Context.Err() != nil || (setDeadline && errors.Is(pollErr, os.ErrDeadlineExceeded)) {
			// The caller canceled the operation or we set a deadline internally
			// and it was reached.
			//
			// Unpack a plain context error. We wait for the context to be done
			// to synchronize state externally. Otherwise we have noticed I/O
			// timeout wakeups when we set a deadline but the context was not
			// yet marked done.
			<-rw.Context.Done()
			return *new(T), os.NewSyscallError(rw.Op, rw.Context.Err())
		}

		// Error from syscall.RawConn methods. Conventionally the standard
		// library does not wrap internal/poll errors in os.NewSyscallError.
		return *new(T), pollErr
	}

	// Result from user function.
	return t, os.NewSyscallError(rw.Op, err)
}

// control executes Conn.control for op using the input function.
func (c *Conn) control(op string, f func(fd int) error) error {
	_, err := controlT(c, op, func(fd int) (struct{}, error) {
		return struct{}{}, f(fd)
	})
	return err
}

// controlT executes c.rc.Control for op using the input function, returning a
// newly allocated result T.
func controlT[T any](c *Conn, op string, f func(fd int) (T, error)) (T, error) {
	if atomic.LoadUint32(&c.closed) != 0 {
		// If the file descriptor is already closed, do nothing.
		return *new(T), os.NewSyscallError(op, unix.EBADF)
	}

	var (
		t   T
		err error
	)

	doErr := c.rc.Control(func(fd uintptr) {
		// Repeatedly attempt the syscall(s) invoked by f until completion is
		// indicated by the return value of ready or the context is canceled.
		//
		// The last values for t and err are captured outside of the closure for
		// use when the loop breaks.
		for {
			t, err = f(int(fd))
			if ready(err) {
				return
			}
		}
	})
	if doErr != nil {
		// Error from syscall.RawConn methods. Conventionally the standard
		// library does not wrap internal/poll errors in os.NewSyscallError.
		return *new(T), doErr
	}

	// Result from user function.
	return t, os.NewSyscallError(op, err)
}

// ready indicates readiness based on the value of err.
func ready(err error) bool {
	switch err {
	case unix.EAGAIN, unix.EINPROGRESS, unix.EINTR:
		// When a socket is in non-blocking mode, we might see a variety of errors:
		//  - EAGAIN: most common case for a socket read not being ready
		//  - EINPROGRESS: reported by some sockets when first calling connect
		//  - EINTR: system call interrupted, more frequently occurs in Go 1.14+
		//    because goroutines can be asynchronously preempted
		//
		// Return false to let the poller wait for readiness. See the source code
		// for internal/poll.FD.RawRead for more details.
		return false
	default:
		// Ready regardless of whether there was an error or no error.
		return true
	}
}

// Darwin and FreeBSD can't read or write 2GB+ files at a time,
// even on 64-bit systems.
// The same is true of socket implementations on many systems.
// See golang.org/issue/7812 and golang.org/issue/16266.
// Use 1GB instead of, say, 2GB-1, to keep subsequent reads aligned.
const maxRW = 1 << 30
