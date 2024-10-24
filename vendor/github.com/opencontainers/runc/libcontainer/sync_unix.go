package libcontainer

import (
	"fmt"
	"io"
	"os"
	"sync/atomic"

	"golang.org/x/sys/unix"
)

// syncSocket is a wrapper around a SOCK_SEQPACKET socket, providing
// packet-oriented methods. This is needed because SOCK_SEQPACKET does not
// allow for partial reads, but the Go stdlib treats it as a streamable source,
// which ends up making things like json.Decoder hang forever if the packet is
// bigger than the internal read buffer.
type syncSocket struct {
	f      *os.File
	closed atomic.Bool
}

func newSyncSocket(f *os.File) *syncSocket {
	return &syncSocket{f: f}
}

func (s *syncSocket) File() *os.File {
	return s.f
}

func (s *syncSocket) Close() error {
	// Even with errors from Close(), we have to assume the pipe was closed.
	s.closed.Store(true)
	return s.f.Close()
}

func (s *syncSocket) isClosed() bool {
	return s.closed.Load()
}

func (s *syncSocket) WritePacket(b []byte) (int, error) {
	return s.f.Write(b)
}

func (s *syncSocket) ReadPacket() ([]byte, error) {
	size, _, err := unix.Recvfrom(int(s.f.Fd()), nil, unix.MSG_TRUNC|unix.MSG_PEEK)
	if err != nil {
		return nil, fmt.Errorf("fetch packet length from socket: %w", err)
	}
	// We will only get a zero size if the socket has been closed from the
	// other end (otherwise recvfrom(2) will block until a packet is ready). In
	// addition, SOCK_SEQPACKET is treated as a stream source by Go stdlib so
	// returning io.EOF here is correct from that perspective too.
	if size == 0 {
		return nil, io.EOF
	}
	buf := make([]byte, size)
	n, err := s.f.Read(buf)
	if err != nil {
		return nil, err
	}
	if n != size {
		return nil, fmt.Errorf("packet read too short: expected %d byte packet but only %d bytes read", size, n)
	}
	return buf, nil
}

func (s *syncSocket) Shutdown(how int) error {
	if err := unix.Shutdown(int(s.f.Fd()), how); err != nil {
		return &os.PathError{Op: "shutdown", Path: s.f.Name() + " (sync pipe)", Err: err}
	}
	return nil
}

// newSyncSockpair returns a new SOCK_SEQPACKET unix socket pair to be used for
// runc-init synchronisation.
func newSyncSockpair(name string) (parent, child *syncSocket, err error) {
	fds, err := unix.Socketpair(unix.AF_LOCAL, unix.SOCK_SEQPACKET|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		return nil, nil, err
	}
	parentFile := os.NewFile(uintptr(fds[1]), name+"-p")
	childFile := os.NewFile(uintptr(fds[0]), name+"-c")
	return newSyncSocket(parentFile), newSyncSocket(childFile), nil
}
