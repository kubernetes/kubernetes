package winio

import (
	"fmt"
	"io"
	"net"
	"os"
	"syscall"
	"time"
	"unsafe"

	"github.com/Microsoft/go-winio/pkg/guid"
)

//sys bind(s syscall.Handle, name unsafe.Pointer, namelen int32) (err error) [failretval==socketError] = ws2_32.bind

const (
	afHvSock = 34 // AF_HYPERV

	socketError = ^uintptr(0)
)

// An HvsockAddr is an address for a AF_HYPERV socket.
type HvsockAddr struct {
	VMID      guid.GUID
	ServiceID guid.GUID
}

type rawHvsockAddr struct {
	Family    uint16
	_         uint16
	VMID      guid.GUID
	ServiceID guid.GUID
}

// Network returns the address's network name, "hvsock".
func (addr *HvsockAddr) Network() string {
	return "hvsock"
}

func (addr *HvsockAddr) String() string {
	return fmt.Sprintf("%s:%s", &addr.VMID, &addr.ServiceID)
}

// VsockServiceID returns an hvsock service ID corresponding to the specified AF_VSOCK port.
func VsockServiceID(port uint32) guid.GUID {
	g, _ := guid.FromString("00000000-facb-11e6-bd58-64006a7986d3")
	g.Data1 = port
	return g
}

func (addr *HvsockAddr) raw() rawHvsockAddr {
	return rawHvsockAddr{
		Family:    afHvSock,
		VMID:      addr.VMID,
		ServiceID: addr.ServiceID,
	}
}

func (addr *HvsockAddr) fromRaw(raw *rawHvsockAddr) {
	addr.VMID = raw.VMID
	addr.ServiceID = raw.ServiceID
}

// HvsockListener is a socket listener for the AF_HYPERV address family.
type HvsockListener struct {
	sock *win32File
	addr HvsockAddr
}

// HvsockConn is a connected socket of the AF_HYPERV address family.
type HvsockConn struct {
	sock          *win32File
	local, remote HvsockAddr
}

func newHvSocket() (*win32File, error) {
	fd, err := syscall.Socket(afHvSock, syscall.SOCK_STREAM, 1)
	if err != nil {
		return nil, os.NewSyscallError("socket", err)
	}
	f, err := makeWin32File(fd)
	if err != nil {
		syscall.Close(fd)
		return nil, err
	}
	f.socket = true
	return f, nil
}

// ListenHvsock listens for connections on the specified hvsock address.
func ListenHvsock(addr *HvsockAddr) (_ *HvsockListener, err error) {
	l := &HvsockListener{addr: *addr}
	sock, err := newHvSocket()
	if err != nil {
		return nil, l.opErr("listen", err)
	}
	sa := addr.raw()
	err = bind(sock.handle, unsafe.Pointer(&sa), int32(unsafe.Sizeof(sa)))
	if err != nil {
		return nil, l.opErr("listen", os.NewSyscallError("socket", err))
	}
	err = syscall.Listen(sock.handle, 16)
	if err != nil {
		return nil, l.opErr("listen", os.NewSyscallError("listen", err))
	}
	return &HvsockListener{sock: sock, addr: *addr}, nil
}

func (l *HvsockListener) opErr(op string, err error) error {
	return &net.OpError{Op: op, Net: "hvsock", Addr: &l.addr, Err: err}
}

// Addr returns the listener's network address.
func (l *HvsockListener) Addr() net.Addr {
	return &l.addr
}

// Accept waits for the next connection and returns it.
func (l *HvsockListener) Accept() (_ net.Conn, err error) {
	sock, err := newHvSocket()
	if err != nil {
		return nil, l.opErr("accept", err)
	}
	defer func() {
		if sock != nil {
			sock.Close()
		}
	}()
	c, err := l.sock.prepareIo()
	if err != nil {
		return nil, l.opErr("accept", err)
	}
	defer l.sock.wg.Done()

	// AcceptEx, per documentation, requires an extra 16 bytes per address.
	const addrlen = uint32(16 + unsafe.Sizeof(rawHvsockAddr{}))
	var addrbuf [addrlen * 2]byte

	var bytes uint32
	err = syscall.AcceptEx(l.sock.handle, sock.handle, &addrbuf[0], 0, addrlen, addrlen, &bytes, &c.o)
	_, err = l.sock.asyncIo(c, nil, bytes, err)
	if err != nil {
		return nil, l.opErr("accept", os.NewSyscallError("acceptex", err))
	}
	conn := &HvsockConn{
		sock: sock,
	}
	conn.local.fromRaw((*rawHvsockAddr)(unsafe.Pointer(&addrbuf[0])))
	conn.remote.fromRaw((*rawHvsockAddr)(unsafe.Pointer(&addrbuf[addrlen])))
	sock = nil
	return conn, nil
}

// Close closes the listener, causing any pending Accept calls to fail.
func (l *HvsockListener) Close() error {
	return l.sock.Close()
}

/* Need to finish ConnectEx handling
func DialHvsock(ctx context.Context, addr *HvsockAddr) (*HvsockConn, error) {
	sock, err := newHvSocket()
	if err != nil {
		return nil, err
	}
	defer func() {
		if sock != nil {
			sock.Close()
		}
	}()
	c, err := sock.prepareIo()
	if err != nil {
		return nil, err
	}
	defer sock.wg.Done()
	var bytes uint32
	err = windows.ConnectEx(windows.Handle(sock.handle), sa, nil, 0, &bytes, &c.o)
	_, err = sock.asyncIo(ctx, c, nil, bytes, err)
	if err != nil {
		return nil, err
	}
	conn := &HvsockConn{
		sock:   sock,
		remote: *addr,
	}
	sock = nil
	return conn, nil
}
*/

func (conn *HvsockConn) opErr(op string, err error) error {
	return &net.OpError{Op: op, Net: "hvsock", Source: &conn.local, Addr: &conn.remote, Err: err}
}

func (conn *HvsockConn) Read(b []byte) (int, error) {
	c, err := conn.sock.prepareIo()
	if err != nil {
		return 0, conn.opErr("read", err)
	}
	defer conn.sock.wg.Done()
	buf := syscall.WSABuf{Buf: &b[0], Len: uint32(len(b))}
	var flags, bytes uint32
	err = syscall.WSARecv(conn.sock.handle, &buf, 1, &bytes, &flags, &c.o, nil)
	n, err := conn.sock.asyncIo(c, &conn.sock.readDeadline, bytes, err)
	if err != nil {
		if _, ok := err.(syscall.Errno); ok {
			err = os.NewSyscallError("wsarecv", err)
		}
		return 0, conn.opErr("read", err)
	} else if n == 0 {
		err = io.EOF
	}
	return n, err
}

func (conn *HvsockConn) Write(b []byte) (int, error) {
	t := 0
	for len(b) != 0 {
		n, err := conn.write(b)
		if err != nil {
			return t + n, err
		}
		t += n
		b = b[n:]
	}
	return t, nil
}

func (conn *HvsockConn) write(b []byte) (int, error) {
	c, err := conn.sock.prepareIo()
	if err != nil {
		return 0, conn.opErr("write", err)
	}
	defer conn.sock.wg.Done()
	buf := syscall.WSABuf{Buf: &b[0], Len: uint32(len(b))}
	var bytes uint32
	err = syscall.WSASend(conn.sock.handle, &buf, 1, &bytes, 0, &c.o, nil)
	n, err := conn.sock.asyncIo(c, &conn.sock.writeDeadline, bytes, err)
	if err != nil {
		if _, ok := err.(syscall.Errno); ok {
			err = os.NewSyscallError("wsasend", err)
		}
		return 0, conn.opErr("write", err)
	}
	return n, err
}

// Close closes the socket connection, failing any pending read or write calls.
func (conn *HvsockConn) Close() error {
	return conn.sock.Close()
}

func (conn *HvsockConn) shutdown(how int) error {
	err := syscall.Shutdown(conn.sock.handle, syscall.SHUT_RD)
	if err != nil {
		return os.NewSyscallError("shutdown", err)
	}
	return nil
}

// CloseRead shuts down the read end of the socket.
func (conn *HvsockConn) CloseRead() error {
	err := conn.shutdown(syscall.SHUT_RD)
	if err != nil {
		return conn.opErr("close", err)
	}
	return nil
}

// CloseWrite shuts down the write end of the socket, notifying the other endpoint that
// no more data will be written.
func (conn *HvsockConn) CloseWrite() error {
	err := conn.shutdown(syscall.SHUT_WR)
	if err != nil {
		return conn.opErr("close", err)
	}
	return nil
}

// LocalAddr returns the local address of the connection.
func (conn *HvsockConn) LocalAddr() net.Addr {
	return &conn.local
}

// RemoteAddr returns the remote address of the connection.
func (conn *HvsockConn) RemoteAddr() net.Addr {
	return &conn.remote
}

// SetDeadline implements the net.Conn SetDeadline method.
func (conn *HvsockConn) SetDeadline(t time.Time) error {
	conn.SetReadDeadline(t)
	conn.SetWriteDeadline(t)
	return nil
}

// SetReadDeadline implements the net.Conn SetReadDeadline method.
func (conn *HvsockConn) SetReadDeadline(t time.Time) error {
	return conn.sock.SetReadDeadline(t)
}

// SetWriteDeadline implements the net.Conn SetWriteDeadline method.
func (conn *HvsockConn) SetWriteDeadline(t time.Time) error {
	return conn.sock.SetWriteDeadline(t)
}
