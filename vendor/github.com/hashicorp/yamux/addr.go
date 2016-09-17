package yamux

import (
	"fmt"
	"net"
)

// hasAddr is used to get the address from the underlying connection
type hasAddr interface {
	LocalAddr() net.Addr
	RemoteAddr() net.Addr
}

// yamuxAddr is used when we cannot get the underlying address
type yamuxAddr struct {
	Addr string
}

func (*yamuxAddr) Network() string {
	return "yamux"
}

func (y *yamuxAddr) String() string {
	return fmt.Sprintf("yamux:%s", y.Addr)
}

// Addr is used to get the address of the listener.
func (s *Session) Addr() net.Addr {
	return s.LocalAddr()
}

// LocalAddr is used to get the local address of the
// underlying connection.
func (s *Session) LocalAddr() net.Addr {
	addr, ok := s.conn.(hasAddr)
	if !ok {
		return &yamuxAddr{"local"}
	}
	return addr.LocalAddr()
}

// RemoteAddr is used to get the address of remote end
// of the underlying connection
func (s *Session) RemoteAddr() net.Addr {
	addr, ok := s.conn.(hasAddr)
	if !ok {
		return &yamuxAddr{"remote"}
	}
	return addr.RemoteAddr()
}

// LocalAddr returns the local address
func (s *Stream) LocalAddr() net.Addr {
	return s.session.LocalAddr()
}

// LocalAddr returns the remote address
func (s *Stream) RemoteAddr() net.Addr {
	return s.session.RemoteAddr()
}
