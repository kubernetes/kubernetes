package sockets

import (
	"errors"
	"net"
	"sync"
)

var errClosed = errors.New("use of closed network connection")

// InmemSocket implements net.Listener using in-memory only connections.
type InmemSocket struct {
	chConn  chan net.Conn
	chClose chan struct{}
	addr    string
	mu      sync.Mutex
}

// dummyAddr is used to satisfy net.Addr for the in-mem socket
// it is just stored as a string and returns the string for all calls
type dummyAddr string

// NewInmemSocket creates an in-memory only net.Listener
// The addr argument can be any string, but is used to satisfy the `Addr()` part
// of the net.Listener interface
func NewInmemSocket(addr string, bufSize int) *InmemSocket {
	return &InmemSocket{
		chConn:  make(chan net.Conn, bufSize),
		chClose: make(chan struct{}),
		addr:    addr,
	}
}

// Addr returns the socket's addr string to satisfy net.Listener
func (s *InmemSocket) Addr() net.Addr {
	return dummyAddr(s.addr)
}

// Accept implements the Accept method in the Listener interface; it waits for the next call and returns a generic Conn.
func (s *InmemSocket) Accept() (net.Conn, error) {
	select {
	case conn := <-s.chConn:
		return conn, nil
	case <-s.chClose:
		return nil, errClosed
	}
}

// Close closes the listener. It will be unavailable for use once closed.
func (s *InmemSocket) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	select {
	case <-s.chClose:
	default:
		close(s.chClose)
	}
	return nil
}

// Dial is used to establish a connection with the in-mem server
func (s *InmemSocket) Dial(network, addr string) (net.Conn, error) {
	srvConn, clientConn := net.Pipe()
	select {
	case s.chConn <- srvConn:
	case <-s.chClose:
		return nil, errClosed
	}

	return clientConn, nil
}

// Network returns the addr string, satisfies net.Addr
func (a dummyAddr) Network() string {
	return string(a)
}

// String returns the string form
func (a dummyAddr) String() string {
	return string(a)
}

// timeoutError is used when there is a timeout with a connection
// this implements the net.Error interface
type timeoutError struct{}

func (e *timeoutError) Error() string   { return "i/o timeout" }
func (e *timeoutError) Timeout() bool   { return true }
func (e *timeoutError) Temporary() bool { return true }
