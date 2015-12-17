package tcp

import (
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"sync"
	"time"
)

const (
	// DefaultTimeout is the default length of time to wait for first byte.
	DefaultTimeout = 30 * time.Second
)

// Mux multiplexes a network connection.
type Mux struct {
	ln net.Listener
	m  map[byte]*listener

	wg sync.WaitGroup

	// The amount of time to wait for the first header byte.
	Timeout time.Duration

	// Out-of-band error logger
	Logger *log.Logger
}

// NewMux returns a new instance of Mux for ln.
func NewMux() *Mux {
	return &Mux{
		m:       make(map[byte]*listener),
		Timeout: DefaultTimeout,
		Logger:  log.New(os.Stderr, "[tcp] ", log.LstdFlags),
	}
}

// Serve handles connections from ln and multiplexes then across registered listener.
func (mux *Mux) Serve(ln net.Listener) error {
	for {
		// Wait for the next connection.
		// If it returns a temporary error then simply retry.
		// If it returns any other error then exit immediately.
		conn, err := ln.Accept()
		if err, ok := err.(interface {
			Temporary() bool
		}); ok && err.Temporary() {
			continue
		}
		if err != nil {
			// Wait for all connections to be demux
			mux.wg.Wait()
			for _, ln := range mux.m {
				close(ln.c)
			}
			return err
		}

		// Demux in a goroutine to
		mux.wg.Add(1)
		go mux.handleConn(conn)
	}
}

func (mux *Mux) handleConn(conn net.Conn) {
	defer mux.wg.Done()
	// Set a read deadline so connections with no data don't timeout.
	if err := conn.SetReadDeadline(time.Now().Add(mux.Timeout)); err != nil {
		conn.Close()
		mux.Logger.Printf("tcp.Mux: cannot set read deadline: %s", err)
		return
	}

	// Read first byte from connection to determine handler.
	var typ [1]byte
	if _, err := io.ReadFull(conn, typ[:]); err != nil {
		conn.Close()
		mux.Logger.Printf("tcp.Mux: cannot read header byte: %s", err)
		return
	}

	// Reset read deadline and let the listener handle that.
	if err := conn.SetReadDeadline(time.Time{}); err != nil {
		conn.Close()
		mux.Logger.Printf("tcp.Mux: cannot reset set read deadline: %s", err)
		return
	}

	// Retrieve handler based on first byte.
	handler := mux.m[typ[0]]
	if handler == nil {
		conn.Close()
		mux.Logger.Printf("tcp.Mux: handler not registered: %d", typ[0])
		return
	}

	// Send connection to handler.  The handler is responsible for closing the connection.
	handler.c <- conn
}

// Listen returns a listener identified by header.
// Any connection accepted by mux is multiplexed based on the initial header byte.
func (mux *Mux) Listen(header byte) net.Listener {
	// Ensure two listeners are not created for the same header byte.
	if _, ok := mux.m[header]; ok {
		panic(fmt.Sprintf("listener already registered under header byte: %d", header))
	}

	// Create a new listener and assign it.
	ln := &listener{
		c: make(chan net.Conn),
	}
	mux.m[header] = ln

	return ln
}

// listener is a receiver for connections received by Mux.
type listener struct {
	c chan net.Conn
}

// Accept waits for and returns the next connection to the listener.
func (ln *listener) Accept() (c net.Conn, err error) {
	conn, ok := <-ln.c
	if !ok {
		return nil, errors.New("network connection closed")
	}
	return conn, nil
}

// Close is a no-op. The mux's listener should be closed instead.
func (ln *listener) Close() error { return nil }

// Addr always returns nil.
func (ln *listener) Addr() net.Addr { return nil }

// Dial connects to a remote mux listener with a given header byte.
func Dial(network, address string, header byte) (net.Conn, error) {
	conn, err := net.Dial(network, address)
	if err != nil {
		return nil, err
	}

	if _, err := conn.Write([]byte{header}); err != nil {
		return nil, fmt.Errorf("write mux header: %s", err)
	}

	return conn, nil
}
