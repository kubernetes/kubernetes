package proxyproto

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	// prefix is the string we look for at the start of a connection
	// to check if this connection is using the proxy protocol
	prefix    = []byte("PROXY ")
	prefixLen = len(prefix)

	ErrInvalidUpstream = errors.New("upstream connection address not trusted for PROXY information")
)

// SourceChecker can be used to decide whether to trust the PROXY info or pass
// the original connection address through. If set, the connecting address is
// passed in as an argument. If the function returns an error due to the source
// being disallowed, it should return ErrInvalidUpstream.
//
// If error is not nil, the call to Accept() will fail. If the reason for
// triggering this failure is due to a disallowed source, it should return
// ErrInvalidUpstream.
//
// If bool is true, the PROXY-set address is used.
//
// If bool is false, the connection's remote address is used, rather than the
// address claimed in the PROXY info.
type SourceChecker func(net.Addr) (bool, error)

// Listener is used to wrap an underlying listener,
// whose connections may be using the HAProxy Proxy Protocol (version 1).
// If the connection is using the protocol, the RemoteAddr() will return
// the correct client address.
//
// Optionally define ProxyHeaderTimeout to set a maximum time to
// receive the Proxy Protocol Header. Zero means no timeout.
type Listener struct {
	Listener           net.Listener
	ProxyHeaderTimeout time.Duration
	SourceCheck        SourceChecker
	UnknownOK          bool // allow PROXY UNKNOWN
}

// Conn is used to wrap and underlying connection which
// may be speaking the Proxy Protocol. If it is, the RemoteAddr() will
// return the address of the client instead of the proxy address.
type Conn struct {
	bufReader          *bufio.Reader
	conn               net.Conn
	dstAddr            *net.TCPAddr
	srcAddr            *net.TCPAddr
	useConnAddr        bool
	once               sync.Once
	proxyHeaderTimeout time.Duration
	unknownOK          bool
}

// Accept waits for and returns the next connection to the listener.
func (p *Listener) Accept() (net.Conn, error) {
	// Get the underlying connection
	conn, err := p.Listener.Accept()
	if err != nil {
		return nil, err
	}
	var useConnAddr bool
	if p.SourceCheck != nil {
		allowed, err := p.SourceCheck(conn.RemoteAddr())
		if err != nil {
			return nil, err
		}
		if !allowed {
			useConnAddr = true
		}
	}
	newConn := NewConn(conn, p.ProxyHeaderTimeout)
	newConn.useConnAddr = useConnAddr
	newConn.unknownOK = p.UnknownOK
	return newConn, nil
}

// Close closes the underlying listener.
func (p *Listener) Close() error {
	return p.Listener.Close()
}

// Addr returns the underlying listener's network address.
func (p *Listener) Addr() net.Addr {
	return p.Listener.Addr()
}

// NewConn is used to wrap a net.Conn that may be speaking
// the proxy protocol into a proxyproto.Conn
func NewConn(conn net.Conn, timeout time.Duration) *Conn {
	pConn := &Conn{
		bufReader:          bufio.NewReader(conn),
		conn:               conn,
		proxyHeaderTimeout: timeout,
	}
	return pConn
}

// Read is check for the proxy protocol header when doing
// the initial scan. If there is an error parsing the header,
// it is returned and the socket is closed.
func (p *Conn) Read(b []byte) (int, error) {
	var err error
	p.once.Do(func() { err = p.checkPrefix() })
	if err != nil {
		return 0, err
	}
	return p.bufReader.Read(b)
}

func (p *Conn) ReadFrom(r io.Reader) (int64, error) {
	if rf, ok := p.conn.(io.ReaderFrom); ok {
		return rf.ReadFrom(r)
	}
	return io.Copy(p.conn, r)
}

func (p *Conn) WriteTo(w io.Writer) (int64, error) {
	var err error
	p.once.Do(func() { err = p.checkPrefix() })
	if err != nil {
		return 0, err
	}
	return p.bufReader.WriteTo(w)
}

func (p *Conn) Write(b []byte) (int, error) {
	return p.conn.Write(b)
}

func (p *Conn) Close() error {
	return p.conn.Close()
}

func (p *Conn) LocalAddr() net.Addr {
	p.checkPrefixOnce()
	if p.dstAddr != nil && !p.useConnAddr {
		return p.dstAddr
	}
	return p.conn.LocalAddr()
}

// RemoteAddr returns the address of the client if the proxy
// protocol is being used, otherwise just returns the address of
// the socket peer. If there is an error parsing the header, the
// address of the client is not returned, and the socket is closed.
// Once implication of this is that the call could block if the
// client is slow. Using a Deadline is recommended if this is called
// before Read()
func (p *Conn) RemoteAddr() net.Addr {
	p.checkPrefixOnce()
	if p.srcAddr != nil && !p.useConnAddr {
		return p.srcAddr
	}
	return p.conn.RemoteAddr()
}

func (p *Conn) SetDeadline(t time.Time) error {
	return p.conn.SetDeadline(t)
}

func (p *Conn) SetReadDeadline(t time.Time) error {
	return p.conn.SetReadDeadline(t)
}

func (p *Conn) SetWriteDeadline(t time.Time) error {
	return p.conn.SetWriteDeadline(t)
}

func (p *Conn) checkPrefixOnce() {
	p.once.Do(func() {
		if err := p.checkPrefix(); err != nil && err != io.EOF {
			log.Printf("[ERR] Failed to read proxy prefix: %v", err)
			p.Close()
			p.bufReader = bufio.NewReader(p.conn)
		}
	})
}

func (p *Conn) checkPrefix() error {
	if p.proxyHeaderTimeout != 0 {
		readDeadLine := time.Now().Add(p.proxyHeaderTimeout)
		p.conn.SetReadDeadline(readDeadLine)
		defer p.conn.SetReadDeadline(time.Time{})
	}

	// Incrementally check each byte of the prefix
	for i := 1; i <= prefixLen; i++ {
		inp, err := p.bufReader.Peek(i)

		if err != nil {
			if neterr, ok := err.(net.Error); ok && neterr.Timeout() {
				return nil
			} else {
				return err
			}
		}

		// Check for a prefix mis-match, quit early
		if !bytes.Equal(inp, prefix[:i]) {
			return nil
		}
	}

	// Read the header line
	header, err := p.bufReader.ReadString('\n')
	if err != nil {
		p.conn.Close()
		return err
	}

	// Strip the carriage return and new line
	header = header[:len(header)-2]

	// Split on spaces, should be (PROXY <type> <src addr> <dst addr> <src port> <dst port>)
	parts := strings.Split(header, " ")
	if len(parts) < 2 {
		p.conn.Close()
		return fmt.Errorf("Invalid header line: %s", header)
	}

	// Verify the type is known
	switch parts[1] {
	case "UNKNOWN":
		if !p.unknownOK || len(parts) != 2 {
			p.conn.Close()
			return fmt.Errorf("Invalid UNKNOWN header line: %s", header)
		}
		p.useConnAddr = true
		return nil
	case "TCP4":
	case "TCP6":
	default:
		p.conn.Close()
		return fmt.Errorf("Unhandled address type: %s", parts[1])
	}

	if len(parts) != 6 {
		p.conn.Close()
		return fmt.Errorf("Invalid header line: %s", header)
	}

	// Parse out the source address
	ip := net.ParseIP(parts[2])
	if ip == nil {
		p.conn.Close()
		return fmt.Errorf("Invalid source ip: %s", parts[2])
	}
	port, err := strconv.Atoi(parts[4])
	if err != nil {
		p.conn.Close()
		return fmt.Errorf("Invalid source port: %s", parts[4])
	}
	p.srcAddr = &net.TCPAddr{IP: ip, Port: port}

	// Parse out the destination address
	ip = net.ParseIP(parts[3])
	if ip == nil {
		p.conn.Close()
		return fmt.Errorf("Invalid destination ip: %s", parts[3])
	}
	port, err = strconv.Atoi(parts[5])
	if err != nil {
		p.conn.Close()
		return fmt.Errorf("Invalid destination port: %s", parts[5])
	}
	p.dstAddr = &net.TCPAddr{IP: ip, Port: port}

	return nil
}
