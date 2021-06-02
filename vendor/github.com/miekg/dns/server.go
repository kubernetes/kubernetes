// DNS server implementation.

package dns

import (
	"context"
	"crypto/tls"
	"encoding/binary"
	"errors"
	"io"
	"net"
	"strings"
	"sync"
	"time"
)

// Default maximum number of TCP queries before we close the socket.
const maxTCPQueries = 128

// aLongTimeAgo is a non-zero time, far in the past, used for
// immediate cancelation of network operations.
var aLongTimeAgo = time.Unix(1, 0)

// Handler is implemented by any value that implements ServeDNS.
type Handler interface {
	ServeDNS(w ResponseWriter, r *Msg)
}

// The HandlerFunc type is an adapter to allow the use of
// ordinary functions as DNS handlers.  If f is a function
// with the appropriate signature, HandlerFunc(f) is a
// Handler object that calls f.
type HandlerFunc func(ResponseWriter, *Msg)

// ServeDNS calls f(w, r).
func (f HandlerFunc) ServeDNS(w ResponseWriter, r *Msg) {
	f(w, r)
}

// A ResponseWriter interface is used by an DNS handler to
// construct an DNS response.
type ResponseWriter interface {
	// LocalAddr returns the net.Addr of the server
	LocalAddr() net.Addr
	// RemoteAddr returns the net.Addr of the client that sent the current request.
	RemoteAddr() net.Addr
	// WriteMsg writes a reply back to the client.
	WriteMsg(*Msg) error
	// Write writes a raw buffer back to the client.
	Write([]byte) (int, error)
	// Close closes the connection.
	Close() error
	// TsigStatus returns the status of the Tsig.
	TsigStatus() error
	// TsigTimersOnly sets the tsig timers only boolean.
	TsigTimersOnly(bool)
	// Hijack lets the caller take over the connection.
	// After a call to Hijack(), the DNS package will not do anything with the connection.
	Hijack()
}

// A ConnectionStater interface is used by a DNS Handler to access TLS connection state
// when available.
type ConnectionStater interface {
	ConnectionState() *tls.ConnectionState
}

type response struct {
	closed         bool // connection has been closed
	hijacked       bool // connection has been hijacked by handler
	tsigTimersOnly bool
	tsigStatus     error
	tsigRequestMAC string
	tsigSecret     map[string]string // the tsig secrets
	udp            net.PacketConn    // i/o connection if UDP was used
	tcp            net.Conn          // i/o connection if TCP was used
	udpSession     *SessionUDP       // oob data to get egress interface right
	pcSession      net.Addr          // address to use when writing to a generic net.PacketConn
	writer         Writer            // writer to output the raw DNS bits
}

// handleRefused returns a HandlerFunc that returns REFUSED for every request it gets.
func handleRefused(w ResponseWriter, r *Msg) {
	m := new(Msg)
	m.SetRcode(r, RcodeRefused)
	w.WriteMsg(m)
}

// HandleFailed returns a HandlerFunc that returns SERVFAIL for every request it gets.
// Deprecated: This function is going away.
func HandleFailed(w ResponseWriter, r *Msg) {
	m := new(Msg)
	m.SetRcode(r, RcodeServerFailure)
	// does not matter if this write fails
	w.WriteMsg(m)
}

// ListenAndServe Starts a server on address and network specified Invoke handler
// for incoming queries.
func ListenAndServe(addr string, network string, handler Handler) error {
	server := &Server{Addr: addr, Net: network, Handler: handler}
	return server.ListenAndServe()
}

// ListenAndServeTLS acts like http.ListenAndServeTLS, more information in
// http://golang.org/pkg/net/http/#ListenAndServeTLS
func ListenAndServeTLS(addr, certFile, keyFile string, handler Handler) error {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return err
	}

	config := tls.Config{
		Certificates: []tls.Certificate{cert},
	}

	server := &Server{
		Addr:      addr,
		Net:       "tcp-tls",
		TLSConfig: &config,
		Handler:   handler,
	}

	return server.ListenAndServe()
}

// ActivateAndServe activates a server with a listener from systemd,
// l and p should not both be non-nil.
// If both l and p are not nil only p will be used.
// Invoke handler for incoming queries.
func ActivateAndServe(l net.Listener, p net.PacketConn, handler Handler) error {
	server := &Server{Listener: l, PacketConn: p, Handler: handler}
	return server.ActivateAndServe()
}

// Writer writes raw DNS messages; each call to Write should send an entire message.
type Writer interface {
	io.Writer
}

// Reader reads raw DNS messages; each call to ReadTCP or ReadUDP should return an entire message.
type Reader interface {
	// ReadTCP reads a raw message from a TCP connection. Implementations may alter
	// connection properties, for example the read-deadline.
	ReadTCP(conn net.Conn, timeout time.Duration) ([]byte, error)
	// ReadUDP reads a raw message from a UDP connection. Implementations may alter
	// connection properties, for example the read-deadline.
	ReadUDP(conn *net.UDPConn, timeout time.Duration) ([]byte, *SessionUDP, error)
}

// PacketConnReader is an optional interface that Readers can implement to support using generic net.PacketConns.
type PacketConnReader interface {
	Reader

	// ReadPacketConn reads a raw message from a generic net.PacketConn UDP connection. Implementations may
	// alter connection properties, for example the read-deadline.
	ReadPacketConn(conn net.PacketConn, timeout time.Duration) ([]byte, net.Addr, error)
}

// defaultReader is an adapter for the Server struct that implements the Reader and
// PacketConnReader interfaces using the readTCP, readUDP and readPacketConn funcs
// of the embedded Server.
type defaultReader struct {
	*Server
}

var _ PacketConnReader = defaultReader{}

func (dr defaultReader) ReadTCP(conn net.Conn, timeout time.Duration) ([]byte, error) {
	return dr.readTCP(conn, timeout)
}

func (dr defaultReader) ReadUDP(conn *net.UDPConn, timeout time.Duration) ([]byte, *SessionUDP, error) {
	return dr.readUDP(conn, timeout)
}

func (dr defaultReader) ReadPacketConn(conn net.PacketConn, timeout time.Duration) ([]byte, net.Addr, error) {
	return dr.readPacketConn(conn, timeout)
}

// DecorateReader is a decorator hook for extending or supplanting the functionality of a Reader.
// Implementations should never return a nil Reader.
// Readers should also implement the optional PacketConnReader interface.
// PacketConnReader is required to use a generic net.PacketConn.
type DecorateReader func(Reader) Reader

// DecorateWriter is a decorator hook for extending or supplanting the functionality of a Writer.
// Implementations should never return a nil Writer.
type DecorateWriter func(Writer) Writer

// A Server defines parameters for running an DNS server.
type Server struct {
	// Address to listen on, ":dns" if empty.
	Addr string
	// if "tcp" or "tcp-tls" (DNS over TLS) it will invoke a TCP listener, otherwise an UDP one
	Net string
	// TCP Listener to use, this is to aid in systemd's socket activation.
	Listener net.Listener
	// TLS connection configuration
	TLSConfig *tls.Config
	// UDP "Listener" to use, this is to aid in systemd's socket activation.
	PacketConn net.PacketConn
	// Handler to invoke, dns.DefaultServeMux if nil.
	Handler Handler
	// Default buffer size to use to read incoming UDP messages. If not set
	// it defaults to MinMsgSize (512 B).
	UDPSize int
	// The net.Conn.SetReadTimeout value for new connections, defaults to 2 * time.Second.
	ReadTimeout time.Duration
	// The net.Conn.SetWriteTimeout value for new connections, defaults to 2 * time.Second.
	WriteTimeout time.Duration
	// TCP idle timeout for multiple queries, if nil, defaults to 8 * time.Second (RFC 5966).
	IdleTimeout func() time.Duration
	// Secret(s) for Tsig map[<zonename>]<base64 secret>. The zonename must be in canonical form (lowercase, fqdn, see RFC 4034 Section 6.2).
	TsigSecret map[string]string
	// If NotifyStartedFunc is set it is called once the server has started listening.
	NotifyStartedFunc func()
	// DecorateReader is optional, allows customization of the process that reads raw DNS messages.
	DecorateReader DecorateReader
	// DecorateWriter is optional, allows customization of the process that writes raw DNS messages.
	DecorateWriter DecorateWriter
	// Maximum number of TCP queries before we close the socket. Default is maxTCPQueries (unlimited if -1).
	MaxTCPQueries int
	// Whether to set the SO_REUSEPORT socket option, allowing multiple listeners to be bound to a single address.
	// It is only supported on go1.11+ and when using ListenAndServe.
	ReusePort bool
	// AcceptMsgFunc will check the incoming message and will reject it early in the process.
	// By default DefaultMsgAcceptFunc will be used.
	MsgAcceptFunc MsgAcceptFunc

	// Shutdown handling
	lock     sync.RWMutex
	started  bool
	shutdown chan struct{}
	conns    map[net.Conn]struct{}

	// A pool for UDP message buffers.
	udpPool sync.Pool
}

func (srv *Server) isStarted() bool {
	srv.lock.RLock()
	started := srv.started
	srv.lock.RUnlock()
	return started
}

func makeUDPBuffer(size int) func() interface{} {
	return func() interface{} {
		return make([]byte, size)
	}
}

func (srv *Server) init() {
	srv.shutdown = make(chan struct{})
	srv.conns = make(map[net.Conn]struct{})

	if srv.UDPSize == 0 {
		srv.UDPSize = MinMsgSize
	}
	if srv.MsgAcceptFunc == nil {
		srv.MsgAcceptFunc = DefaultMsgAcceptFunc
	}
	if srv.Handler == nil {
		srv.Handler = DefaultServeMux
	}

	srv.udpPool.New = makeUDPBuffer(srv.UDPSize)
}

func unlockOnce(l sync.Locker) func() {
	var once sync.Once
	return func() { once.Do(l.Unlock) }
}

// ListenAndServe starts a nameserver on the configured address in *Server.
func (srv *Server) ListenAndServe() error {
	unlock := unlockOnce(&srv.lock)
	srv.lock.Lock()
	defer unlock()

	if srv.started {
		return &Error{err: "server already started"}
	}

	addr := srv.Addr
	if addr == "" {
		addr = ":domain"
	}

	srv.init()

	switch srv.Net {
	case "tcp", "tcp4", "tcp6":
		l, err := listenTCP(srv.Net, addr, srv.ReusePort)
		if err != nil {
			return err
		}
		srv.Listener = l
		srv.started = true
		unlock()
		return srv.serveTCP(l)
	case "tcp-tls", "tcp4-tls", "tcp6-tls":
		if srv.TLSConfig == nil || (len(srv.TLSConfig.Certificates) == 0 && srv.TLSConfig.GetCertificate == nil) {
			return errors.New("dns: neither Certificates nor GetCertificate set in Config")
		}
		network := strings.TrimSuffix(srv.Net, "-tls")
		l, err := listenTCP(network, addr, srv.ReusePort)
		if err != nil {
			return err
		}
		l = tls.NewListener(l, srv.TLSConfig)
		srv.Listener = l
		srv.started = true
		unlock()
		return srv.serveTCP(l)
	case "udp", "udp4", "udp6":
		l, err := listenUDP(srv.Net, addr, srv.ReusePort)
		if err != nil {
			return err
		}
		u := l.(*net.UDPConn)
		if e := setUDPSocketOptions(u); e != nil {
			return e
		}
		srv.PacketConn = l
		srv.started = true
		unlock()
		return srv.serveUDP(u)
	}
	return &Error{err: "bad network"}
}

// ActivateAndServe starts a nameserver with the PacketConn or Listener
// configured in *Server. Its main use is to start a server from systemd.
func (srv *Server) ActivateAndServe() error {
	unlock := unlockOnce(&srv.lock)
	srv.lock.Lock()
	defer unlock()

	if srv.started {
		return &Error{err: "server already started"}
	}

	srv.init()

	if srv.PacketConn != nil {
		// Check PacketConn interface's type is valid and value
		// is not nil
		if t, ok := srv.PacketConn.(*net.UDPConn); ok && t != nil {
			if e := setUDPSocketOptions(t); e != nil {
				return e
			}
		}
		srv.started = true
		unlock()
		return srv.serveUDP(srv.PacketConn)
	}
	if srv.Listener != nil {
		srv.started = true
		unlock()
		return srv.serveTCP(srv.Listener)
	}
	return &Error{err: "bad listeners"}
}

// Shutdown shuts down a server. After a call to Shutdown, ListenAndServe and
// ActivateAndServe will return.
func (srv *Server) Shutdown() error {
	return srv.ShutdownContext(context.Background())
}

// ShutdownContext shuts down a server. After a call to ShutdownContext,
// ListenAndServe and ActivateAndServe will return.
//
// A context.Context may be passed to limit how long to wait for connections
// to terminate.
func (srv *Server) ShutdownContext(ctx context.Context) error {
	srv.lock.Lock()
	if !srv.started {
		srv.lock.Unlock()
		return &Error{err: "server not started"}
	}

	srv.started = false

	if srv.PacketConn != nil {
		srv.PacketConn.SetReadDeadline(aLongTimeAgo) // Unblock reads
	}

	if srv.Listener != nil {
		srv.Listener.Close()
	}

	for rw := range srv.conns {
		rw.SetReadDeadline(aLongTimeAgo) // Unblock reads
	}

	srv.lock.Unlock()

	if testShutdownNotify != nil {
		testShutdownNotify.Broadcast()
	}

	var ctxErr error
	select {
	case <-srv.shutdown:
	case <-ctx.Done():
		ctxErr = ctx.Err()
	}

	if srv.PacketConn != nil {
		srv.PacketConn.Close()
	}

	return ctxErr
}

var testShutdownNotify *sync.Cond

// getReadTimeout is a helper func to use system timeout if server did not intend to change it.
func (srv *Server) getReadTimeout() time.Duration {
	if srv.ReadTimeout != 0 {
		return srv.ReadTimeout
	}
	return dnsTimeout
}

// serveTCP starts a TCP listener for the server.
func (srv *Server) serveTCP(l net.Listener) error {
	defer l.Close()

	if srv.NotifyStartedFunc != nil {
		srv.NotifyStartedFunc()
	}

	var wg sync.WaitGroup
	defer func() {
		wg.Wait()
		close(srv.shutdown)
	}()

	for srv.isStarted() {
		rw, err := l.Accept()
		if err != nil {
			if !srv.isStarted() {
				return nil
			}
			if neterr, ok := err.(net.Error); ok && neterr.Temporary() {
				continue
			}
			return err
		}
		srv.lock.Lock()
		// Track the connection to allow unblocking reads on shutdown.
		srv.conns[rw] = struct{}{}
		srv.lock.Unlock()
		wg.Add(1)
		go srv.serveTCPConn(&wg, rw)
	}

	return nil
}

// serveUDP starts a UDP listener for the server.
func (srv *Server) serveUDP(l net.PacketConn) error {
	defer l.Close()

	reader := Reader(defaultReader{srv})
	if srv.DecorateReader != nil {
		reader = srv.DecorateReader(reader)
	}

	lUDP, isUDP := l.(*net.UDPConn)
	readerPC, canPacketConn := reader.(PacketConnReader)
	if !isUDP && !canPacketConn {
		return &Error{err: "PacketConnReader was not implemented on Reader returned from DecorateReader but is required for net.PacketConn"}
	}

	if srv.NotifyStartedFunc != nil {
		srv.NotifyStartedFunc()
	}

	var wg sync.WaitGroup
	defer func() {
		wg.Wait()
		close(srv.shutdown)
	}()

	rtimeout := srv.getReadTimeout()
	// deadline is not used here
	for srv.isStarted() {
		var (
			m    []byte
			sPC  net.Addr
			sUDP *SessionUDP
			err  error
		)
		if isUDP {
			m, sUDP, err = reader.ReadUDP(lUDP, rtimeout)
		} else {
			m, sPC, err = readerPC.ReadPacketConn(l, rtimeout)
		}
		if err != nil {
			if !srv.isStarted() {
				return nil
			}
			if netErr, ok := err.(net.Error); ok && netErr.Temporary() {
				continue
			}
			return err
		}
		if len(m) < headerSize {
			if cap(m) == srv.UDPSize {
				srv.udpPool.Put(m[:srv.UDPSize])
			}
			continue
		}
		wg.Add(1)
		go srv.serveUDPPacket(&wg, m, l, sUDP, sPC)
	}

	return nil
}

// Serve a new TCP connection.
func (srv *Server) serveTCPConn(wg *sync.WaitGroup, rw net.Conn) {
	w := &response{tsigSecret: srv.TsigSecret, tcp: rw}
	if srv.DecorateWriter != nil {
		w.writer = srv.DecorateWriter(w)
	} else {
		w.writer = w
	}

	reader := Reader(defaultReader{srv})
	if srv.DecorateReader != nil {
		reader = srv.DecorateReader(reader)
	}

	idleTimeout := tcpIdleTimeout
	if srv.IdleTimeout != nil {
		idleTimeout = srv.IdleTimeout()
	}

	timeout := srv.getReadTimeout()

	limit := srv.MaxTCPQueries
	if limit == 0 {
		limit = maxTCPQueries
	}

	for q := 0; (q < limit || limit == -1) && srv.isStarted(); q++ {
		m, err := reader.ReadTCP(w.tcp, timeout)
		if err != nil {
			// TODO(tmthrgd): handle error
			break
		}
		srv.serveDNS(m, w)
		if w.closed {
			break // Close() was called
		}
		if w.hijacked {
			break // client will call Close() themselves
		}
		// The first read uses the read timeout, the rest use the
		// idle timeout.
		timeout = idleTimeout
	}

	if !w.hijacked {
		w.Close()
	}

	srv.lock.Lock()
	delete(srv.conns, w.tcp)
	srv.lock.Unlock()

	wg.Done()
}

// Serve a new UDP request.
func (srv *Server) serveUDPPacket(wg *sync.WaitGroup, m []byte, u net.PacketConn, udpSession *SessionUDP, pcSession net.Addr) {
	w := &response{tsigSecret: srv.TsigSecret, udp: u, udpSession: udpSession, pcSession: pcSession}
	if srv.DecorateWriter != nil {
		w.writer = srv.DecorateWriter(w)
	} else {
		w.writer = w
	}

	srv.serveDNS(m, w)
	wg.Done()
}

func (srv *Server) serveDNS(m []byte, w *response) {
	dh, off, err := unpackMsgHdr(m, 0)
	if err != nil {
		// Let client hang, they are sending crap; any reply can be used to amplify.
		return
	}

	req := new(Msg)
	req.setHdr(dh)

	switch action := srv.MsgAcceptFunc(dh); action {
	case MsgAccept:
		if req.unpack(dh, m, off) == nil {
			break
		}

		fallthrough
	case MsgReject, MsgRejectNotImplemented:
		opcode := req.Opcode
		req.SetRcodeFormatError(req)
		req.Zero = false
		if action == MsgRejectNotImplemented {
			req.Opcode = opcode
			req.Rcode = RcodeNotImplemented
		}

		// Are we allowed to delete any OPT records here?
		req.Ns, req.Answer, req.Extra = nil, nil, nil

		w.WriteMsg(req)
		fallthrough
	case MsgIgnore:
		if w.udp != nil && cap(m) == srv.UDPSize {
			srv.udpPool.Put(m[:srv.UDPSize])
		}

		return
	}

	w.tsigStatus = nil
	if w.tsigSecret != nil {
		if t := req.IsTsig(); t != nil {
			if secret, ok := w.tsigSecret[t.Hdr.Name]; ok {
				w.tsigStatus = TsigVerify(m, secret, "", false)
			} else {
				w.tsigStatus = ErrSecret
			}
			w.tsigTimersOnly = false
			w.tsigRequestMAC = req.Extra[len(req.Extra)-1].(*TSIG).MAC
		}
	}

	if w.udp != nil && cap(m) == srv.UDPSize {
		srv.udpPool.Put(m[:srv.UDPSize])
	}

	srv.Handler.ServeDNS(w, req) // Writes back to the client
}

func (srv *Server) readTCP(conn net.Conn, timeout time.Duration) ([]byte, error) {
	// If we race with ShutdownContext, the read deadline may
	// have been set in the distant past to unblock the read
	// below. We must not override it, otherwise we may block
	// ShutdownContext.
	srv.lock.RLock()
	if srv.started {
		conn.SetReadDeadline(time.Now().Add(timeout))
	}
	srv.lock.RUnlock()

	var length uint16
	if err := binary.Read(conn, binary.BigEndian, &length); err != nil {
		return nil, err
	}

	m := make([]byte, length)
	if _, err := io.ReadFull(conn, m); err != nil {
		return nil, err
	}

	return m, nil
}

func (srv *Server) readUDP(conn *net.UDPConn, timeout time.Duration) ([]byte, *SessionUDP, error) {
	srv.lock.RLock()
	if srv.started {
		// See the comment in readTCP above.
		conn.SetReadDeadline(time.Now().Add(timeout))
	}
	srv.lock.RUnlock()

	m := srv.udpPool.Get().([]byte)
	n, s, err := ReadFromSessionUDP(conn, m)
	if err != nil {
		srv.udpPool.Put(m)
		return nil, nil, err
	}
	m = m[:n]
	return m, s, nil
}

func (srv *Server) readPacketConn(conn net.PacketConn, timeout time.Duration) ([]byte, net.Addr, error) {
	srv.lock.RLock()
	if srv.started {
		// See the comment in readTCP above.
		conn.SetReadDeadline(time.Now().Add(timeout))
	}
	srv.lock.RUnlock()

	m := srv.udpPool.Get().([]byte)
	n, addr, err := conn.ReadFrom(m)
	if err != nil {
		srv.udpPool.Put(m)
		return nil, nil, err
	}
	m = m[:n]
	return m, addr, nil
}

// WriteMsg implements the ResponseWriter.WriteMsg method.
func (w *response) WriteMsg(m *Msg) (err error) {
	if w.closed {
		return &Error{err: "WriteMsg called after Close"}
	}

	var data []byte
	if w.tsigSecret != nil { // if no secrets, dont check for the tsig (which is a longer check)
		if t := m.IsTsig(); t != nil {
			data, w.tsigRequestMAC, err = TsigGenerate(m, w.tsigSecret[t.Hdr.Name], w.tsigRequestMAC, w.tsigTimersOnly)
			if err != nil {
				return err
			}
			_, err = w.writer.Write(data)
			return err
		}
	}
	data, err = m.Pack()
	if err != nil {
		return err
	}
	_, err = w.writer.Write(data)
	return err
}

// Write implements the ResponseWriter.Write method.
func (w *response) Write(m []byte) (int, error) {
	if w.closed {
		return 0, &Error{err: "Write called after Close"}
	}

	switch {
	case w.udp != nil:
		if u, ok := w.udp.(*net.UDPConn); ok {
			return WriteToSessionUDP(u, m, w.udpSession)
		}
		return w.udp.WriteTo(m, w.pcSession)
	case w.tcp != nil:
		if len(m) > MaxMsgSize {
			return 0, &Error{err: "message too large"}
		}

		l := make([]byte, 2)
		binary.BigEndian.PutUint16(l, uint16(len(m)))

		n, err := (&net.Buffers{l, m}).WriteTo(w.tcp)
		return int(n), err
	default:
		panic("dns: internal error: udp and tcp both nil")
	}
}

// LocalAddr implements the ResponseWriter.LocalAddr method.
func (w *response) LocalAddr() net.Addr {
	switch {
	case w.udp != nil:
		return w.udp.LocalAddr()
	case w.tcp != nil:
		return w.tcp.LocalAddr()
	default:
		panic("dns: internal error: udp and tcp both nil")
	}
}

// RemoteAddr implements the ResponseWriter.RemoteAddr method.
func (w *response) RemoteAddr() net.Addr {
	switch {
	case w.udpSession != nil:
		return w.udpSession.RemoteAddr()
	case w.pcSession != nil:
		return w.pcSession
	case w.tcp != nil:
		return w.tcp.RemoteAddr()
	default:
		panic("dns: internal error: udpSession, pcSession and tcp are all nil")
	}
}

// TsigStatus implements the ResponseWriter.TsigStatus method.
func (w *response) TsigStatus() error { return w.tsigStatus }

// TsigTimersOnly implements the ResponseWriter.TsigTimersOnly method.
func (w *response) TsigTimersOnly(b bool) { w.tsigTimersOnly = b }

// Hijack implements the ResponseWriter.Hijack method.
func (w *response) Hijack() { w.hijacked = true }

// Close implements the ResponseWriter.Close method
func (w *response) Close() error {
	if w.closed {
		return &Error{err: "connection already closed"}
	}
	w.closed = true

	switch {
	case w.udp != nil:
		// Can't close the udp conn, as that is actually the listener.
		return nil
	case w.tcp != nil:
		return w.tcp.Close()
	default:
		panic("dns: internal error: udp and tcp both nil")
	}
}

// ConnectionState() implements the ConnectionStater.ConnectionState() interface.
func (w *response) ConnectionState() *tls.ConnectionState {
	type tlsConnectionStater interface {
		ConnectionState() tls.ConnectionState
	}
	if v, ok := w.tcp.(tlsConnectionStater); ok {
		t := v.ConnectionState()
		return &t
	}
	return nil
}
