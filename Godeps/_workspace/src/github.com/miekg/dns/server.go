// DNS server implementation.

package dns

import (
	"bytes"
	"io"
	"net"
	"sync"
	"time"
)

// Handler is implemented by any value that implements ServeDNS.
type Handler interface {
	ServeDNS(w ResponseWriter, r *Msg)
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

type response struct {
	hijacked       bool // connection has been hijacked by handler
	tsigStatus     error
	tsigTimersOnly bool
	tsigRequestMAC string
	tsigSecret     map[string]string // the tsig secrets
	udp            *net.UDPConn      // i/o connection if UDP was used
	tcp            *net.TCPConn      // i/o connection if TCP was used
	udpSession     *SessionUDP       // oob data to get egress interface right
	remoteAddr     net.Addr          // address of the client
}

// ServeMux is an DNS request multiplexer. It matches the
// zone name of each incoming request against a list of
// registered patterns add calls the handler for the pattern
// that most closely matches the zone name. ServeMux is DNSSEC aware, meaning
// that queries for the DS record are redirected to the parent zone (if that
// is also registered), otherwise the child gets the query.
// ServeMux is also safe for concurrent access from multiple goroutines.
type ServeMux struct {
	z map[string]Handler
	m *sync.RWMutex
}

// NewServeMux allocates and returns a new ServeMux.
func NewServeMux() *ServeMux { return &ServeMux{z: make(map[string]Handler), m: new(sync.RWMutex)} }

// DefaultServeMux is the default ServeMux used by Serve.
var DefaultServeMux = NewServeMux()

// The HandlerFunc type is an adapter to allow the use of
// ordinary functions as DNS handlers.  If f is a function
// with the appropriate signature, HandlerFunc(f) is a
// Handler object that calls f.
type HandlerFunc func(ResponseWriter, *Msg)

// ServeDNS calls f(w, r).
func (f HandlerFunc) ServeDNS(w ResponseWriter, r *Msg) {
	f(w, r)
}

// HandleFailed returns a HandlerFunc that returns SERVFAIL for every request it gets.
func HandleFailed(w ResponseWriter, r *Msg) {
	m := new(Msg)
	m.SetRcode(r, RcodeServerFailure)
	// does not matter if this write fails
	w.WriteMsg(m)
}

func failedHandler() Handler { return HandlerFunc(HandleFailed) }

// ListenAndServe Starts a server on addresss and network speficied. Invoke handler
// for incoming queries.
func ListenAndServe(addr string, network string, handler Handler) error {
	server := &Server{Addr: addr, Net: network, Handler: handler}
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

func (mux *ServeMux) match(q string, t uint16) Handler {
	mux.m.RLock()
	defer mux.m.RUnlock()
	var handler Handler
	b := make([]byte, len(q)) // worst case, one label of length q
	off := 0
	end := false
	for {
		l := len(q[off:])
		for i := 0; i < l; i++ {
			b[i] = q[off+i]
			if b[i] >= 'A' && b[i] <= 'Z' {
				b[i] |= ('a' - 'A')
			}
		}
		if h, ok := mux.z[string(b[:l])]; ok { // 'causes garbage, might want to change the map key
			if t != TypeDS {
				return h
			}
			// Continue for DS to see if we have a parent too, if so delegeate to the parent
			handler = h
		}
		off, end = NextLabel(q, off)
		if end {
			break
		}
	}
	// Wildcard match, if we have found nothing try the root zone as a last resort.
	if h, ok := mux.z["."]; ok {
		return h
	}
	return handler
}

// Handle adds a handler to the ServeMux for pattern.
func (mux *ServeMux) Handle(pattern string, handler Handler) {
	if pattern == "" {
		panic("dns: invalid pattern " + pattern)
	}
	mux.m.Lock()
	mux.z[Fqdn(pattern)] = handler
	mux.m.Unlock()
}

// HandleFunc adds a handler function to the ServeMux for pattern.
func (mux *ServeMux) HandleFunc(pattern string, handler func(ResponseWriter, *Msg)) {
	mux.Handle(pattern, HandlerFunc(handler))
}

// HandleRemove deregistrars the handler specific for pattern from the ServeMux.
func (mux *ServeMux) HandleRemove(pattern string) {
	if pattern == "" {
		panic("dns: invalid pattern " + pattern)
	}
	// don't need a mutex here, because deleting is OK, even if the
	// entry is note there.
	delete(mux.z, Fqdn(pattern))
}

// ServeDNS dispatches the request to the handler whose
// pattern most closely matches the request message. If DefaultServeMux
// is used the correct thing for DS queries is done: a possible parent
// is sought.
// If no handler is found a standard SERVFAIL message is returned
// If the request message does not have exactly one question in the
// question section a SERVFAIL is returned, unlesss Unsafe is true.
func (mux *ServeMux) ServeDNS(w ResponseWriter, request *Msg) {
	var h Handler
	if len(request.Question) < 1 { // allow more than one question
		h = failedHandler()
	} else {
		if h = mux.match(request.Question[0].Name, request.Question[0].Qtype); h == nil {
			h = failedHandler()
		}
	}
	h.ServeDNS(w, request)
}

// Handle registers the handler with the given pattern
// in the DefaultServeMux. The documentation for
// ServeMux explains how patterns are matched.
func Handle(pattern string, handler Handler) { DefaultServeMux.Handle(pattern, handler) }

// HandleRemove deregisters the handle with the given pattern
// in the DefaultServeMux.
func HandleRemove(pattern string) { DefaultServeMux.HandleRemove(pattern) }

// HandleFunc registers the handler function with the given pattern
// in the DefaultServeMux.
func HandleFunc(pattern string, handler func(ResponseWriter, *Msg)) {
	DefaultServeMux.HandleFunc(pattern, handler)
}

// A Server defines parameters for running an DNS server.
type Server struct {
	// Address to listen on, ":dns" if empty.
	Addr string
	// if "tcp" it will invoke a TCP listener, otherwise an UDP one.
	Net string
	// TCP Listener to use, this is to aid in systemd's socket activation.
	Listener net.Listener
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
	// Secret(s) for Tsig map[<zonename>]<base64 secret>.
	TsigSecret map[string]string
	// Unsafe instructs the server to disregard any sanity checks and directly hand the message to
	// the handler. It will specfically not check if the query has the QR bit not set.
	Unsafe bool
	// If NotifyStartedFunc is set is is called, once the server has started listening.
	NotifyStartedFunc func()

	// For graceful shutdown.
	stopUDP chan bool
	stopTCP chan bool
	wgUDP   sync.WaitGroup
	wgTCP   sync.WaitGroup

	// make start/shutdown not racy
	lock    sync.Mutex
	started bool
}

// ListenAndServe starts a nameserver on the configured address in *Server.
func (srv *Server) ListenAndServe() error {
	srv.lock.Lock()
	if srv.started {
		srv.lock.Unlock()
		return &Error{err: "server already started"}
	}
	srv.stopUDP, srv.stopTCP = make(chan bool), make(chan bool)
	srv.started = true
	srv.lock.Unlock()
	addr := srv.Addr
	if addr == "" {
		addr = ":domain"
	}
	if srv.UDPSize == 0 {
		srv.UDPSize = MinMsgSize
	}
	switch srv.Net {
	case "tcp", "tcp4", "tcp6":
		a, e := net.ResolveTCPAddr(srv.Net, addr)
		if e != nil {
			return e
		}
		l, e := net.ListenTCP(srv.Net, a)
		if e != nil {
			return e
		}
		srv.Listener = l
		return srv.serveTCP(l)
	case "udp", "udp4", "udp6":
		a, e := net.ResolveUDPAddr(srv.Net, addr)
		if e != nil {
			return e
		}
		l, e := net.ListenUDP(srv.Net, a)
		if e != nil {
			return e
		}
		if e := setUDPSocketOptions(l); e != nil {
			return e
		}
		srv.PacketConn = l
		return srv.serveUDP(l)
	}
	return &Error{err: "bad network"}
}

// ActivateAndServe starts a nameserver with the PacketConn or Listener
// configured in *Server. Its main use is to start a server from systemd.
func (srv *Server) ActivateAndServe() error {
	srv.lock.Lock()
	if srv.started {
		srv.lock.Unlock()
		return &Error{err: "server already started"}
	}
	srv.stopUDP, srv.stopTCP = make(chan bool), make(chan bool)
	srv.started = true
	srv.lock.Unlock()
	if srv.PacketConn != nil {
		if srv.UDPSize == 0 {
			srv.UDPSize = MinMsgSize
		}
		if t, ok := srv.PacketConn.(*net.UDPConn); ok {
			if e := setUDPSocketOptions(t); e != nil {
				return e
			}
			return srv.serveUDP(t)
		}
	}
	if srv.Listener != nil {
		if t, ok := srv.Listener.(*net.TCPListener); ok {
			return srv.serveTCP(t)
		}
	}
	return &Error{err: "bad listeners"}
}

// Shutdown gracefully shuts down a server. After a call to Shutdown, ListenAndServe and
// ActivateAndServe will return. All in progress queries are completed before the server
// is taken down. If the Shutdown is taking longer than the reading timeout and error
// is returned.
func (srv *Server) Shutdown() error {
	srv.lock.Lock()
	if !srv.started {
		srv.lock.Unlock()
		return &Error{err: "server not started"}
	}
	srv.started = false
	srv.lock.Unlock()
	net, addr := srv.Net, srv.Addr
	switch {
	case srv.Listener != nil:
		a := srv.Listener.Addr()
		net, addr = a.Network(), a.String()
	case srv.PacketConn != nil:
		a := srv.PacketConn.LocalAddr()
		net, addr = a.Network(), a.String()
	}

	fin := make(chan bool)
	switch net {
	case "tcp", "tcp4", "tcp6":
		go func() {
			srv.stopTCP <- true
			srv.wgTCP.Wait()
			fin <- true
		}()

	case "udp", "udp4", "udp6":
		go func() {
			srv.stopUDP <- true
			srv.wgUDP.Wait()
			fin <- true
		}()
	}

	c := &Client{Net: net}
	go c.Exchange(new(Msg), addr) // extra query to help ReadXXX loop to pass

	select {
	case <-time.After(srv.getReadTimeout()):
		return &Error{err: "server shutdown is pending"}
	case <-fin:
		return nil
	}
}

// getReadTimeout is a helper func to use system timeout if server did not intend to change it.
func (srv *Server) getReadTimeout() time.Duration {
	rtimeout := dnsTimeout
	if srv.ReadTimeout != 0 {
		rtimeout = srv.ReadTimeout
	}
	return rtimeout
}

// serveTCP starts a TCP listener for the server.
// Each request is handled in a separate goroutine.
func (srv *Server) serveTCP(l *net.TCPListener) error {
	defer l.Close()

	if srv.NotifyStartedFunc != nil {
		srv.NotifyStartedFunc()
	}

	handler := srv.Handler
	if handler == nil {
		handler = DefaultServeMux
	}
	rtimeout := srv.getReadTimeout()
	// deadline is not used here
	for {
		rw, e := l.AcceptTCP()
		if e != nil {
			continue
		}
		m, e := srv.readTCP(rw, rtimeout)
		select {
		case <-srv.stopTCP:
			return nil
		default:
		}
		if e != nil {
			continue
		}
		srv.wgTCP.Add(1)
		go srv.serve(rw.RemoteAddr(), handler, m, nil, nil, rw)
	}
	panic("dns: not reached")
}

// serveUDP starts a UDP listener for the server.
// Each request is handled in a separate goroutine.
func (srv *Server) serveUDP(l *net.UDPConn) error {
	defer l.Close()

	if srv.NotifyStartedFunc != nil {
		srv.NotifyStartedFunc()
	}

	handler := srv.Handler
	if handler == nil {
		handler = DefaultServeMux
	}
	rtimeout := srv.getReadTimeout()
	// deadline is not used here
	for {
		m, s, e := srv.readUDP(l, rtimeout)
		select {
		case <-srv.stopUDP:
			return nil
		default:
		}
		if e != nil {
			continue
		}
		srv.wgUDP.Add(1)
		go srv.serve(s.RemoteAddr(), handler, m, l, s, nil)
	}
	panic("dns: not reached")
}

// Serve a new connection.
func (srv *Server) serve(a net.Addr, h Handler, m []byte, u *net.UDPConn, s *SessionUDP, t *net.TCPConn) {
	w := &response{tsigSecret: srv.TsigSecret, udp: u, tcp: t, remoteAddr: a, udpSession: s}
	q := 0
	defer func() {
		if u != nil {
			srv.wgUDP.Done()
		}
		if t != nil {
			srv.wgTCP.Done()
		}
	}()
Redo:
	req := new(Msg)
	err := req.Unpack(m)
	if err != nil { // Send a FormatError back
		x := new(Msg)
		x.SetRcodeFormatError(req)
		w.WriteMsg(x)
		goto Exit
	}
	if !srv.Unsafe && req.Response {
		goto Exit
	}

	w.tsigStatus = nil
	if w.tsigSecret != nil {
		if t := req.IsTsig(); t != nil {
			secret := t.Hdr.Name
			if _, ok := w.tsigSecret[secret]; !ok {
				w.tsigStatus = ErrKeyAlg
			}
			w.tsigStatus = TsigVerify(m, w.tsigSecret[secret], "", false)
			w.tsigTimersOnly = false
			w.tsigRequestMAC = req.Extra[len(req.Extra)-1].(*TSIG).MAC
		}
	}
	h.ServeDNS(w, req) // Writes back to the client

Exit:
	if w.hijacked {
		return // client calls Close()
	}
	if u != nil { // UDP, "close" and return
		w.Close()
		return
	}
	idleTimeout := tcpIdleTimeout
	if srv.IdleTimeout != nil {
		idleTimeout = srv.IdleTimeout()
	}
	m, e := srv.readTCP(w.tcp, idleTimeout)
	if e == nil {
		q++
		// TODO(miek): make this number configurable?
		if q > 128 { // close socket after this many queries
			w.Close()
			return
		}
		goto Redo
	}
	w.Close()
	return
}

func (srv *Server) readTCP(conn *net.TCPConn, timeout time.Duration) ([]byte, error) {
	conn.SetReadDeadline(time.Now().Add(timeout))
	l := make([]byte, 2)
	n, err := conn.Read(l)
	if err != nil || n != 2 {
		if err != nil {
			return nil, err
		}
		return nil, ErrShortRead
	}
	length, _ := unpackUint16(l, 0)
	if length == 0 {
		return nil, ErrShortRead
	}
	m := make([]byte, int(length))
	n, err = conn.Read(m[:int(length)])
	if err != nil || n == 0 {
		if err != nil {
			return nil, err
		}
		return nil, ErrShortRead
	}
	i := n
	for i < int(length) {
		j, err := conn.Read(m[i:int(length)])
		if err != nil {
			return nil, err
		}
		i += j
	}
	n = i
	m = m[:n]
	return m, nil
}

func (srv *Server) readUDP(conn *net.UDPConn, timeout time.Duration) ([]byte, *SessionUDP, error) {
	conn.SetReadDeadline(time.Now().Add(timeout))
	m := make([]byte, srv.UDPSize)
	n, s, e := ReadFromSessionUDP(conn, m)
	if e != nil || n == 0 {
		if e != nil {
			return nil, nil, e
		}
		return nil, nil, ErrShortRead
	}
	m = m[:n]
	return m, s, nil
}

// WriteMsg implements the ResponseWriter.WriteMsg method.
func (w *response) WriteMsg(m *Msg) (err error) {
	var data []byte
	if w.tsigSecret != nil { // if no secrets, dont check for the tsig (which is a longer check)
		if t := m.IsTsig(); t != nil {
			data, w.tsigRequestMAC, err = TsigGenerate(m, w.tsigSecret[t.Hdr.Name], w.tsigRequestMAC, w.tsigTimersOnly)
			if err != nil {
				return err
			}
			_, err = w.Write(data)
			return err
		}
	}
	data, err = m.Pack()
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

// Write implements the ResponseWriter.Write method.
func (w *response) Write(m []byte) (int, error) {
	switch {
	case w.udp != nil:
		n, err := WriteToSessionUDP(w.udp, m, w.udpSession)
		return n, err
	case w.tcp != nil:
		lm := len(m)
		if lm < 2 {
			return 0, io.ErrShortBuffer
		}
		if lm > MaxMsgSize {
			return 0, &Error{err: "message too large"}
		}
		l := make([]byte, 2, 2+lm)
		l[0], l[1] = packUint16(uint16(lm))
		m = append(l, m...)

		n, err := io.Copy(w.tcp, bytes.NewReader(m))
		return int(n), err
	}
	panic("not reached")
}

// LocalAddr implements the ResponseWriter.LocalAddr method.
func (w *response) LocalAddr() net.Addr {
	if w.tcp != nil {
		return w.tcp.LocalAddr()
	}
	return w.udp.LocalAddr()
}

// RemoteAddr implements the ResponseWriter.RemoteAddr method.
func (w *response) RemoteAddr() net.Addr { return w.remoteAddr }

// TsigStatus implements the ResponseWriter.TsigStatus method.
func (w *response) TsigStatus() error { return w.tsigStatus }

// TsigTimersOnly implements the ResponseWriter.TsigTimersOnly method.
func (w *response) TsigTimersOnly(b bool) { w.tsigTimersOnly = b }

// Hijack implements the ResponseWriter.Hijack method.
func (w *response) Hijack() { w.hijacked = true }

// Close implements the ResponseWriter.Close method
func (w *response) Close() error {
	// Can't close the udp conn, as that is actually the listener.
	if w.tcp != nil {
		e := w.tcp.Close()
		w.tcp = nil
		return e
	}
	return nil
}
