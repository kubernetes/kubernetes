// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Transport code.

package http2

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/rand"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	mathrand "math/rand"
	"net"
	"net/http"
	"net/http/httptrace"
	"net/textproto"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/net/http/httpguts"
	"golang.org/x/net/http2/hpack"
	"golang.org/x/net/idna"
)

const (
	// transportDefaultConnFlow is how many connection-level flow control
	// tokens we give the server at start-up, past the default 64k.
	transportDefaultConnFlow = 1 << 30

	// transportDefaultStreamFlow is how many stream-level flow
	// control tokens we announce to the peer, and how many bytes
	// we buffer per stream.
	transportDefaultStreamFlow = 4 << 20

	// transportDefaultStreamMinRefresh is the minimum number of bytes we'll send
	// a stream-level WINDOW_UPDATE for at a time.
	transportDefaultStreamMinRefresh = 4 << 10

	defaultUserAgent = "Go-http-client/2.0"
)

// Transport is an HTTP/2 Transport.
//
// A Transport internally caches connections to servers. It is safe
// for concurrent use by multiple goroutines.
type Transport struct {
	// DialTLS specifies an optional dial function for creating
	// TLS connections for requests.
	//
	// If DialTLS is nil, tls.Dial is used.
	//
	// If the returned net.Conn has a ConnectionState method like tls.Conn,
	// it will be used to set http.Response.TLS.
	DialTLS func(network, addr string, cfg *tls.Config) (net.Conn, error)

	// TLSClientConfig specifies the TLS configuration to use with
	// tls.Client. If nil, the default configuration is used.
	TLSClientConfig *tls.Config

	// ConnPool optionally specifies an alternate connection pool to use.
	// If nil, the default is used.
	ConnPool ClientConnPool

	// DisableCompression, if true, prevents the Transport from
	// requesting compression with an "Accept-Encoding: gzip"
	// request header when the Request contains no existing
	// Accept-Encoding value. If the Transport requests gzip on
	// its own and gets a gzipped response, it's transparently
	// decoded in the Response.Body. However, if the user
	// explicitly requested gzip it is not automatically
	// uncompressed.
	DisableCompression bool

	// AllowHTTP, if true, permits HTTP/2 requests using the insecure,
	// plain-text "http" scheme. Note that this does not enable h2c support.
	AllowHTTP bool

	// MaxHeaderListSize is the http2 SETTINGS_MAX_HEADER_LIST_SIZE to
	// send in the initial settings frame. It is how many bytes
	// of response headers are allowed. Unlike the http2 spec, zero here
	// means to use a default limit (currently 10MB). If you actually
	// want to advertise an unlimited value to the peer, Transport
	// interprets the highest possible value here (0xffffffff or 1<<32-1)
	// to mean no limit.
	MaxHeaderListSize uint32

	// StrictMaxConcurrentStreams controls whether the server's
	// SETTINGS_MAX_CONCURRENT_STREAMS should be respected
	// globally. If false, new TCP connections are created to the
	// server as needed to keep each under the per-connection
	// SETTINGS_MAX_CONCURRENT_STREAMS limit. If true, the
	// server's SETTINGS_MAX_CONCURRENT_STREAMS is interpreted as
	// a global limit and callers of RoundTrip block when needed,
	// waiting for their turn.
	StrictMaxConcurrentStreams bool

	// ReadIdleTimeout is the timeout after which a health check using ping
	// frame will be carried out if no frame is received on the connection.
	// Note that a ping response will is considered a received frame, so if
	// there is no other traffic on the connection, the health check will
	// be performed every ReadIdleTimeout interval.
	// If zero, no health check is performed.
	ReadIdleTimeout time.Duration

	// PingTimeout is the timeout after which the connection will be closed
	// if a response to Ping is not received.
	// Defaults to 15s.
	PingTimeout time.Duration

	// t1, if non-nil, is the standard library Transport using
	// this transport. Its settings are used (but not its
	// RoundTrip method, etc).
	t1 *http.Transport

	connPoolOnce  sync.Once
	connPoolOrDef ClientConnPool // non-nil version of ConnPool
}

func (t *Transport) maxHeaderListSize() uint32 {
	if t.MaxHeaderListSize == 0 {
		return 10 << 20
	}
	if t.MaxHeaderListSize == 0xffffffff {
		return 0
	}
	return t.MaxHeaderListSize
}

func (t *Transport) disableCompression() bool {
	return t.DisableCompression || (t.t1 != nil && t.t1.DisableCompression)
}

func (t *Transport) pingTimeout() time.Duration {
	if t.PingTimeout == 0 {
		return 15 * time.Second
	}
	return t.PingTimeout

}

// ConfigureTransport configures a net/http HTTP/1 Transport to use HTTP/2.
// It returns an error if t1 has already been HTTP/2-enabled.
//
// Use ConfigureTransports instead to configure the HTTP/2 Transport.
func ConfigureTransport(t1 *http.Transport) error {
	_, err := ConfigureTransports(t1)
	return err
}

// ConfigureTransports configures a net/http HTTP/1 Transport to use HTTP/2.
// It returns a new HTTP/2 Transport for further configuration.
// It returns an error if t1 has already been HTTP/2-enabled.
func ConfigureTransports(t1 *http.Transport) (*Transport, error) {
	return configureTransports(t1)
}

func configureTransports(t1 *http.Transport) (*Transport, error) {
	connPool := new(clientConnPool)
	t2 := &Transport{
		ConnPool: noDialClientConnPool{connPool},
		t1:       t1,
	}
	connPool.t = t2
	if err := registerHTTPSProtocol(t1, noDialH2RoundTripper{t2}); err != nil {
		return nil, err
	}
	if t1.TLSClientConfig == nil {
		t1.TLSClientConfig = new(tls.Config)
	}
	if !strSliceContains(t1.TLSClientConfig.NextProtos, "h2") {
		t1.TLSClientConfig.NextProtos = append([]string{"h2"}, t1.TLSClientConfig.NextProtos...)
	}
	if !strSliceContains(t1.TLSClientConfig.NextProtos, "http/1.1") {
		t1.TLSClientConfig.NextProtos = append(t1.TLSClientConfig.NextProtos, "http/1.1")
	}
	upgradeFn := func(authority string, c *tls.Conn) http.RoundTripper {
		addr := authorityAddr("https", authority)
		if used, err := connPool.addConnIfNeeded(addr, t2, c); err != nil {
			go c.Close()
			return erringRoundTripper{err}
		} else if !used {
			// Turns out we don't need this c.
			// For example, two goroutines made requests to the same host
			// at the same time, both kicking off TCP dials. (since protocol
			// was unknown)
			go c.Close()
		}
		return t2
	}
	if m := t1.TLSNextProto; len(m) == 0 {
		t1.TLSNextProto = map[string]func(string, *tls.Conn) http.RoundTripper{
			"h2": upgradeFn,
		}
	} else {
		m["h2"] = upgradeFn
	}
	return t2, nil
}

func (t *Transport) connPool() ClientConnPool {
	t.connPoolOnce.Do(t.initConnPool)
	return t.connPoolOrDef
}

func (t *Transport) initConnPool() {
	if t.ConnPool != nil {
		t.connPoolOrDef = t.ConnPool
	} else {
		t.connPoolOrDef = &clientConnPool{t: t}
	}
}

// ClientConn is the state of a single HTTP/2 client connection to an
// HTTP/2 server.
type ClientConn struct {
	t         *Transport
	tconn     net.Conn             // usually *tls.Conn, except specialized impls
	tlsState  *tls.ConnectionState // nil only for specialized impls
	reused    uint32               // whether conn is being reused; atomic
	singleUse bool                 // whether being used for a single http.Request

	// readLoop goroutine fields:
	readerDone chan struct{} // closed on error
	readerErr  error         // set before readerDone is closed

	idleTimeout time.Duration // or 0 for never
	idleTimer   *time.Timer

	mu              sync.Mutex // guards following
	cond            *sync.Cond // hold mu; broadcast on flow/closed changes
	flow            flow       // our conn-level flow control quota (cs.flow is per stream)
	inflow          flow       // peer's conn-level flow control
	closing         bool
	closed          bool
	wantSettingsAck bool                     // we sent a SETTINGS frame and haven't heard back
	goAway          *GoAwayFrame             // if non-nil, the GoAwayFrame we received
	goAwayDebug     string                   // goAway frame's debug data, retained as a string
	streams         map[uint32]*clientStream // client-initiated
	nextStreamID    uint32
	pendingRequests int                       // requests blocked and waiting to be sent because len(streams) == maxConcurrentStreams
	pings           map[[8]byte]chan struct{} // in flight ping data to notification channel
	bw              *bufio.Writer
	br              *bufio.Reader
	fr              *Framer
	lastActive      time.Time
	lastIdle        time.Time // time last idle
	// Settings from peer: (also guarded by mu)
	maxFrameSize          uint32
	maxConcurrentStreams  uint32
	peerMaxHeaderListSize uint64
	initialWindowSize     uint32

	hbuf    bytes.Buffer // HPACK encoder writes into this
	henc    *hpack.Encoder
	freeBuf [][]byte

	wmu  sync.Mutex // held while writing; acquire AFTER mu if holding both
	werr error      // first write error that has occurred
}

// clientStream is the state for a single HTTP/2 stream. One of these
// is created for each Transport.RoundTrip call.
type clientStream struct {
	cc            *ClientConn
	req           *http.Request
	trace         *httptrace.ClientTrace // or nil
	ID            uint32
	resc          chan resAndError
	bufPipe       pipe // buffered pipe with the flow-controlled response payload
	startedWrite  bool // started request body write; guarded by cc.mu
	requestedGzip bool
	on100         func() // optional code to run if get a 100 continue response

	flow        flow  // guarded by cc.mu
	inflow      flow  // guarded by cc.mu
	bytesRemain int64 // -1 means unknown; owned by transportResponseBody.Read
	readErr     error // sticky read error; owned by transportResponseBody.Read
	stopReqBody error // if non-nil, stop writing req body; guarded by cc.mu
	didReset    bool  // whether we sent a RST_STREAM to the server; guarded by cc.mu

	peerReset chan struct{} // closed on peer reset
	resetErr  error         // populated before peerReset is closed

	done chan struct{} // closed when stream remove from cc.streams map; close calls guarded by cc.mu

	// owned by clientConnReadLoop:
	firstByte    bool  // got the first response byte
	pastHeaders  bool  // got first MetaHeadersFrame (actual headers)
	pastTrailers bool  // got optional second MetaHeadersFrame (trailers)
	num1xx       uint8 // number of 1xx responses seen

	trailer    http.Header  // accumulated trailers
	resTrailer *http.Header // client's Response.Trailer
}

// awaitRequestCancel waits for the user to cancel a request or for the done
// channel to be signaled. A non-nil error is returned only if the request was
// canceled.
func awaitRequestCancel(req *http.Request, done <-chan struct{}) error {
	ctx := req.Context()
	if req.Cancel == nil && ctx.Done() == nil {
		return nil
	}
	select {
	case <-req.Cancel:
		return errRequestCanceled
	case <-ctx.Done():
		return ctx.Err()
	case <-done:
		return nil
	}
}

var got1xxFuncForTests func(int, textproto.MIMEHeader) error

// get1xxTraceFunc returns the value of request's httptrace.ClientTrace.Got1xxResponse func,
// if any. It returns nil if not set or if the Go version is too old.
func (cs *clientStream) get1xxTraceFunc() func(int, textproto.MIMEHeader) error {
	if fn := got1xxFuncForTests; fn != nil {
		return fn
	}
	return traceGot1xxResponseFunc(cs.trace)
}

// awaitRequestCancel waits for the user to cancel a request, its context to
// expire, or for the request to be done (any way it might be removed from the
// cc.streams map: peer reset, successful completion, TCP connection breakage,
// etc). If the request is canceled, then cs will be canceled and closed.
func (cs *clientStream) awaitRequestCancel(req *http.Request) {
	if err := awaitRequestCancel(req, cs.done); err != nil {
		cs.cancelStream()
		cs.bufPipe.CloseWithError(err)
	}
}

func (cs *clientStream) cancelStream() {
	cc := cs.cc
	cc.mu.Lock()
	didReset := cs.didReset
	cs.didReset = true
	cc.mu.Unlock()

	if !didReset {
		cc.writeStreamReset(cs.ID, ErrCodeCancel, nil)
		cc.forgetStreamID(cs.ID)
	}
}

// checkResetOrDone reports any error sent in a RST_STREAM frame by the
// server, or errStreamClosed if the stream is complete.
func (cs *clientStream) checkResetOrDone() error {
	select {
	case <-cs.peerReset:
		return cs.resetErr
	case <-cs.done:
		return errStreamClosed
	default:
		return nil
	}
}

func (cs *clientStream) getStartedWrite() bool {
	cc := cs.cc
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cs.startedWrite
}

func (cs *clientStream) abortRequestBodyWrite(err error) {
	if err == nil {
		panic("nil error")
	}
	cc := cs.cc
	cc.mu.Lock()
	cs.stopReqBody = err
	cc.cond.Broadcast()
	cc.mu.Unlock()
}

type stickyErrWriter struct {
	w   io.Writer
	err *error
}

func (sew stickyErrWriter) Write(p []byte) (n int, err error) {
	if *sew.err != nil {
		return 0, *sew.err
	}
	n, err = sew.w.Write(p)
	*sew.err = err
	return
}

// noCachedConnError is the concrete type of ErrNoCachedConn, which
// needs to be detected by net/http regardless of whether it's its
// bundled version (in h2_bundle.go with a rewritten type name) or
// from a user's x/net/http2. As such, as it has a unique method name
// (IsHTTP2NoCachedConnError) that net/http sniffs for via func
// isNoCachedConnError.
type noCachedConnError struct{}

func (noCachedConnError) IsHTTP2NoCachedConnError() {}
func (noCachedConnError) Error() string             { return "http2: no cached connection was available" }

// isNoCachedConnError reports whether err is of type noCachedConnError
// or its equivalent renamed type in net/http2's h2_bundle.go. Both types
// may coexist in the same running program.
func isNoCachedConnError(err error) bool {
	_, ok := err.(interface{ IsHTTP2NoCachedConnError() })
	return ok
}

var ErrNoCachedConn error = noCachedConnError{}

// RoundTripOpt are options for the Transport.RoundTripOpt method.
type RoundTripOpt struct {
	// OnlyCachedConn controls whether RoundTripOpt may
	// create a new TCP connection. If set true and
	// no cached connection is available, RoundTripOpt
	// will return ErrNoCachedConn.
	OnlyCachedConn bool
}

func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	return t.RoundTripOpt(req, RoundTripOpt{})
}

// authorityAddr returns a given authority (a host/IP, or host:port / ip:port)
// and returns a host:port. The port 443 is added if needed.
func authorityAddr(scheme string, authority string) (addr string) {
	host, port, err := net.SplitHostPort(authority)
	if err != nil { // authority didn't have a port
		port = "443"
		if scheme == "http" {
			port = "80"
		}
		host = authority
	}
	if a, err := idna.ToASCII(host); err == nil {
		host = a
	}
	// IPv6 address literal, without a port:
	if strings.HasPrefix(host, "[") && strings.HasSuffix(host, "]") {
		return host + ":" + port
	}
	return net.JoinHostPort(host, port)
}

// RoundTripOpt is like RoundTrip, but takes options.
func (t *Transport) RoundTripOpt(req *http.Request, opt RoundTripOpt) (*http.Response, error) {
	if !(req.URL.Scheme == "https" || (req.URL.Scheme == "http" && t.AllowHTTP)) {
		return nil, errors.New("http2: unsupported scheme")
	}

	addr := authorityAddr(req.URL.Scheme, req.URL.Host)
	for retry := 0; ; retry++ {
		cc, err := t.connPool().GetClientConn(req, addr)
		if err != nil {
			t.vlogf("http2: Transport failed to get client conn for %s: %v", addr, err)
			return nil, err
		}
		reused := !atomic.CompareAndSwapUint32(&cc.reused, 0, 1)
		traceGotConn(req, cc, reused)
		res, gotErrAfterReqBodyWrite, err := cc.roundTrip(req)
		if err != nil && retry <= 6 {
			if req, err = shouldRetryRequest(req, err, gotErrAfterReqBodyWrite); err == nil {
				// After the first retry, do exponential backoff with 10% jitter.
				if retry == 0 {
					continue
				}
				backoff := float64(uint(1) << (uint(retry) - 1))
				backoff += backoff * (0.1 * mathrand.Float64())
				select {
				case <-time.After(time.Second * time.Duration(backoff)):
					continue
				case <-req.Context().Done():
					return nil, req.Context().Err()
				}
			}
		}
		if err != nil {
			t.vlogf("RoundTrip failure: %v", err)
			return nil, err
		}
		return res, nil
	}
}

// CloseIdleConnections closes any connections which were previously
// connected from previous requests but are now sitting idle.
// It does not interrupt any connections currently in use.
func (t *Transport) CloseIdleConnections() {
	if cp, ok := t.connPool().(clientConnPoolIdleCloser); ok {
		cp.closeIdleConnections()
	}
}

var (
	errClientConnClosed    = errors.New("http2: client conn is closed")
	errClientConnUnusable  = errors.New("http2: client conn not usable")
	errClientConnGotGoAway = errors.New("http2: Transport received Server's graceful shutdown GOAWAY")
)

// shouldRetryRequest is called by RoundTrip when a request fails to get
// response headers. It is always called with a non-nil error.
// It returns either a request to retry (either the same request, or a
// modified clone), or an error if the request can't be replayed.
func shouldRetryRequest(req *http.Request, err error, afterBodyWrite bool) (*http.Request, error) {
	if !canRetryError(err) {
		return nil, err
	}
	// If the Body is nil (or http.NoBody), it's safe to reuse
	// this request and its Body.
	if req.Body == nil || req.Body == http.NoBody {
		return req, nil
	}

	// If the request body can be reset back to its original
	// state via the optional req.GetBody, do that.
	if req.GetBody != nil {
		// TODO: consider a req.Body.Close here? or audit that all caller paths do?
		body, err := req.GetBody()
		if err != nil {
			return nil, err
		}
		newReq := *req
		newReq.Body = body
		return &newReq, nil
	}

	// The Request.Body can't reset back to the beginning, but we
	// don't seem to have started to read from it yet, so reuse
	// the request directly. The "afterBodyWrite" means the
	// bodyWrite process has started, which becomes true before
	// the first Read.
	if !afterBodyWrite {
		return req, nil
	}

	return nil, fmt.Errorf("http2: Transport: cannot retry err [%v] after Request.Body was written; define Request.GetBody to avoid this error", err)
}

func canRetryError(err error) bool {
	if err == errClientConnUnusable || err == errClientConnGotGoAway {
		return true
	}
	if se, ok := err.(StreamError); ok {
		return se.Code == ErrCodeRefusedStream
	}
	return false
}

func (t *Transport) dialClientConn(ctx context.Context, addr string, singleUse bool) (*ClientConn, error) {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, err
	}
	tconn, err := t.dialTLS(ctx)("tcp", addr, t.newTLSConfig(host))
	if err != nil {
		return nil, err
	}
	return t.newClientConn(tconn, singleUse)
}

func (t *Transport) newTLSConfig(host string) *tls.Config {
	cfg := new(tls.Config)
	if t.TLSClientConfig != nil {
		*cfg = *t.TLSClientConfig.Clone()
	}
	if !strSliceContains(cfg.NextProtos, NextProtoTLS) {
		cfg.NextProtos = append([]string{NextProtoTLS}, cfg.NextProtos...)
	}
	if cfg.ServerName == "" {
		cfg.ServerName = host
	}
	return cfg
}

func (t *Transport) dialTLS(ctx context.Context) func(string, string, *tls.Config) (net.Conn, error) {
	if t.DialTLS != nil {
		return t.DialTLS
	}
	return func(network, addr string, cfg *tls.Config) (net.Conn, error) {
		tlsCn, err := t.dialTLSWithContext(ctx, network, addr, cfg)
		if err != nil {
			return nil, err
		}
		state := tlsCn.ConnectionState()
		if p := state.NegotiatedProtocol; p != NextProtoTLS {
			return nil, fmt.Errorf("http2: unexpected ALPN protocol %q; want %q", p, NextProtoTLS)
		}
		if !state.NegotiatedProtocolIsMutual {
			return nil, errors.New("http2: could not negotiate protocol mutually")
		}
		return tlsCn, nil
	}
}

// disableKeepAlives reports whether connections should be closed as
// soon as possible after handling the first request.
func (t *Transport) disableKeepAlives() bool {
	return t.t1 != nil && t.t1.DisableKeepAlives
}

func (t *Transport) expectContinueTimeout() time.Duration {
	if t.t1 == nil {
		return 0
	}
	return t.t1.ExpectContinueTimeout
}

func (t *Transport) NewClientConn(c net.Conn) (*ClientConn, error) {
	return t.newClientConn(c, t.disableKeepAlives())
}

func (t *Transport) newClientConn(c net.Conn, singleUse bool) (*ClientConn, error) {
	cc := &ClientConn{
		t:                     t,
		tconn:                 c,
		readerDone:            make(chan struct{}),
		nextStreamID:          1,
		maxFrameSize:          16 << 10,           // spec default
		initialWindowSize:     65535,              // spec default
		maxConcurrentStreams:  1000,               // "infinite", per spec. 1000 seems good enough.
		peerMaxHeaderListSize: 0xffffffffffffffff, // "infinite", per spec. Use 2^64-1 instead.
		streams:               make(map[uint32]*clientStream),
		singleUse:             singleUse,
		wantSettingsAck:       true,
		pings:                 make(map[[8]byte]chan struct{}),
	}
	if d := t.idleConnTimeout(); d != 0 {
		cc.idleTimeout = d
		cc.idleTimer = time.AfterFunc(d, cc.onIdleTimeout)
	}
	if VerboseLogs {
		t.vlogf("http2: Transport creating client conn %p to %v", cc, c.RemoteAddr())
	}

	cc.cond = sync.NewCond(&cc.mu)
	cc.flow.add(int32(initialWindowSize))

	// TODO: adjust this writer size to account for frame size +
	// MTU + crypto/tls record padding.
	cc.bw = bufio.NewWriter(stickyErrWriter{c, &cc.werr})
	cc.br = bufio.NewReader(c)
	cc.fr = NewFramer(cc.bw, cc.br)
	cc.fr.ReadMetaHeaders = hpack.NewDecoder(initialHeaderTableSize, nil)
	cc.fr.MaxHeaderListSize = t.maxHeaderListSize()

	// TODO: SetMaxDynamicTableSize, SetMaxDynamicTableSizeLimit on
	// henc in response to SETTINGS frames?
	cc.henc = hpack.NewEncoder(&cc.hbuf)

	if t.AllowHTTP {
		cc.nextStreamID = 3
	}

	if cs, ok := c.(connectionStater); ok {
		state := cs.ConnectionState()
		cc.tlsState = &state
	}

	initialSettings := []Setting{
		{ID: SettingEnablePush, Val: 0},
		{ID: SettingInitialWindowSize, Val: transportDefaultStreamFlow},
	}
	if max := t.maxHeaderListSize(); max != 0 {
		initialSettings = append(initialSettings, Setting{ID: SettingMaxHeaderListSize, Val: max})
	}

	cc.bw.Write(clientPreface)
	cc.fr.WriteSettings(initialSettings...)
	cc.fr.WriteWindowUpdate(0, transportDefaultConnFlow)
	cc.inflow.add(transportDefaultConnFlow + initialWindowSize)
	cc.bw.Flush()
	if cc.werr != nil {
		cc.Close()
		return nil, cc.werr
	}

	go cc.readLoop()
	return cc, nil
}

func (cc *ClientConn) healthCheck() {
	pingTimeout := cc.t.pingTimeout()
	// We don't need to periodically ping in the health check, because the readLoop of ClientConn will
	// trigger the healthCheck again if there is no frame received.
	ctx, cancel := context.WithTimeout(context.Background(), pingTimeout)
	defer cancel()
	err := cc.Ping(ctx)
	if err != nil {
		cc.closeForLostPing()
		cc.t.connPool().MarkDead(cc)
		return
	}
}

func (cc *ClientConn) setGoAway(f *GoAwayFrame) {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	old := cc.goAway
	cc.goAway = f

	// Merge the previous and current GoAway error frames.
	if cc.goAwayDebug == "" {
		cc.goAwayDebug = string(f.DebugData())
	}
	if old != nil && old.ErrCode != ErrCodeNo {
		cc.goAway.ErrCode = old.ErrCode
	}
	last := f.LastStreamID
	for streamID, cs := range cc.streams {
		if streamID > last {
			select {
			case cs.resc <- resAndError{err: errClientConnGotGoAway}:
			default:
			}
		}
	}
}

// CanTakeNewRequest reports whether the connection can take a new request,
// meaning it has not been closed or received or sent a GOAWAY.
func (cc *ClientConn) CanTakeNewRequest() bool {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.canTakeNewRequestLocked()
}

// clientConnIdleState describes the suitability of a client
// connection to initiate a new RoundTrip request.
type clientConnIdleState struct {
	canTakeNewRequest bool
	freshConn         bool // whether it's unused by any previous request
}

func (cc *ClientConn) idleState() clientConnIdleState {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.idleStateLocked()
}

func (cc *ClientConn) idleStateLocked() (st clientConnIdleState) {
	if cc.singleUse && cc.nextStreamID > 1 {
		return
	}
	var maxConcurrentOkay bool
	if cc.t.StrictMaxConcurrentStreams {
		// We'll tell the caller we can take a new request to
		// prevent the caller from dialing a new TCP
		// connection, but then we'll block later before
		// writing it.
		maxConcurrentOkay = true
	} else {
		maxConcurrentOkay = int64(len(cc.streams)+1) < int64(cc.maxConcurrentStreams)
	}

	st.canTakeNewRequest = cc.goAway == nil && !cc.closed && !cc.closing && maxConcurrentOkay &&
		int64(cc.nextStreamID)+2*int64(cc.pendingRequests) < math.MaxInt32 &&
		!cc.tooIdleLocked()
	st.freshConn = cc.nextStreamID == 1 && st.canTakeNewRequest
	return
}

func (cc *ClientConn) canTakeNewRequestLocked() bool {
	st := cc.idleStateLocked()
	return st.canTakeNewRequest
}

// tooIdleLocked reports whether this connection has been been sitting idle
// for too much wall time.
func (cc *ClientConn) tooIdleLocked() bool {
	// The Round(0) strips the monontonic clock reading so the
	// times are compared based on their wall time. We don't want
	// to reuse a connection that's been sitting idle during
	// VM/laptop suspend if monotonic time was also frozen.
	return cc.idleTimeout != 0 && !cc.lastIdle.IsZero() && time.Since(cc.lastIdle.Round(0)) > cc.idleTimeout
}

// onIdleTimeout is called from a time.AfterFunc goroutine. It will
// only be called when we're idle, but because we're coming from a new
// goroutine, there could be a new request coming in at the same time,
// so this simply calls the synchronized closeIfIdle to shut down this
// connection. The timer could just call closeIfIdle, but this is more
// clear.
func (cc *ClientConn) onIdleTimeout() {
	cc.closeIfIdle()
}

func (cc *ClientConn) closeIfIdle() {
	cc.mu.Lock()
	if len(cc.streams) > 0 {
		cc.mu.Unlock()
		return
	}
	cc.closed = true
	nextID := cc.nextStreamID
	// TODO: do clients send GOAWAY too? maybe? Just Close:
	cc.mu.Unlock()

	if VerboseLogs {
		cc.vlogf("http2: Transport closing idle conn %p (forSingleUse=%v, maxStream=%v)", cc, cc.singleUse, nextID-2)
	}
	cc.tconn.Close()
}

var shutdownEnterWaitStateHook = func() {}

// Shutdown gracefully close the client connection, waiting for running streams to complete.
func (cc *ClientConn) Shutdown(ctx context.Context) error {
	if err := cc.sendGoAway(); err != nil {
		return err
	}
	// Wait for all in-flight streams to complete or connection to close
	done := make(chan error, 1)
	cancelled := false // guarded by cc.mu
	go func() {
		cc.mu.Lock()
		defer cc.mu.Unlock()
		for {
			if len(cc.streams) == 0 || cc.closed {
				cc.closed = true
				done <- cc.tconn.Close()
				break
			}
			if cancelled {
				break
			}
			cc.cond.Wait()
		}
	}()
	shutdownEnterWaitStateHook()
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		cc.mu.Lock()
		// Free the goroutine above
		cancelled = true
		cc.cond.Broadcast()
		cc.mu.Unlock()
		return ctx.Err()
	}
}

func (cc *ClientConn) sendGoAway() error {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.wmu.Lock()
	defer cc.wmu.Unlock()
	if cc.closing {
		// GOAWAY sent already
		return nil
	}
	// Send a graceful shutdown frame to server
	maxStreamID := cc.nextStreamID
	if err := cc.fr.WriteGoAway(maxStreamID, ErrCodeNo, nil); err != nil {
		return err
	}
	if err := cc.bw.Flush(); err != nil {
		return err
	}
	// Prevent new requests
	cc.closing = true
	return nil
}

// closes the client connection immediately. In-flight requests are interrupted.
// err is sent to streams.
func (cc *ClientConn) closeForError(err error) error {
	cc.mu.Lock()
	defer cc.cond.Broadcast()
	defer cc.mu.Unlock()
	for id, cs := range cc.streams {
		select {
		case cs.resc <- resAndError{err: err}:
		default:
		}
		cs.bufPipe.CloseWithError(err)
		delete(cc.streams, id)
	}
	cc.closed = true
	return cc.tconn.Close()
}

// Close closes the client connection immediately.
//
// In-flight requests are interrupted. For a graceful shutdown, use Shutdown instead.
func (cc *ClientConn) Close() error {
	err := errors.New("http2: client connection force closed via ClientConn.Close")
	return cc.closeForError(err)
}

// closes the client connection immediately. In-flight requests are interrupted.
func (cc *ClientConn) closeForLostPing() error {
	err := errors.New("http2: client connection lost")
	return cc.closeForError(err)
}

const maxAllocFrameSize = 512 << 10

// frameBuffer returns a scratch buffer suitable for writing DATA frames.
// They're capped at the min of the peer's max frame size or 512KB
// (kinda arbitrarily), but definitely capped so we don't allocate 4GB
// bufers.
func (cc *ClientConn) frameScratchBuffer() []byte {
	cc.mu.Lock()
	size := cc.maxFrameSize
	if size > maxAllocFrameSize {
		size = maxAllocFrameSize
	}
	for i, buf := range cc.freeBuf {
		if len(buf) >= int(size) {
			cc.freeBuf[i] = nil
			cc.mu.Unlock()
			return buf[:size]
		}
	}
	cc.mu.Unlock()
	return make([]byte, size)
}

func (cc *ClientConn) putFrameScratchBuffer(buf []byte) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	const maxBufs = 4 // arbitrary; 4 concurrent requests per conn? investigate.
	if len(cc.freeBuf) < maxBufs {
		cc.freeBuf = append(cc.freeBuf, buf)
		return
	}
	for i, old := range cc.freeBuf {
		if old == nil {
			cc.freeBuf[i] = buf
			return
		}
	}
	// forget about it.
}

// errRequestCanceled is a copy of net/http's errRequestCanceled because it's not
// exported. At least they'll be DeepEqual for h1-vs-h2 comparisons tests.
var errRequestCanceled = errors.New("net/http: request canceled")

func commaSeparatedTrailers(req *http.Request) (string, error) {
	keys := make([]string, 0, len(req.Trailer))
	for k := range req.Trailer {
		k = http.CanonicalHeaderKey(k)
		switch k {
		case "Transfer-Encoding", "Trailer", "Content-Length":
			return "", fmt.Errorf("invalid Trailer key %q", k)
		}
		keys = append(keys, k)
	}
	if len(keys) > 0 {
		sort.Strings(keys)
		return strings.Join(keys, ","), nil
	}
	return "", nil
}

func (cc *ClientConn) responseHeaderTimeout() time.Duration {
	if cc.t.t1 != nil {
		return cc.t.t1.ResponseHeaderTimeout
	}
	// No way to do this (yet?) with just an http2.Transport. Probably
	// no need. Request.Cancel this is the new way. We only need to support
	// this for compatibility with the old http.Transport fields when
	// we're doing transparent http2.
	return 0
}

// checkConnHeaders checks whether req has any invalid connection-level headers.
// per RFC 7540 section 8.1.2.2: Connection-Specific Header Fields.
// Certain headers are special-cased as okay but not transmitted later.
func checkConnHeaders(req *http.Request) error {
	if v := req.Header.Get("Upgrade"); v != "" {
		return fmt.Errorf("http2: invalid Upgrade request header: %q", req.Header["Upgrade"])
	}
	if vv := req.Header["Transfer-Encoding"]; len(vv) > 0 && (len(vv) > 1 || vv[0] != "" && vv[0] != "chunked") {
		return fmt.Errorf("http2: invalid Transfer-Encoding request header: %q", vv)
	}
	if vv := req.Header["Connection"]; len(vv) > 0 && (len(vv) > 1 || vv[0] != "" && !asciiEqualFold(vv[0], "close") && !asciiEqualFold(vv[0], "keep-alive")) {
		return fmt.Errorf("http2: invalid Connection request header: %q", vv)
	}
	return nil
}

// actualContentLength returns a sanitized version of
// req.ContentLength, where 0 actually means zero (not unknown) and -1
// means unknown.
func actualContentLength(req *http.Request) int64 {
	if req.Body == nil || req.Body == http.NoBody {
		return 0
	}
	if req.ContentLength != 0 {
		return req.ContentLength
	}
	return -1
}

func (cc *ClientConn) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, _, err := cc.roundTrip(req)
	return resp, err
}

func (cc *ClientConn) roundTrip(req *http.Request) (res *http.Response, gotErrAfterReqBodyWrite bool, err error) {
	if err := checkConnHeaders(req); err != nil {
		return nil, false, err
	}
	if cc.idleTimer != nil {
		cc.idleTimer.Stop()
	}

	trailers, err := commaSeparatedTrailers(req)
	if err != nil {
		return nil, false, err
	}
	hasTrailers := trailers != ""

	cc.mu.Lock()
	if err := cc.awaitOpenSlotForRequest(req); err != nil {
		cc.mu.Unlock()
		return nil, false, err
	}

	body := req.Body
	contentLen := actualContentLength(req)
	hasBody := contentLen != 0

	// TODO(bradfitz): this is a copy of the logic in net/http. Unify somewhere?
	var requestedGzip bool
	if !cc.t.disableCompression() &&
		req.Header.Get("Accept-Encoding") == "" &&
		req.Header.Get("Range") == "" &&
		req.Method != "HEAD" {
		// Request gzip only, not deflate. Deflate is ambiguous and
		// not as universally supported anyway.
		// See: https://zlib.net/zlib_faq.html#faq39
		//
		// Note that we don't request this for HEAD requests,
		// due to a bug in nginx:
		//   http://trac.nginx.org/nginx/ticket/358
		//   https://golang.org/issue/5522
		//
		// We don't request gzip if the request is for a range, since
		// auto-decoding a portion of a gzipped document will just fail
		// anyway. See https://golang.org/issue/8923
		requestedGzip = true
	}

	// we send: HEADERS{1}, CONTINUATION{0,} + DATA{0,} (DATA is
	// sent by writeRequestBody below, along with any Trailers,
	// again in form HEADERS{1}, CONTINUATION{0,})
	hdrs, err := cc.encodeHeaders(req, requestedGzip, trailers, contentLen)
	if err != nil {
		cc.mu.Unlock()
		return nil, false, err
	}

	cs := cc.newStream()
	cs.req = req
	cs.trace = httptrace.ContextClientTrace(req.Context())
	cs.requestedGzip = requestedGzip
	bodyWriter := cc.t.getBodyWriterState(cs, body)
	cs.on100 = bodyWriter.on100

	defer func() {
		cc.wmu.Lock()
		werr := cc.werr
		cc.wmu.Unlock()
		if werr != nil {
			cc.Close()
		}
	}()

	cc.wmu.Lock()
	endStream := !hasBody && !hasTrailers
	werr := cc.writeHeaders(cs.ID, endStream, int(cc.maxFrameSize), hdrs)
	cc.wmu.Unlock()
	traceWroteHeaders(cs.trace)
	cc.mu.Unlock()

	if werr != nil {
		if hasBody {
			req.Body.Close() // per RoundTripper contract
			bodyWriter.cancel()
		}
		cc.forgetStreamID(cs.ID)
		// Don't bother sending a RST_STREAM (our write already failed;
		// no need to keep writing)
		traceWroteRequest(cs.trace, werr)
		return nil, false, werr
	}

	var respHeaderTimer <-chan time.Time
	if hasBody {
		bodyWriter.scheduleBodyWrite()
	} else {
		traceWroteRequest(cs.trace, nil)
		if d := cc.responseHeaderTimeout(); d != 0 {
			timer := time.NewTimer(d)
			defer timer.Stop()
			respHeaderTimer = timer.C
		}
	}

	readLoopResCh := cs.resc
	bodyWritten := false
	ctx := req.Context()

	handleReadLoopResponse := func(re resAndError) (*http.Response, bool, error) {
		res := re.res
		if re.err != nil || res.StatusCode > 299 {
			// On error or status code 3xx, 4xx, 5xx, etc abort any
			// ongoing write, assuming that the server doesn't care
			// about our request body. If the server replied with 1xx or
			// 2xx, however, then assume the server DOES potentially
			// want our body (e.g. full-duplex streaming:
			// golang.org/issue/13444). If it turns out the server
			// doesn't, they'll RST_STREAM us soon enough. This is a
			// heuristic to avoid adding knobs to Transport. Hopefully
			// we can keep it.
			bodyWriter.cancel()
			cs.abortRequestBodyWrite(errStopReqBodyWrite)
			if hasBody && !bodyWritten {
				<-bodyWriter.resc
			}
		}
		if re.err != nil {
			cc.forgetStreamID(cs.ID)
			return nil, cs.getStartedWrite(), re.err
		}
		res.Request = req
		res.TLS = cc.tlsState
		return res, false, nil
	}

	for {
		select {
		case re := <-readLoopResCh:
			return handleReadLoopResponse(re)
		case <-respHeaderTimer:
			if !hasBody || bodyWritten {
				cc.writeStreamReset(cs.ID, ErrCodeCancel, nil)
			} else {
				bodyWriter.cancel()
				cs.abortRequestBodyWrite(errStopReqBodyWriteAndCancel)
				<-bodyWriter.resc
			}
			cc.forgetStreamID(cs.ID)
			return nil, cs.getStartedWrite(), errTimeout
		case <-ctx.Done():
			if !hasBody || bodyWritten {
				cc.writeStreamReset(cs.ID, ErrCodeCancel, nil)
			} else {
				bodyWriter.cancel()
				cs.abortRequestBodyWrite(errStopReqBodyWriteAndCancel)
				<-bodyWriter.resc
			}
			cc.forgetStreamID(cs.ID)
			return nil, cs.getStartedWrite(), ctx.Err()
		case <-req.Cancel:
			if !hasBody || bodyWritten {
				cc.writeStreamReset(cs.ID, ErrCodeCancel, nil)
			} else {
				bodyWriter.cancel()
				cs.abortRequestBodyWrite(errStopReqBodyWriteAndCancel)
				<-bodyWriter.resc
			}
			cc.forgetStreamID(cs.ID)
			return nil, cs.getStartedWrite(), errRequestCanceled
		case <-cs.peerReset:
			// processResetStream already removed the
			// stream from the streams map; no need for
			// forgetStreamID.
			return nil, cs.getStartedWrite(), cs.resetErr
		case err := <-bodyWriter.resc:
			bodyWritten = true
			// Prefer the read loop's response, if available. Issue 16102.
			select {
			case re := <-readLoopResCh:
				return handleReadLoopResponse(re)
			default:
			}
			if err != nil {
				cc.forgetStreamID(cs.ID)
				return nil, cs.getStartedWrite(), err
			}
			if d := cc.responseHeaderTimeout(); d != 0 {
				timer := time.NewTimer(d)
				defer timer.Stop()
				respHeaderTimer = timer.C
			}
		}
	}
}

// awaitOpenSlotForRequest waits until len(streams) < maxConcurrentStreams.
// Must hold cc.mu.
func (cc *ClientConn) awaitOpenSlotForRequest(req *http.Request) error {
	var waitingForConn chan struct{}
	var waitingForConnErr error // guarded by cc.mu
	for {
		cc.lastActive = time.Now()
		if cc.closed || !cc.canTakeNewRequestLocked() {
			if waitingForConn != nil {
				close(waitingForConn)
			}
			return errClientConnUnusable
		}
		cc.lastIdle = time.Time{}
		if int64(len(cc.streams))+1 <= int64(cc.maxConcurrentStreams) {
			if waitingForConn != nil {
				close(waitingForConn)
			}
			return nil
		}
		// Unfortunately, we cannot wait on a condition variable and channel at
		// the same time, so instead, we spin up a goroutine to check if the
		// request is canceled while we wait for a slot to open in the connection.
		if waitingForConn == nil {
			waitingForConn = make(chan struct{})
			go func() {
				if err := awaitRequestCancel(req, waitingForConn); err != nil {
					cc.mu.Lock()
					waitingForConnErr = err
					cc.cond.Broadcast()
					cc.mu.Unlock()
				}
			}()
		}
		cc.pendingRequests++
		cc.cond.Wait()
		cc.pendingRequests--
		if waitingForConnErr != nil {
			return waitingForConnErr
		}
	}
}

// requires cc.wmu be held
func (cc *ClientConn) writeHeaders(streamID uint32, endStream bool, maxFrameSize int, hdrs []byte) error {
	first := true // first frame written (HEADERS is first, then CONTINUATION)
	for len(hdrs) > 0 && cc.werr == nil {
		chunk := hdrs
		if len(chunk) > maxFrameSize {
			chunk = chunk[:maxFrameSize]
		}
		hdrs = hdrs[len(chunk):]
		endHeaders := len(hdrs) == 0
		if first {
			cc.fr.WriteHeaders(HeadersFrameParam{
				StreamID:      streamID,
				BlockFragment: chunk,
				EndStream:     endStream,
				EndHeaders:    endHeaders,
			})
			first = false
		} else {
			cc.fr.WriteContinuation(streamID, endHeaders, chunk)
		}
	}
	// TODO(bradfitz): this Flush could potentially block (as
	// could the WriteHeaders call(s) above), which means they
	// wouldn't respond to Request.Cancel being readable. That's
	// rare, but this should probably be in a goroutine.
	cc.bw.Flush()
	return cc.werr
}

// internal error values; they don't escape to callers
var (
	// abort request body write; don't send cancel
	errStopReqBodyWrite = errors.New("http2: aborting request body write")

	// abort request body write, but send stream reset of cancel.
	errStopReqBodyWriteAndCancel = errors.New("http2: canceling request")

	errReqBodyTooLong = errors.New("http2: request body larger than specified content length")
)

func (cs *clientStream) writeRequestBody(body io.Reader, bodyCloser io.Closer) (err error) {
	cc := cs.cc
	sentEnd := false // whether we sent the final DATA frame w/ END_STREAM
	buf := cc.frameScratchBuffer()
	defer cc.putFrameScratchBuffer(buf)

	defer func() {
		traceWroteRequest(cs.trace, err)
		// TODO: write h12Compare test showing whether
		// Request.Body is closed by the Transport,
		// and in multiple cases: server replies <=299 and >299
		// while still writing request body
		cerr := bodyCloser.Close()
		if err == nil {
			err = cerr
		}
	}()

	req := cs.req
	hasTrailers := req.Trailer != nil
	remainLen := actualContentLength(req)
	hasContentLen := remainLen != -1

	var sawEOF bool
	for !sawEOF {
		n, err := body.Read(buf[:len(buf)-1])
		if hasContentLen {
			remainLen -= int64(n)
			if remainLen == 0 && err == nil {
				// The request body's Content-Length was predeclared and
				// we just finished reading it all, but the underlying io.Reader
				// returned the final chunk with a nil error (which is one of
				// the two valid things a Reader can do at EOF). Because we'd prefer
				// to send the END_STREAM bit early, double-check that we're actually
				// at EOF. Subsequent reads should return (0, EOF) at this point.
				// If either value is different, we return an error in one of two ways below.
				var n1 int
				n1, err = body.Read(buf[n:])
				remainLen -= int64(n1)
			}
			if remainLen < 0 {
				err = errReqBodyTooLong
				cc.writeStreamReset(cs.ID, ErrCodeCancel, err)
				return err
			}
		}
		if err == io.EOF {
			sawEOF = true
			err = nil
		} else if err != nil {
			cc.writeStreamReset(cs.ID, ErrCodeCancel, err)
			return err
		}

		remain := buf[:n]
		for len(remain) > 0 && err == nil {
			var allowed int32
			allowed, err = cs.awaitFlowControl(len(remain))
			switch {
			case err == errStopReqBodyWrite:
				return err
			case err == errStopReqBodyWriteAndCancel:
				cc.writeStreamReset(cs.ID, ErrCodeCancel, nil)
				return err
			case err != nil:
				return err
			}
			cc.wmu.Lock()
			data := remain[:allowed]
			remain = remain[allowed:]
			sentEnd = sawEOF && len(remain) == 0 && !hasTrailers
			err = cc.fr.WriteData(cs.ID, sentEnd, data)
			if err == nil {
				// TODO(bradfitz): this flush is for latency, not bandwidth.
				// Most requests won't need this. Make this opt-in or
				// opt-out?  Use some heuristic on the body type? Nagel-like
				// timers?  Based on 'n'? Only last chunk of this for loop,
				// unless flow control tokens are low? For now, always.
				// If we change this, see comment below.
				err = cc.bw.Flush()
			}
			cc.wmu.Unlock()
		}
		if err != nil {
			return err
		}
	}

	if sentEnd {
		// Already sent END_STREAM (which implies we have no
		// trailers) and flushed, because currently all
		// WriteData frames above get a flush. So we're done.
		return nil
	}

	var trls []byte
	if hasTrailers {
		cc.mu.Lock()
		trls, err = cc.encodeTrailers(req)
		cc.mu.Unlock()
		if err != nil {
			cc.writeStreamReset(cs.ID, ErrCodeInternal, err)
			cc.forgetStreamID(cs.ID)
			return err
		}
	}

	cc.mu.Lock()
	maxFrameSize := int(cc.maxFrameSize)
	cc.mu.Unlock()

	cc.wmu.Lock()
	defer cc.wmu.Unlock()

	// Two ways to send END_STREAM: either with trailers, or
	// with an empty DATA frame.
	if len(trls) > 0 {
		err = cc.writeHeaders(cs.ID, true, maxFrameSize, trls)
	} else {
		err = cc.fr.WriteData(cs.ID, true, nil)
	}
	if ferr := cc.bw.Flush(); ferr != nil && err == nil {
		err = ferr
	}
	return err
}

// awaitFlowControl waits for [1, min(maxBytes, cc.cs.maxFrameSize)] flow
// control tokens from the server.
// It returns either the non-zero number of tokens taken or an error
// if the stream is dead.
func (cs *clientStream) awaitFlowControl(maxBytes int) (taken int32, err error) {
	cc := cs.cc
	cc.mu.Lock()
	defer cc.mu.Unlock()
	for {
		if cc.closed {
			return 0, errClientConnClosed
		}
		if cs.stopReqBody != nil {
			return 0, cs.stopReqBody
		}
		if err := cs.checkResetOrDone(); err != nil {
			return 0, err
		}
		if a := cs.flow.available(); a > 0 {
			take := a
			if int(take) > maxBytes {

				take = int32(maxBytes) // can't truncate int; take is int32
			}
			if take > int32(cc.maxFrameSize) {
				take = int32(cc.maxFrameSize)
			}
			cs.flow.take(take)
			return take, nil
		}
		cc.cond.Wait()
	}
}

// requires cc.mu be held.
func (cc *ClientConn) encodeHeaders(req *http.Request, addGzipHeader bool, trailers string, contentLength int64) ([]byte, error) {
	cc.hbuf.Reset()

	host := req.Host
	if host == "" {
		host = req.URL.Host
	}
	host, err := httpguts.PunycodeHostPort(host)
	if err != nil {
		return nil, err
	}

	var path string
	if req.Method != "CONNECT" {
		path = req.URL.RequestURI()
		if !validPseudoPath(path) {
			orig := path
			path = strings.TrimPrefix(path, req.URL.Scheme+"://"+host)
			if !validPseudoPath(path) {
				if req.URL.Opaque != "" {
					return nil, fmt.Errorf("invalid request :path %q from URL.Opaque = %q", orig, req.URL.Opaque)
				} else {
					return nil, fmt.Errorf("invalid request :path %q", orig)
				}
			}
		}
	}

	// Check for any invalid headers and return an error before we
	// potentially pollute our hpack state. (We want to be able to
	// continue to reuse the hpack encoder for future requests)
	for k, vv := range req.Header {
		if !httpguts.ValidHeaderFieldName(k) {
			return nil, fmt.Errorf("invalid HTTP header name %q", k)
		}
		for _, v := range vv {
			if !httpguts.ValidHeaderFieldValue(v) {
				return nil, fmt.Errorf("invalid HTTP header value %q for header %q", v, k)
			}
		}
	}

	enumerateHeaders := func(f func(name, value string)) {
		// 8.1.2.3 Request Pseudo-Header Fields
		// The :path pseudo-header field includes the path and query parts of the
		// target URI (the path-absolute production and optionally a '?' character
		// followed by the query production (see Sections 3.3 and 3.4 of
		// [RFC3986]).
		f(":authority", host)
		m := req.Method
		if m == "" {
			m = http.MethodGet
		}
		f(":method", m)
		if req.Method != "CONNECT" {
			f(":path", path)
			f(":scheme", req.URL.Scheme)
		}
		if trailers != "" {
			f("trailer", trailers)
		}

		var didUA bool
		for k, vv := range req.Header {
			if asciiEqualFold(k, "host") || asciiEqualFold(k, "content-length") {
				// Host is :authority, already sent.
				// Content-Length is automatic, set below.
				continue
			} else if asciiEqualFold(k, "connection") ||
				asciiEqualFold(k, "proxy-connection") ||
				asciiEqualFold(k, "transfer-encoding") ||
				asciiEqualFold(k, "upgrade") ||
				asciiEqualFold(k, "keep-alive") {
				// Per 8.1.2.2 Connection-Specific Header
				// Fields, don't send connection-specific
				// fields. We have already checked if any
				// are error-worthy so just ignore the rest.
				continue
			} else if asciiEqualFold(k, "user-agent") {
				// Match Go's http1 behavior: at most one
				// User-Agent. If set to nil or empty string,
				// then omit it. Otherwise if not mentioned,
				// include the default (below).
				didUA = true
				if len(vv) < 1 {
					continue
				}
				vv = vv[:1]
				if vv[0] == "" {
					continue
				}
			} else if asciiEqualFold(k, "cookie") {
				// Per 8.1.2.5 To allow for better compression efficiency, the
				// Cookie header field MAY be split into separate header fields,
				// each with one or more cookie-pairs.
				for _, v := range vv {
					for {
						p := strings.IndexByte(v, ';')
						if p < 0 {
							break
						}
						f("cookie", v[:p])
						p++
						// strip space after semicolon if any.
						for p+1 <= len(v) && v[p] == ' ' {
							p++
						}
						v = v[p:]
					}
					if len(v) > 0 {
						f("cookie", v)
					}
				}
				continue
			}

			for _, v := range vv {
				f(k, v)
			}
		}
		if shouldSendReqContentLength(req.Method, contentLength) {
			f("content-length", strconv.FormatInt(contentLength, 10))
		}
		if addGzipHeader {
			f("accept-encoding", "gzip")
		}
		if !didUA {
			f("user-agent", defaultUserAgent)
		}
	}

	// Do a first pass over the headers counting bytes to ensure
	// we don't exceed cc.peerMaxHeaderListSize. This is done as a
	// separate pass before encoding the headers to prevent
	// modifying the hpack state.
	hlSize := uint64(0)
	enumerateHeaders(func(name, value string) {
		hf := hpack.HeaderField{Name: name, Value: value}
		hlSize += uint64(hf.Size())
	})

	if hlSize > cc.peerMaxHeaderListSize {
		return nil, errRequestHeaderListSize
	}

	trace := httptrace.ContextClientTrace(req.Context())
	traceHeaders := traceHasWroteHeaderField(trace)

	// Header list size is ok. Write the headers.
	enumerateHeaders(func(name, value string) {
		name, ascii := asciiToLower(name)
		if !ascii {
			// Skip writing invalid headers. Per RFC 7540, Section 8.1.2, header
			// field names have to be ASCII characters (just as in HTTP/1.x).
			return
		}
		cc.writeHeader(name, value)
		if traceHeaders {
			traceWroteHeaderField(trace, name, value)
		}
	})

	return cc.hbuf.Bytes(), nil
}

// shouldSendReqContentLength reports whether the http2.Transport should send
// a "content-length" request header. This logic is basically a copy of the net/http
// transferWriter.shouldSendContentLength.
// The contentLength is the corrected contentLength (so 0 means actually 0, not unknown).
// -1 means unknown.
func shouldSendReqContentLength(method string, contentLength int64) bool {
	if contentLength > 0 {
		return true
	}
	if contentLength < 0 {
		return false
	}
	// For zero bodies, whether we send a content-length depends on the method.
	// It also kinda doesn't matter for http2 either way, with END_STREAM.
	switch method {
	case "POST", "PUT", "PATCH":
		return true
	default:
		return false
	}
}

// requires cc.mu be held.
func (cc *ClientConn) encodeTrailers(req *http.Request) ([]byte, error) {
	cc.hbuf.Reset()

	hlSize := uint64(0)
	for k, vv := range req.Trailer {
		for _, v := range vv {
			hf := hpack.HeaderField{Name: k, Value: v}
			hlSize += uint64(hf.Size())
		}
	}
	if hlSize > cc.peerMaxHeaderListSize {
		return nil, errRequestHeaderListSize
	}

	for k, vv := range req.Trailer {
		lowKey, ascii := asciiToLower(k)
		if !ascii {
			// Skip writing invalid headers. Per RFC 7540, Section 8.1.2, header
			// field names have to be ASCII characters (just as in HTTP/1.x).
			continue
		}
		// Transfer-Encoding, etc.. have already been filtered at the
		// start of RoundTrip
		for _, v := range vv {
			cc.writeHeader(lowKey, v)
		}
	}
	return cc.hbuf.Bytes(), nil
}

func (cc *ClientConn) writeHeader(name, value string) {
	if VerboseLogs {
		log.Printf("http2: Transport encoding header %q = %q", name, value)
	}
	cc.henc.WriteField(hpack.HeaderField{Name: name, Value: value})
}

type resAndError struct {
	_   incomparable
	res *http.Response
	err error
}

// requires cc.mu be held.
func (cc *ClientConn) newStream() *clientStream {
	cs := &clientStream{
		cc:        cc,
		ID:        cc.nextStreamID,
		resc:      make(chan resAndError, 1),
		peerReset: make(chan struct{}),
		done:      make(chan struct{}),
	}
	cs.flow.add(int32(cc.initialWindowSize))
	cs.flow.setConnFlow(&cc.flow)
	cs.inflow.add(transportDefaultStreamFlow)
	cs.inflow.setConnFlow(&cc.inflow)
	cc.nextStreamID += 2
	cc.streams[cs.ID] = cs
	return cs
}

func (cc *ClientConn) forgetStreamID(id uint32) {
	cc.streamByID(id, true)
}

func (cc *ClientConn) streamByID(id uint32, andRemove bool) *clientStream {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cs := cc.streams[id]
	if andRemove && cs != nil && !cc.closed {
		cc.lastActive = time.Now()
		delete(cc.streams, id)
		if len(cc.streams) == 0 && cc.idleTimer != nil {
			cc.idleTimer.Reset(cc.idleTimeout)
			cc.lastIdle = time.Now()
		}
		close(cs.done)
		// Wake up checkResetOrDone via clientStream.awaitFlowControl and
		// wake up RoundTrip if there is a pending request.
		cc.cond.Broadcast()
	}
	return cs
}

// clientConnReadLoop is the state owned by the clientConn's frame-reading readLoop.
type clientConnReadLoop struct {
	_             incomparable
	cc            *ClientConn
	closeWhenIdle bool
}

// readLoop runs in its own goroutine and reads and dispatches frames.
func (cc *ClientConn) readLoop() {
	rl := &clientConnReadLoop{cc: cc}
	defer rl.cleanup()
	cc.readerErr = rl.run()
	if ce, ok := cc.readerErr.(ConnectionError); ok {
		cc.wmu.Lock()
		cc.fr.WriteGoAway(0, ErrCode(ce), nil)
		cc.wmu.Unlock()
	}
}

// GoAwayError is returned by the Transport when the server closes the
// TCP connection after sending a GOAWAY frame.
type GoAwayError struct {
	LastStreamID uint32
	ErrCode      ErrCode
	DebugData    string
}

func (e GoAwayError) Error() string {
	return fmt.Sprintf("http2: server sent GOAWAY and closed the connection; LastStreamID=%v, ErrCode=%v, debug=%q",
		e.LastStreamID, e.ErrCode, e.DebugData)
}

func isEOFOrNetReadError(err error) bool {
	if err == io.EOF {
		return true
	}
	ne, ok := err.(*net.OpError)
	return ok && ne.Op == "read"
}

func (rl *clientConnReadLoop) cleanup() {
	cc := rl.cc
	defer cc.tconn.Close()
	defer cc.t.connPool().MarkDead(cc)
	defer close(cc.readerDone)

	if cc.idleTimer != nil {
		cc.idleTimer.Stop()
	}

	// Close any response bodies if the server closes prematurely.
	// TODO: also do this if we've written the headers but not
	// gotten a response yet.
	err := cc.readerErr
	cc.mu.Lock()
	if cc.goAway != nil && isEOFOrNetReadError(err) {
		err = GoAwayError{
			LastStreamID: cc.goAway.LastStreamID,
			ErrCode:      cc.goAway.ErrCode,
			DebugData:    cc.goAwayDebug,
		}
	} else if err == io.EOF {
		err = io.ErrUnexpectedEOF
	}
	for _, cs := range cc.streams {
		cs.bufPipe.CloseWithError(err) // no-op if already closed
		select {
		case cs.resc <- resAndError{err: err}:
		default:
		}
		close(cs.done)
	}
	cc.closed = true
	cc.cond.Broadcast()
	cc.mu.Unlock()
}

func (rl *clientConnReadLoop) run() error {
	cc := rl.cc
	rl.closeWhenIdle = cc.t.disableKeepAlives() || cc.singleUse
	gotReply := false // ever saw a HEADERS reply
	gotSettings := false
	readIdleTimeout := cc.t.ReadIdleTimeout
	var t *time.Timer
	if readIdleTimeout != 0 {
		t = time.AfterFunc(readIdleTimeout, cc.healthCheck)
		defer t.Stop()
	}
	for {
		f, err := cc.fr.ReadFrame()
		if t != nil {
			t.Reset(readIdleTimeout)
		}
		if err != nil {
			cc.vlogf("http2: Transport readFrame error on conn %p: (%T) %v", cc, err, err)
		}
		if se, ok := err.(StreamError); ok {
			if cs := cc.streamByID(se.StreamID, false); cs != nil {
				cs.cc.writeStreamReset(cs.ID, se.Code, err)
				cs.cc.forgetStreamID(cs.ID)
				if se.Cause == nil {
					se.Cause = cc.fr.errDetail
				}
				rl.endStreamError(cs, se)
			}
			continue
		} else if err != nil {
			return err
		}
		if VerboseLogs {
			cc.vlogf("http2: Transport received %s", summarizeFrame(f))
		}
		if !gotSettings {
			if _, ok := f.(*SettingsFrame); !ok {
				cc.logf("protocol error: received %T before a SETTINGS frame", f)
				return ConnectionError(ErrCodeProtocol)
			}
			gotSettings = true
		}
		maybeIdle := false // whether frame might transition us to idle

		switch f := f.(type) {
		case *MetaHeadersFrame:
			err = rl.processHeaders(f)
			maybeIdle = true
			gotReply = true
		case *DataFrame:
			err = rl.processData(f)
			maybeIdle = true
		case *GoAwayFrame:
			err = rl.processGoAway(f)
			maybeIdle = true
		case *RSTStreamFrame:
			err = rl.processResetStream(f)
			maybeIdle = true
		case *SettingsFrame:
			err = rl.processSettings(f)
		case *PushPromiseFrame:
			err = rl.processPushPromise(f)
		case *WindowUpdateFrame:
			err = rl.processWindowUpdate(f)
		case *PingFrame:
			err = rl.processPing(f)
		default:
			cc.logf("Transport: unhandled response frame type %T", f)
		}
		if err != nil {
			if VerboseLogs {
				cc.vlogf("http2: Transport conn %p received error from processing frame %v: %v", cc, summarizeFrame(f), err)
			}
			return err
		}
		if rl.closeWhenIdle && gotReply && maybeIdle {
			cc.closeIfIdle()
		}
	}
}

func (rl *clientConnReadLoop) processHeaders(f *MetaHeadersFrame) error {
	cc := rl.cc
	cs := cc.streamByID(f.StreamID, false)
	if cs == nil {
		// We'd get here if we canceled a request while the
		// server had its response still in flight. So if this
		// was just something we canceled, ignore it.
		return nil
	}
	if f.StreamEnded() {
		// Issue 20521: If the stream has ended, streamByID() causes
		// clientStream.done to be closed, which causes the request's bodyWriter
		// to be closed with an errStreamClosed, which may be received by
		// clientConn.RoundTrip before the result of processing these headers.
		// Deferring stream closure allows the header processing to occur first.
		// clientConn.RoundTrip may still receive the bodyWriter error first, but
		// the fix for issue 16102 prioritises any response.
		//
		// Issue 22413: If there is no request body, we should close the
		// stream before writing to cs.resc so that the stream is closed
		// immediately once RoundTrip returns.
		if cs.req.Body != nil {
			defer cc.forgetStreamID(f.StreamID)
		} else {
			cc.forgetStreamID(f.StreamID)
		}
	}
	if !cs.firstByte {
		if cs.trace != nil {
			// TODO(bradfitz): move first response byte earlier,
			// when we first read the 9 byte header, not waiting
			// until all the HEADERS+CONTINUATION frames have been
			// merged. This works for now.
			traceFirstResponseByte(cs.trace)
		}
		cs.firstByte = true
	}
	if !cs.pastHeaders {
		cs.pastHeaders = true
	} else {
		return rl.processTrailers(cs, f)
	}

	res, err := rl.handleResponse(cs, f)
	if err != nil {
		if _, ok := err.(ConnectionError); ok {
			return err
		}
		// Any other error type is a stream error.
		cs.cc.writeStreamReset(f.StreamID, ErrCodeProtocol, err)
		cc.forgetStreamID(cs.ID)
		cs.resc <- resAndError{err: err}
		return nil // return nil from process* funcs to keep conn alive
	}
	if res == nil {
		// (nil, nil) special case. See handleResponse docs.
		return nil
	}
	cs.resTrailer = &res.Trailer
	cs.resc <- resAndError{res: res}
	return nil
}

// may return error types nil, or ConnectionError. Any other error value
// is a StreamError of type ErrCodeProtocol. The returned error in that case
// is the detail.
//
// As a special case, handleResponse may return (nil, nil) to skip the
// frame (currently only used for 1xx responses).
func (rl *clientConnReadLoop) handleResponse(cs *clientStream, f *MetaHeadersFrame) (*http.Response, error) {
	if f.Truncated {
		return nil, errResponseHeaderListSize
	}

	status := f.PseudoValue("status")
	if status == "" {
		return nil, errors.New("malformed response from server: missing status pseudo header")
	}
	statusCode, err := strconv.Atoi(status)
	if err != nil {
		return nil, errors.New("malformed response from server: malformed non-numeric status pseudo header")
	}

	regularFields := f.RegularFields()
	strs := make([]string, len(regularFields))
	header := make(http.Header, len(regularFields))
	res := &http.Response{
		Proto:      "HTTP/2.0",
		ProtoMajor: 2,
		Header:     header,
		StatusCode: statusCode,
		Status:     status + " " + http.StatusText(statusCode),
	}
	for _, hf := range regularFields {
		key := http.CanonicalHeaderKey(hf.Name)
		if key == "Trailer" {
			t := res.Trailer
			if t == nil {
				t = make(http.Header)
				res.Trailer = t
			}
			foreachHeaderElement(hf.Value, func(v string) {
				t[http.CanonicalHeaderKey(v)] = nil
			})
		} else {
			vv := header[key]
			if vv == nil && len(strs) > 0 {
				// More than likely this will be a single-element key.
				// Most headers aren't multi-valued.
				// Set the capacity on strs[0] to 1, so any future append
				// won't extend the slice into the other strings.
				vv, strs = strs[:1:1], strs[1:]
				vv[0] = hf.Value
				header[key] = vv
			} else {
				header[key] = append(vv, hf.Value)
			}
		}
	}

	if statusCode >= 100 && statusCode <= 199 {
		cs.num1xx++
		const max1xxResponses = 5 // arbitrary bound on number of informational responses, same as net/http
		if cs.num1xx > max1xxResponses {
			return nil, errors.New("http2: too many 1xx informational responses")
		}
		if fn := cs.get1xxTraceFunc(); fn != nil {
			if err := fn(statusCode, textproto.MIMEHeader(header)); err != nil {
				return nil, err
			}
		}
		if statusCode == 100 {
			traceGot100Continue(cs.trace)
			if cs.on100 != nil {
				cs.on100() // forces any write delay timer to fire
			}
		}
		cs.pastHeaders = false // do it all again
		return nil, nil
	}

	streamEnded := f.StreamEnded()
	isHead := cs.req.Method == "HEAD"
	if !streamEnded || isHead {
		res.ContentLength = -1
		if clens := res.Header["Content-Length"]; len(clens) == 1 {
			if cl, err := strconv.ParseUint(clens[0], 10, 63); err == nil {
				res.ContentLength = int64(cl)
			} else {
				// TODO: care? unlike http/1, it won't mess up our framing, so it's
				// more safe smuggling-wise to ignore.
			}
		} else if len(clens) > 1 {
			// TODO: care? unlike http/1, it won't mess up our framing, so it's
			// more safe smuggling-wise to ignore.
		}
	}

	if streamEnded || isHead {
		res.Body = noBody
		return res, nil
	}

	cs.bufPipe = pipe{b: &dataBuffer{expected: res.ContentLength}}
	cs.bytesRemain = res.ContentLength
	res.Body = transportResponseBody{cs}
	go cs.awaitRequestCancel(cs.req)

	if cs.requestedGzip && res.Header.Get("Content-Encoding") == "gzip" {
		res.Header.Del("Content-Encoding")
		res.Header.Del("Content-Length")
		res.ContentLength = -1
		res.Body = &gzipReader{body: res.Body}
		res.Uncompressed = true
	}
	return res, nil
}

func (rl *clientConnReadLoop) processTrailers(cs *clientStream, f *MetaHeadersFrame) error {
	if cs.pastTrailers {
		// Too many HEADERS frames for this stream.
		return ConnectionError(ErrCodeProtocol)
	}
	cs.pastTrailers = true
	if !f.StreamEnded() {
		// We expect that any headers for trailers also
		// has END_STREAM.
		return ConnectionError(ErrCodeProtocol)
	}
	if len(f.PseudoFields()) > 0 {
		// No pseudo header fields are defined for trailers.
		// TODO: ConnectionError might be overly harsh? Check.
		return ConnectionError(ErrCodeProtocol)
	}

	trailer := make(http.Header)
	for _, hf := range f.RegularFields() {
		key := http.CanonicalHeaderKey(hf.Name)
		trailer[key] = append(trailer[key], hf.Value)
	}
	cs.trailer = trailer

	rl.endStream(cs)
	return nil
}

// transportResponseBody is the concrete type of Transport.RoundTrip's
// Response.Body. It is an io.ReadCloser. On Read, it reads from cs.body.
// On Close it sends RST_STREAM if EOF wasn't already seen.
type transportResponseBody struct {
	cs *clientStream
}

func (b transportResponseBody) Read(p []byte) (n int, err error) {
	cs := b.cs
	cc := cs.cc

	if cs.readErr != nil {
		return 0, cs.readErr
	}
	n, err = b.cs.bufPipe.Read(p)
	if cs.bytesRemain != -1 {
		if int64(n) > cs.bytesRemain {
			n = int(cs.bytesRemain)
			if err == nil {
				err = errors.New("net/http: server replied with more than declared Content-Length; truncated")
				cc.writeStreamReset(cs.ID, ErrCodeProtocol, err)
			}
			cs.readErr = err
			return int(cs.bytesRemain), err
		}
		cs.bytesRemain -= int64(n)
		if err == io.EOF && cs.bytesRemain > 0 {
			err = io.ErrUnexpectedEOF
			cs.readErr = err
			return n, err
		}
	}
	if n == 0 {
		// No flow control tokens to send back.
		return
	}

	cc.mu.Lock()
	defer cc.mu.Unlock()

	var connAdd, streamAdd int32
	// Check the conn-level first, before the stream-level.
	if v := cc.inflow.available(); v < transportDefaultConnFlow/2 {
		connAdd = transportDefaultConnFlow - v
		cc.inflow.add(connAdd)
	}
	if err == nil { // No need to refresh if the stream is over or failed.
		// Consider any buffered body data (read from the conn but not
		// consumed by the client) when computing flow control for this
		// stream.
		v := int(cs.inflow.available()) + cs.bufPipe.Len()
		if v < transportDefaultStreamFlow-transportDefaultStreamMinRefresh {
			streamAdd = int32(transportDefaultStreamFlow - v)
			cs.inflow.add(streamAdd)
		}
	}
	if connAdd != 0 || streamAdd != 0 {
		cc.wmu.Lock()
		defer cc.wmu.Unlock()
		if connAdd != 0 {
			cc.fr.WriteWindowUpdate(0, mustUint31(connAdd))
		}
		if streamAdd != 0 {
			cc.fr.WriteWindowUpdate(cs.ID, mustUint31(streamAdd))
		}
		cc.bw.Flush()
	}
	return
}

var errClosedResponseBody = errors.New("http2: response body closed")

func (b transportResponseBody) Close() error {
	cs := b.cs
	cc := cs.cc

	serverSentStreamEnd := cs.bufPipe.Err() == io.EOF
	unread := cs.bufPipe.Len()

	if unread > 0 || !serverSentStreamEnd {
		cc.mu.Lock()
		cc.wmu.Lock()
		if !serverSentStreamEnd {
			cc.fr.WriteRSTStream(cs.ID, ErrCodeCancel)
			cs.didReset = true
		}
		// Return connection-level flow control.
		if unread > 0 {
			cc.inflow.add(int32(unread))
			cc.fr.WriteWindowUpdate(0, uint32(unread))
		}
		cc.bw.Flush()
		cc.wmu.Unlock()
		cc.mu.Unlock()
	}

	cs.bufPipe.BreakWithError(errClosedResponseBody)
	cc.forgetStreamID(cs.ID)
	return nil
}

func (rl *clientConnReadLoop) processData(f *DataFrame) error {
	cc := rl.cc
	cs := cc.streamByID(f.StreamID, f.StreamEnded())
	data := f.Data()
	if cs == nil {
		cc.mu.Lock()
		neverSent := cc.nextStreamID
		cc.mu.Unlock()
		if f.StreamID >= neverSent {
			// We never asked for this.
			cc.logf("http2: Transport received unsolicited DATA frame; closing connection")
			return ConnectionError(ErrCodeProtocol)
		}
		// We probably did ask for this, but canceled. Just ignore it.
		// TODO: be stricter here? only silently ignore things which
		// we canceled, but not things which were closed normally
		// by the peer? Tough without accumulating too much state.

		// But at least return their flow control:
		if f.Length > 0 {
			cc.mu.Lock()
			cc.inflow.add(int32(f.Length))
			cc.mu.Unlock()

			cc.wmu.Lock()
			cc.fr.WriteWindowUpdate(0, uint32(f.Length))
			cc.bw.Flush()
			cc.wmu.Unlock()
		}
		return nil
	}
	if !cs.firstByte {
		cc.logf("protocol error: received DATA before a HEADERS frame")
		rl.endStreamError(cs, StreamError{
			StreamID: f.StreamID,
			Code:     ErrCodeProtocol,
		})
		return nil
	}
	if f.Length > 0 {
		if cs.req.Method == "HEAD" && len(data) > 0 {
			cc.logf("protocol error: received DATA on a HEAD request")
			rl.endStreamError(cs, StreamError{
				StreamID: f.StreamID,
				Code:     ErrCodeProtocol,
			})
			return nil
		}
		// Check connection-level flow control.
		cc.mu.Lock()
		if cs.inflow.available() >= int32(f.Length) {
			cs.inflow.take(int32(f.Length))
		} else {
			cc.mu.Unlock()
			return ConnectionError(ErrCodeFlowControl)
		}
		// Return any padded flow control now, since we won't
		// refund it later on body reads.
		var refund int
		if pad := int(f.Length) - len(data); pad > 0 {
			refund += pad
		}
		// Return len(data) now if the stream is already closed,
		// since data will never be read.
		didReset := cs.didReset
		if didReset {
			refund += len(data)
		}
		if refund > 0 {
			cc.inflow.add(int32(refund))
			cc.wmu.Lock()
			cc.fr.WriteWindowUpdate(0, uint32(refund))
			if !didReset {
				cs.inflow.add(int32(refund))
				cc.fr.WriteWindowUpdate(cs.ID, uint32(refund))
			}
			cc.bw.Flush()
			cc.wmu.Unlock()
		}
		cc.mu.Unlock()

		if len(data) > 0 && !didReset {
			if _, err := cs.bufPipe.Write(data); err != nil {
				rl.endStreamError(cs, err)
				return err
			}
		}
	}

	if f.StreamEnded() {
		rl.endStream(cs)
	}
	return nil
}

func (rl *clientConnReadLoop) endStream(cs *clientStream) {
	// TODO: check that any declared content-length matches, like
	// server.go's (*stream).endStream method.
	rl.endStreamError(cs, nil)
}

func (rl *clientConnReadLoop) endStreamError(cs *clientStream, err error) {
	var code func()
	if err == nil {
		err = io.EOF
		code = cs.copyTrailers
	}
	if isConnectionCloseRequest(cs.req) {
		rl.closeWhenIdle = true
	}
	cs.bufPipe.closeWithErrorAndCode(err, code)

	select {
	case cs.resc <- resAndError{err: err}:
	default:
	}
}

func (cs *clientStream) copyTrailers() {
	for k, vv := range cs.trailer {
		t := cs.resTrailer
		if *t == nil {
			*t = make(http.Header)
		}
		(*t)[k] = vv
	}
}

func (rl *clientConnReadLoop) processGoAway(f *GoAwayFrame) error {
	cc := rl.cc
	cc.t.connPool().MarkDead(cc)
	if f.ErrCode != 0 {
		// TODO: deal with GOAWAY more. particularly the error code
		cc.vlogf("transport got GOAWAY with error code = %v", f.ErrCode)
	}
	cc.setGoAway(f)
	return nil
}

func (rl *clientConnReadLoop) processSettings(f *SettingsFrame) error {
	cc := rl.cc
	cc.mu.Lock()
	defer cc.mu.Unlock()

	if f.IsAck() {
		if cc.wantSettingsAck {
			cc.wantSettingsAck = false
			return nil
		}
		return ConnectionError(ErrCodeProtocol)
	}

	err := f.ForeachSetting(func(s Setting) error {
		switch s.ID {
		case SettingMaxFrameSize:
			cc.maxFrameSize = s.Val
		case SettingMaxConcurrentStreams:
			cc.maxConcurrentStreams = s.Val
		case SettingMaxHeaderListSize:
			cc.peerMaxHeaderListSize = uint64(s.Val)
		case SettingInitialWindowSize:
			// Values above the maximum flow-control
			// window size of 2^31-1 MUST be treated as a
			// connection error (Section 5.4.1) of type
			// FLOW_CONTROL_ERROR.
			if s.Val > math.MaxInt32 {
				return ConnectionError(ErrCodeFlowControl)
			}

			// Adjust flow control of currently-open
			// frames by the difference of the old initial
			// window size and this one.
			delta := int32(s.Val) - int32(cc.initialWindowSize)
			for _, cs := range cc.streams {
				cs.flow.add(delta)
			}
			cc.cond.Broadcast()

			cc.initialWindowSize = s.Val
		default:
			// TODO(bradfitz): handle more settings? SETTINGS_HEADER_TABLE_SIZE probably.
			cc.vlogf("Unhandled Setting: %v", s)
		}
		return nil
	})
	if err != nil {
		return err
	}

	cc.wmu.Lock()
	defer cc.wmu.Unlock()

	cc.fr.WriteSettingsAck()
	cc.bw.Flush()
	return cc.werr
}

func (rl *clientConnReadLoop) processWindowUpdate(f *WindowUpdateFrame) error {
	cc := rl.cc
	cs := cc.streamByID(f.StreamID, false)
	if f.StreamID != 0 && cs == nil {
		return nil
	}

	cc.mu.Lock()
	defer cc.mu.Unlock()

	fl := &cc.flow
	if cs != nil {
		fl = &cs.flow
	}
	if !fl.add(int32(f.Increment)) {
		return ConnectionError(ErrCodeFlowControl)
	}
	cc.cond.Broadcast()
	return nil
}

func (rl *clientConnReadLoop) processResetStream(f *RSTStreamFrame) error {
	cs := rl.cc.streamByID(f.StreamID, true)
	if cs == nil {
		// TODO: return error if server tries to RST_STEAM an idle stream
		return nil
	}
	select {
	case <-cs.peerReset:
		// Already reset.
		// This is the only goroutine
		// which closes this, so there
		// isn't a race.
	default:
		err := streamError(cs.ID, f.ErrCode)
		cs.resetErr = err
		close(cs.peerReset)
		cs.bufPipe.CloseWithError(err)
		cs.cc.cond.Broadcast() // wake up checkResetOrDone via clientStream.awaitFlowControl
	}
	return nil
}

// Ping sends a PING frame to the server and waits for the ack.
func (cc *ClientConn) Ping(ctx context.Context) error {
	c := make(chan struct{})
	// Generate a random payload
	var p [8]byte
	for {
		if _, err := rand.Read(p[:]); err != nil {
			return err
		}
		cc.mu.Lock()
		// check for dup before insert
		if _, found := cc.pings[p]; !found {
			cc.pings[p] = c
			cc.mu.Unlock()
			break
		}
		cc.mu.Unlock()
	}
	cc.wmu.Lock()
	if err := cc.fr.WritePing(false, p); err != nil {
		cc.wmu.Unlock()
		return err
	}
	if err := cc.bw.Flush(); err != nil {
		cc.wmu.Unlock()
		return err
	}
	cc.wmu.Unlock()
	select {
	case <-c:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-cc.readerDone:
		// connection closed
		return cc.readerErr
	}
}

func (rl *clientConnReadLoop) processPing(f *PingFrame) error {
	if f.IsAck() {
		cc := rl.cc
		cc.mu.Lock()
		defer cc.mu.Unlock()
		// If ack, notify listener if any
		if c, ok := cc.pings[f.Data]; ok {
			close(c)
			delete(cc.pings, f.Data)
		}
		return nil
	}
	cc := rl.cc
	cc.wmu.Lock()
	defer cc.wmu.Unlock()
	if err := cc.fr.WritePing(true, f.Data); err != nil {
		return err
	}
	return cc.bw.Flush()
}

func (rl *clientConnReadLoop) processPushPromise(f *PushPromiseFrame) error {
	// We told the peer we don't want them.
	// Spec says:
	// "PUSH_PROMISE MUST NOT be sent if the SETTINGS_ENABLE_PUSH
	// setting of the peer endpoint is set to 0. An endpoint that
	// has set this setting and has received acknowledgement MUST
	// treat the receipt of a PUSH_PROMISE frame as a connection
	// error (Section 5.4.1) of type PROTOCOL_ERROR."
	return ConnectionError(ErrCodeProtocol)
}

func (cc *ClientConn) writeStreamReset(streamID uint32, code ErrCode, err error) {
	// TODO: map err to more interesting error codes, once the
	// HTTP community comes up with some. But currently for
	// RST_STREAM there's no equivalent to GOAWAY frame's debug
	// data, and the error codes are all pretty vague ("cancel").
	cc.wmu.Lock()
	cc.fr.WriteRSTStream(streamID, code)
	cc.bw.Flush()
	cc.wmu.Unlock()
}

var (
	errResponseHeaderListSize = errors.New("http2: response header list larger than advertised limit")
	errRequestHeaderListSize  = errors.New("http2: request header list larger than peer's advertised limit")
)

func (cc *ClientConn) logf(format string, args ...interface{}) {
	cc.t.logf(format, args...)
}

func (cc *ClientConn) vlogf(format string, args ...interface{}) {
	cc.t.vlogf(format, args...)
}

func (t *Transport) vlogf(format string, args ...interface{}) {
	if VerboseLogs {
		t.logf(format, args...)
	}
}

func (t *Transport) logf(format string, args ...interface{}) {
	log.Printf(format, args...)
}

var noBody io.ReadCloser = ioutil.NopCloser(bytes.NewReader(nil))

func strSliceContains(ss []string, s string) bool {
	for _, v := range ss {
		if v == s {
			return true
		}
	}
	return false
}

type erringRoundTripper struct{ err error }

func (rt erringRoundTripper) RoundTripErr() error                             { return rt.err }
func (rt erringRoundTripper) RoundTrip(*http.Request) (*http.Response, error) { return nil, rt.err }

// gzipReader wraps a response body so it can lazily
// call gzip.NewReader on the first call to Read
type gzipReader struct {
	_    incomparable
	body io.ReadCloser // underlying Response.Body
	zr   *gzip.Reader  // lazily-initialized gzip reader
	zerr error         // sticky error
}

func (gz *gzipReader) Read(p []byte) (n int, err error) {
	if gz.zerr != nil {
		return 0, gz.zerr
	}
	if gz.zr == nil {
		gz.zr, err = gzip.NewReader(gz.body)
		if err != nil {
			gz.zerr = err
			return 0, err
		}
	}
	return gz.zr.Read(p)
}

func (gz *gzipReader) Close() error {
	return gz.body.Close()
}

type errorReader struct{ err error }

func (r errorReader) Read(p []byte) (int, error) { return 0, r.err }

// bodyWriterState encapsulates various state around the Transport's writing
// of the request body, particularly regarding doing delayed writes of the body
// when the request contains "Expect: 100-continue".
type bodyWriterState struct {
	cs     *clientStream
	timer  *time.Timer   // if non-nil, we're doing a delayed write
	fnonce *sync.Once    // to call fn with
	fn     func()        // the code to run in the goroutine, writing the body
	resc   chan error    // result of fn's execution
	delay  time.Duration // how long we should delay a delayed write for
}

func (t *Transport) getBodyWriterState(cs *clientStream, body io.Reader) (s bodyWriterState) {
	s.cs = cs
	if body == nil {
		return
	}
	resc := make(chan error, 1)
	s.resc = resc
	s.fn = func() {
		cs.cc.mu.Lock()
		cs.startedWrite = true
		cs.cc.mu.Unlock()
		resc <- cs.writeRequestBody(body, cs.req.Body)
	}
	s.delay = t.expectContinueTimeout()
	if s.delay == 0 ||
		!httpguts.HeaderValuesContainsToken(
			cs.req.Header["Expect"],
			"100-continue") {
		return
	}
	s.fnonce = new(sync.Once)

	// Arm the timer with a very large duration, which we'll
	// intentionally lower later. It has to be large now because
	// we need a handle to it before writing the headers, but the
	// s.delay value is defined to not start until after the
	// request headers were written.
	const hugeDuration = 365 * 24 * time.Hour
	s.timer = time.AfterFunc(hugeDuration, func() {
		s.fnonce.Do(s.fn)
	})
	return
}

func (s bodyWriterState) cancel() {
	if s.timer != nil {
		if s.timer.Stop() {
			s.resc <- nil
		}
	}
}

func (s bodyWriterState) on100() {
	if s.timer == nil {
		// If we didn't do a delayed write, ignore the server's
		// bogus 100 continue response.
		return
	}
	s.timer.Stop()
	go func() { s.fnonce.Do(s.fn) }()
}

// scheduleBodyWrite starts writing the body, either immediately (in
// the common case) or after the delay timeout. It should not be
// called until after the headers have been written.
func (s bodyWriterState) scheduleBodyWrite() {
	if s.timer == nil {
		// We're not doing a delayed write (see
		// getBodyWriterState), so just start the writing
		// goroutine immediately.
		go s.fn()
		return
	}
	traceWait100Continue(s.cs.trace)
	if s.timer.Stop() {
		s.timer.Reset(s.delay)
	}
}

// isConnectionCloseRequest reports whether req should use its own
// connection for a single request and then close the connection.
func isConnectionCloseRequest(req *http.Request) bool {
	return req.Close || httpguts.HeaderValuesContainsToken(req.Header["Connection"], "close")
}

// registerHTTPSProtocol calls Transport.RegisterProtocol but
// converting panics into errors.
func registerHTTPSProtocol(t *http.Transport, rt noDialH2RoundTripper) (err error) {
	defer func() {
		if e := recover(); e != nil {
			err = fmt.Errorf("%v", e)
		}
	}()
	t.RegisterProtocol("https", rt)
	return nil
}

// noDialH2RoundTripper is a RoundTripper which only tries to complete the request
// if there's already has a cached connection to the host.
// (The field is exported so it can be accessed via reflect from net/http; tested
// by TestNoDialH2RoundTripperType)
type noDialH2RoundTripper struct{ *Transport }

func (rt noDialH2RoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	res, err := rt.Transport.RoundTrip(req)
	if isNoCachedConnError(err) {
		return nil, http.ErrSkipAltProtocol
	}
	return res, err
}

func (t *Transport) idleConnTimeout() time.Duration {
	if t.t1 != nil {
		return t.t1.IdleConnTimeout
	}
	return 0
}

func traceGetConn(req *http.Request, hostPort string) {
	trace := httptrace.ContextClientTrace(req.Context())
	if trace == nil || trace.GetConn == nil {
		return
	}
	trace.GetConn(hostPort)
}

func traceGotConn(req *http.Request, cc *ClientConn, reused bool) {
	trace := httptrace.ContextClientTrace(req.Context())
	if trace == nil || trace.GotConn == nil {
		return
	}
	ci := httptrace.GotConnInfo{Conn: cc.tconn}
	ci.Reused = reused
	cc.mu.Lock()
	ci.WasIdle = len(cc.streams) == 0 && reused
	if ci.WasIdle && !cc.lastActive.IsZero() {
		ci.IdleTime = time.Now().Sub(cc.lastActive)
	}
	cc.mu.Unlock()

	trace.GotConn(ci)
}

func traceWroteHeaders(trace *httptrace.ClientTrace) {
	if trace != nil && trace.WroteHeaders != nil {
		trace.WroteHeaders()
	}
}

func traceGot100Continue(trace *httptrace.ClientTrace) {
	if trace != nil && trace.Got100Continue != nil {
		trace.Got100Continue()
	}
}

func traceWait100Continue(trace *httptrace.ClientTrace) {
	if trace != nil && trace.Wait100Continue != nil {
		trace.Wait100Continue()
	}
}

func traceWroteRequest(trace *httptrace.ClientTrace, err error) {
	if trace != nil && trace.WroteRequest != nil {
		trace.WroteRequest(httptrace.WroteRequestInfo{Err: err})
	}
}

func traceFirstResponseByte(trace *httptrace.ClientTrace) {
	if trace != nil && trace.GotFirstResponseByte != nil {
		trace.GotFirstResponseByte()
	}
}
