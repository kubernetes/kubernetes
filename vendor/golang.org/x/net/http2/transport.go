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
	"io/fs"
	"log"
	"math"
	"math/bits"
	mathrand "math/rand"
	"net"
	"net/http"
	"net/http/httptrace"
	"net/textproto"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/net/http/httpguts"
	"golang.org/x/net/http2/hpack"
	"golang.org/x/net/idna"
	"golang.org/x/net/internal/httpcommon"
)

const (
	// transportDefaultConnFlow is how many connection-level flow control
	// tokens we give the server at start-up, past the default 64k.
	transportDefaultConnFlow = 1 << 30

	// transportDefaultStreamFlow is how many stream-level flow
	// control tokens we announce to the peer, and how many bytes
	// we buffer per stream.
	transportDefaultStreamFlow = 4 << 20

	defaultUserAgent = "Go-http-client/2.0"

	// initialMaxConcurrentStreams is a connections maxConcurrentStreams until
	// it's received servers initial SETTINGS frame, which corresponds with the
	// spec's minimum recommended value.
	initialMaxConcurrentStreams = 100

	// defaultMaxConcurrentStreams is a connections default maxConcurrentStreams
	// if the server doesn't include one in its initial SETTINGS frame.
	defaultMaxConcurrentStreams = 1000
)

// Transport is an HTTP/2 Transport.
//
// A Transport internally caches connections to servers. It is safe
// for concurrent use by multiple goroutines.
type Transport struct {
	// DialTLSContext specifies an optional dial function with context for
	// creating TLS connections for requests.
	//
	// If DialTLSContext and DialTLS is nil, tls.Dial is used.
	//
	// If the returned net.Conn has a ConnectionState method like tls.Conn,
	// it will be used to set http.Response.TLS.
	DialTLSContext func(ctx context.Context, network, addr string, cfg *tls.Config) (net.Conn, error)

	// DialTLS specifies an optional dial function for creating
	// TLS connections for requests.
	//
	// If DialTLSContext and DialTLS is nil, tls.Dial is used.
	//
	// Deprecated: Use DialTLSContext instead, which allows the transport
	// to cancel dials as soon as they are no longer needed.
	// If both are set, DialTLSContext takes priority.
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

	// MaxReadFrameSize is the http2 SETTINGS_MAX_FRAME_SIZE to send in the
	// initial settings frame. It is the size in bytes of the largest frame
	// payload that the sender is willing to receive. If 0, no setting is
	// sent, and the value is provided by the peer, which should be 16384
	// according to the spec:
	// https://datatracker.ietf.org/doc/html/rfc7540#section-6.5.2.
	// Values are bounded in the range 16k to 16M.
	MaxReadFrameSize uint32

	// MaxDecoderHeaderTableSize optionally specifies the http2
	// SETTINGS_HEADER_TABLE_SIZE to send in the initial settings frame. It
	// informs the remote endpoint of the maximum size of the header compression
	// table used to decode header blocks, in octets. If zero, the default value
	// of 4096 is used.
	MaxDecoderHeaderTableSize uint32

	// MaxEncoderHeaderTableSize optionally specifies an upper limit for the
	// header compression table used for encoding request headers. Received
	// SETTINGS_HEADER_TABLE_SIZE settings are capped at this limit. If zero,
	// the default value of 4096 is used.
	MaxEncoderHeaderTableSize uint32

	// StrictMaxConcurrentStreams controls whether the server's
	// SETTINGS_MAX_CONCURRENT_STREAMS should be respected
	// globally. If false, new TCP connections are created to the
	// server as needed to keep each under the per-connection
	// SETTINGS_MAX_CONCURRENT_STREAMS limit. If true, the
	// server's SETTINGS_MAX_CONCURRENT_STREAMS is interpreted as
	// a global limit and callers of RoundTrip block when needed,
	// waiting for their turn.
	StrictMaxConcurrentStreams bool

	// IdleConnTimeout is the maximum amount of time an idle
	// (keep-alive) connection will remain idle before closing
	// itself.
	// Zero means no limit.
	IdleConnTimeout time.Duration

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

	// WriteByteTimeout is the timeout after which the connection will be
	// closed no data can be written to it. The timeout begins when data is
	// available to write, and is extended whenever any bytes are written.
	WriteByteTimeout time.Duration

	// CountError, if non-nil, is called on HTTP/2 transport errors.
	// It's intended to increment a metric for monitoring, such
	// as an expvar or Prometheus metric.
	// The errType consists of only ASCII word characters.
	CountError func(errType string)

	// t1, if non-nil, is the standard library Transport using
	// this transport. Its settings are used (but not its
	// RoundTrip method, etc).
	t1 *http.Transport

	connPoolOnce  sync.Once
	connPoolOrDef ClientConnPool // non-nil version of ConnPool

	*transportTestHooks
}

// Hook points used for testing.
// Outside of tests, t.transportTestHooks is nil and these all have minimal implementations.
// Inside tests, see the testSyncHooks function docs.

type transportTestHooks struct {
	newclientconn func(*ClientConn)
	group         synctestGroupInterface
}

func (t *Transport) markNewGoroutine() {
	if t != nil && t.transportTestHooks != nil {
		t.transportTestHooks.group.Join()
	}
}

func (t *Transport) now() time.Time {
	if t != nil && t.transportTestHooks != nil {
		return t.transportTestHooks.group.Now()
	}
	return time.Now()
}

func (t *Transport) timeSince(when time.Time) time.Duration {
	if t != nil && t.transportTestHooks != nil {
		return t.now().Sub(when)
	}
	return time.Since(when)
}

// newTimer creates a new time.Timer, or a synthetic timer in tests.
func (t *Transport) newTimer(d time.Duration) timer {
	if t.transportTestHooks != nil {
		return t.transportTestHooks.group.NewTimer(d)
	}
	return timeTimer{time.NewTimer(d)}
}

// afterFunc creates a new time.AfterFunc timer, or a synthetic timer in tests.
func (t *Transport) afterFunc(d time.Duration, f func()) timer {
	if t.transportTestHooks != nil {
		return t.transportTestHooks.group.AfterFunc(d, f)
	}
	return timeTimer{time.AfterFunc(d, f)}
}

func (t *Transport) contextWithTimeout(ctx context.Context, d time.Duration) (context.Context, context.CancelFunc) {
	if t.transportTestHooks != nil {
		return t.transportTestHooks.group.ContextWithTimeout(ctx, d)
	}
	return context.WithTimeout(ctx, d)
}

func (t *Transport) maxHeaderListSize() uint32 {
	n := int64(t.MaxHeaderListSize)
	if t.t1 != nil && t.t1.MaxResponseHeaderBytes != 0 {
		n = t.t1.MaxResponseHeaderBytes
		if n > 0 {
			n = adjustHTTP1MaxHeaderSize(n)
		}
	}
	if n <= 0 {
		return 10 << 20
	}
	if n >= 0xffffffff {
		return 0
	}
	return uint32(n)
}

func (t *Transport) disableCompression() bool {
	return t.DisableCompression || (t.t1 != nil && t.t1.DisableCompression)
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
	upgradeFn := func(scheme, authority string, c net.Conn) http.RoundTripper {
		addr := authorityAddr(scheme, authority)
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
		if scheme == "http" {
			return (*unencryptedTransport)(t2)
		}
		return t2
	}
	if t1.TLSNextProto == nil {
		t1.TLSNextProto = make(map[string]func(string, *tls.Conn) http.RoundTripper)
	}
	t1.TLSNextProto[NextProtoTLS] = func(authority string, c *tls.Conn) http.RoundTripper {
		return upgradeFn("https", authority, c)
	}
	// The "unencrypted_http2" TLSNextProto key is used to pass off non-TLS HTTP/2 conns.
	t1.TLSNextProto[nextProtoUnencryptedHTTP2] = func(authority string, c *tls.Conn) http.RoundTripper {
		nc, err := unencryptedNetConnFromTLSConn(c)
		if err != nil {
			go c.Close()
			return erringRoundTripper{err}
		}
		return upgradeFn("http", authority, nc)
	}
	return t2, nil
}

// unencryptedTransport is a Transport with a RoundTrip method that
// always permits http:// URLs.
type unencryptedTransport Transport

func (t *unencryptedTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return (*Transport)(t).RoundTripOpt(req, RoundTripOpt{allowHTTP: true})
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
	t             *Transport
	tconn         net.Conn             // usually *tls.Conn, except specialized impls
	tlsState      *tls.ConnectionState // nil only for specialized impls
	atomicReused  uint32               // whether conn is being reused; atomic
	singleUse     bool                 // whether being used for a single http.Request
	getConnCalled bool                 // used by clientConnPool

	// readLoop goroutine fields:
	readerDone chan struct{} // closed on error
	readerErr  error         // set before readerDone is closed

	idleTimeout time.Duration // or 0 for never
	idleTimer   timer

	mu               sync.Mutex // guards following
	cond             *sync.Cond // hold mu; broadcast on flow/closed changes
	flow             outflow    // our conn-level flow control quota (cs.outflow is per stream)
	inflow           inflow     // peer's conn-level flow control
	doNotReuse       bool       // whether conn is marked to not be reused for any future requests
	closing          bool
	closed           bool
	closedOnIdle     bool                     // true if conn was closed for idleness
	seenSettings     bool                     // true if we've seen a settings frame, false otherwise
	seenSettingsChan chan struct{}            // closed when seenSettings is true or frame reading fails
	wantSettingsAck  bool                     // we sent a SETTINGS frame and haven't heard back
	goAway           *GoAwayFrame             // if non-nil, the GoAwayFrame we received
	goAwayDebug      string                   // goAway frame's debug data, retained as a string
	streams          map[uint32]*clientStream // client-initiated
	streamsReserved  int                      // incr by ReserveNewRequest; decr on RoundTrip
	nextStreamID     uint32
	pendingRequests  int                       // requests blocked and waiting to be sent because len(streams) == maxConcurrentStreams
	pings            map[[8]byte]chan struct{} // in flight ping data to notification channel
	br               *bufio.Reader
	lastActive       time.Time
	lastIdle         time.Time // time last idle
	// Settings from peer: (also guarded by wmu)
	maxFrameSize                uint32
	maxConcurrentStreams        uint32
	peerMaxHeaderListSize       uint64
	peerMaxHeaderTableSize      uint32
	initialWindowSize           uint32
	initialStreamRecvWindowSize int32
	readIdleTimeout             time.Duration
	pingTimeout                 time.Duration
	extendedConnectAllowed      bool

	// rstStreamPingsBlocked works around an unfortunate gRPC behavior.
	// gRPC strictly limits the number of PING frames that it will receive.
	// The default is two pings per two hours, but the limit resets every time
	// the gRPC endpoint sends a HEADERS or DATA frame. See golang/go#70575.
	//
	// rstStreamPingsBlocked is set after receiving a response to a PING frame
	// bundled with an RST_STREAM (see pendingResets below), and cleared after
	// receiving a HEADERS or DATA frame.
	rstStreamPingsBlocked bool

	// pendingResets is the number of RST_STREAM frames we have sent to the peer,
	// without confirming that the peer has received them. When we send a RST_STREAM,
	// we bundle it with a PING frame, unless a PING is already in flight. We count
	// the reset stream against the connection's concurrency limit until we get
	// a PING response. This limits the number of requests we'll try to send to a
	// completely unresponsive connection.
	pendingResets int

	// reqHeaderMu is a 1-element semaphore channel controlling access to sending new requests.
	// Write to reqHeaderMu to lock it, read from it to unlock.
	// Lock reqmu BEFORE mu or wmu.
	reqHeaderMu chan struct{}

	// wmu is held while writing.
	// Acquire BEFORE mu when holding both, to avoid blocking mu on network writes.
	// Only acquire both at the same time when changing peer settings.
	wmu  sync.Mutex
	bw   *bufio.Writer
	fr   *Framer
	werr error        // first write error that has occurred
	hbuf bytes.Buffer // HPACK encoder writes into this
	henc *hpack.Encoder
}

// clientStream is the state for a single HTTP/2 stream. One of these
// is created for each Transport.RoundTrip call.
type clientStream struct {
	cc *ClientConn

	// Fields of Request that we may access even after the response body is closed.
	ctx       context.Context
	reqCancel <-chan struct{}

	trace         *httptrace.ClientTrace // or nil
	ID            uint32
	bufPipe       pipe // buffered pipe with the flow-controlled response payload
	requestedGzip bool
	isHead        bool

	abortOnce sync.Once
	abort     chan struct{} // closed to signal stream should end immediately
	abortErr  error         // set if abort is closed

	peerClosed chan struct{} // closed when the peer sends an END_STREAM flag
	donec      chan struct{} // closed after the stream is in the closed state
	on100      chan struct{} // buffered; written to if a 100 is received

	respHeaderRecv chan struct{}  // closed when headers are received
	res            *http.Response // set if respHeaderRecv is closed

	flow        outflow // guarded by cc.mu
	inflow      inflow  // guarded by cc.mu
	bytesRemain int64   // -1 means unknown; owned by transportResponseBody.Read
	readErr     error   // sticky read error; owned by transportResponseBody.Read

	reqBody              io.ReadCloser
	reqBodyContentLength int64         // -1 means unknown
	reqBodyClosed        chan struct{} // guarded by cc.mu; non-nil on Close, closed when done

	// owned by writeRequest:
	sentEndStream bool // sent an END_STREAM flag to the peer
	sentHeaders   bool

	// owned by clientConnReadLoop:
	firstByte       bool  // got the first response byte
	pastHeaders     bool  // got first MetaHeadersFrame (actual headers)
	pastTrailers    bool  // got optional second MetaHeadersFrame (trailers)
	readClosed      bool  // peer sent an END_STREAM flag
	readAborted     bool  // read loop reset the stream
	totalHeaderSize int64 // total size of 1xx headers seen

	trailer    http.Header  // accumulated trailers
	resTrailer *http.Header // client's Response.Trailer
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

func (cs *clientStream) abortStream(err error) {
	cs.cc.mu.Lock()
	defer cs.cc.mu.Unlock()
	cs.abortStreamLocked(err)
}

func (cs *clientStream) abortStreamLocked(err error) {
	cs.abortOnce.Do(func() {
		cs.abortErr = err
		close(cs.abort)
	})
	if cs.reqBody != nil {
		cs.closeReqBodyLocked()
	}
	// TODO(dneil): Clean up tests where cs.cc.cond is nil.
	if cs.cc.cond != nil {
		// Wake up writeRequestBody if it is waiting on flow control.
		cs.cc.cond.Broadcast()
	}
}

func (cs *clientStream) abortRequestBodyWrite() {
	cc := cs.cc
	cc.mu.Lock()
	defer cc.mu.Unlock()
	if cs.reqBody != nil && cs.reqBodyClosed == nil {
		cs.closeReqBodyLocked()
		cc.cond.Broadcast()
	}
}

func (cs *clientStream) closeReqBodyLocked() {
	if cs.reqBodyClosed != nil {
		return
	}
	cs.reqBodyClosed = make(chan struct{})
	reqBodyClosed := cs.reqBodyClosed
	go func() {
		cs.cc.t.markNewGoroutine()
		cs.reqBody.Close()
		close(reqBodyClosed)
	}()
}

type stickyErrWriter struct {
	group   synctestGroupInterface
	conn    net.Conn
	timeout time.Duration
	err     *error
}

func (sew stickyErrWriter) Write(p []byte) (n int, err error) {
	if *sew.err != nil {
		return 0, *sew.err
	}
	n, err = writeWithByteTimeout(sew.group, sew.conn, sew.timeout, p)
	*sew.err = err
	return n, err
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

	allowHTTP bool // allow http:// URLs
}

func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	return t.RoundTripOpt(req, RoundTripOpt{})
}

// authorityAddr returns a given authority (a host/IP, or host:port / ip:port)
// and returns a host:port. The port 443 is added if needed.
func authorityAddr(scheme string, authority string) (addr string) {
	host, port, err := net.SplitHostPort(authority)
	if err != nil { // authority didn't have a port
		host = authority
		port = ""
	}
	if port == "" { // authority's port was empty
		port = "443"
		if scheme == "http" {
			port = "80"
		}
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
	switch req.URL.Scheme {
	case "https":
		// Always okay.
	case "http":
		if !t.AllowHTTP && !opt.allowHTTP {
			return nil, errors.New("http2: unencrypted HTTP/2 not enabled")
		}
	default:
		return nil, errors.New("http2: unsupported scheme")
	}

	addr := authorityAddr(req.URL.Scheme, req.URL.Host)
	for retry := 0; ; retry++ {
		cc, err := t.connPool().GetClientConn(req, addr)
		if err != nil {
			t.vlogf("http2: Transport failed to get client conn for %s: %v", addr, err)
			return nil, err
		}
		reused := !atomic.CompareAndSwapUint32(&cc.atomicReused, 0, 1)
		traceGotConn(req, cc, reused)
		res, err := cc.RoundTrip(req)
		if err != nil && retry <= 6 {
			roundTripErr := err
			if req, err = shouldRetryRequest(req, err); err == nil {
				// After the first retry, do exponential backoff with 10% jitter.
				if retry == 0 {
					t.vlogf("RoundTrip retrying after failure: %v", roundTripErr)
					continue
				}
				backoff := float64(uint(1) << (uint(retry) - 1))
				backoff += backoff * (0.1 * mathrand.Float64())
				d := time.Second * time.Duration(backoff)
				tm := t.newTimer(d)
				select {
				case <-tm.C():
					t.vlogf("RoundTrip retrying after failure: %v", roundTripErr)
					continue
				case <-req.Context().Done():
					tm.Stop()
					err = req.Context().Err()
				}
			}
		}
		if err == errClientConnNotEstablished {
			// This ClientConn was created recently,
			// this is the first request to use it,
			// and the connection is closed and not usable.
			//
			// In this state, cc.idleTimer will remove the conn from the pool
			// when it fires. Stop the timer and remove it here so future requests
			// won't try to use this connection.
			//
			// If the timer has already fired and we're racing it, the redundant
			// call to MarkDead is harmless.
			if cc.idleTimer != nil {
				cc.idleTimer.Stop()
			}
			t.connPool().MarkDead(cc)
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
	errClientConnClosed         = errors.New("http2: client conn is closed")
	errClientConnUnusable       = errors.New("http2: client conn not usable")
	errClientConnNotEstablished = errors.New("http2: client conn could not be established")
	errClientConnGotGoAway      = errors.New("http2: Transport received Server's graceful shutdown GOAWAY")
)

// shouldRetryRequest is called by RoundTrip when a request fails to get
// response headers. It is always called with a non-nil error.
// It returns either a request to retry (either the same request, or a
// modified clone), or an error if the request can't be replayed.
func shouldRetryRequest(req *http.Request, err error) (*http.Request, error) {
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
	// the request directly.
	if err == errClientConnUnusable {
		return req, nil
	}

	return nil, fmt.Errorf("http2: Transport: cannot retry err [%v] after Request.Body was written; define Request.GetBody to avoid this error", err)
}

func canRetryError(err error) bool {
	if err == errClientConnUnusable || err == errClientConnGotGoAway {
		return true
	}
	if se, ok := err.(StreamError); ok {
		if se.Code == ErrCodeProtocol && se.Cause == errFromPeer {
			// See golang/go#47635, golang/go#42777
			return true
		}
		return se.Code == ErrCodeRefusedStream
	}
	return false
}

func (t *Transport) dialClientConn(ctx context.Context, addr string, singleUse bool) (*ClientConn, error) {
	if t.transportTestHooks != nil {
		return t.newClientConn(nil, singleUse)
	}
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, err
	}
	tconn, err := t.dialTLS(ctx, "tcp", addr, t.newTLSConfig(host))
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

func (t *Transport) dialTLS(ctx context.Context, network, addr string, tlsCfg *tls.Config) (net.Conn, error) {
	if t.DialTLSContext != nil {
		return t.DialTLSContext(ctx, network, addr, tlsCfg)
	} else if t.DialTLS != nil {
		return t.DialTLS(network, addr, tlsCfg)
	}

	tlsCn, err := t.dialTLSWithContext(ctx, network, addr, tlsCfg)
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
	conf := configFromTransport(t)
	cc := &ClientConn{
		t:                           t,
		tconn:                       c,
		readerDone:                  make(chan struct{}),
		nextStreamID:                1,
		maxFrameSize:                16 << 10, // spec default
		initialWindowSize:           65535,    // spec default
		initialStreamRecvWindowSize: conf.MaxUploadBufferPerStream,
		maxConcurrentStreams:        initialMaxConcurrentStreams, // "infinite", per spec. Use a smaller value until we have received server settings.
		peerMaxHeaderListSize:       0xffffffffffffffff,          // "infinite", per spec. Use 2^64-1 instead.
		streams:                     make(map[uint32]*clientStream),
		singleUse:                   singleUse,
		seenSettingsChan:            make(chan struct{}),
		wantSettingsAck:             true,
		readIdleTimeout:             conf.SendPingTimeout,
		pingTimeout:                 conf.PingTimeout,
		pings:                       make(map[[8]byte]chan struct{}),
		reqHeaderMu:                 make(chan struct{}, 1),
		lastActive:                  t.now(),
	}
	var group synctestGroupInterface
	if t.transportTestHooks != nil {
		t.markNewGoroutine()
		t.transportTestHooks.newclientconn(cc)
		c = cc.tconn
		group = t.group
	}
	if VerboseLogs {
		t.vlogf("http2: Transport creating client conn %p to %v", cc, c.RemoteAddr())
	}

	cc.cond = sync.NewCond(&cc.mu)
	cc.flow.add(int32(initialWindowSize))

	// TODO: adjust this writer size to account for frame size +
	// MTU + crypto/tls record padding.
	cc.bw = bufio.NewWriter(stickyErrWriter{
		group:   group,
		conn:    c,
		timeout: conf.WriteByteTimeout,
		err:     &cc.werr,
	})
	cc.br = bufio.NewReader(c)
	cc.fr = NewFramer(cc.bw, cc.br)
	cc.fr.SetMaxReadFrameSize(conf.MaxReadFrameSize)
	if t.CountError != nil {
		cc.fr.countError = t.CountError
	}
	maxHeaderTableSize := conf.MaxDecoderHeaderTableSize
	cc.fr.ReadMetaHeaders = hpack.NewDecoder(maxHeaderTableSize, nil)
	cc.fr.MaxHeaderListSize = t.maxHeaderListSize()

	cc.henc = hpack.NewEncoder(&cc.hbuf)
	cc.henc.SetMaxDynamicTableSizeLimit(conf.MaxEncoderHeaderTableSize)
	cc.peerMaxHeaderTableSize = initialHeaderTableSize

	if cs, ok := c.(connectionStater); ok {
		state := cs.ConnectionState()
		cc.tlsState = &state
	}

	initialSettings := []Setting{
		{ID: SettingEnablePush, Val: 0},
		{ID: SettingInitialWindowSize, Val: uint32(cc.initialStreamRecvWindowSize)},
	}
	initialSettings = append(initialSettings, Setting{ID: SettingMaxFrameSize, Val: conf.MaxReadFrameSize})
	if max := t.maxHeaderListSize(); max != 0 {
		initialSettings = append(initialSettings, Setting{ID: SettingMaxHeaderListSize, Val: max})
	}
	if maxHeaderTableSize != initialHeaderTableSize {
		initialSettings = append(initialSettings, Setting{ID: SettingHeaderTableSize, Val: maxHeaderTableSize})
	}

	cc.bw.Write(clientPreface)
	cc.fr.WriteSettings(initialSettings...)
	cc.fr.WriteWindowUpdate(0, uint32(conf.MaxUploadBufferPerConnection))
	cc.inflow.init(conf.MaxUploadBufferPerConnection + initialWindowSize)
	cc.bw.Flush()
	if cc.werr != nil {
		cc.Close()
		return nil, cc.werr
	}

	// Start the idle timer after the connection is fully initialized.
	if d := t.idleConnTimeout(); d != 0 {
		cc.idleTimeout = d
		cc.idleTimer = t.afterFunc(d, cc.onIdleTimeout)
	}

	go cc.readLoop()
	return cc, nil
}

func (cc *ClientConn) healthCheck() {
	pingTimeout := cc.pingTimeout
	// We don't need to periodically ping in the health check, because the readLoop of ClientConn will
	// trigger the healthCheck again if there is no frame received.
	ctx, cancel := cc.t.contextWithTimeout(context.Background(), pingTimeout)
	defer cancel()
	cc.vlogf("http2: Transport sending health check")
	err := cc.Ping(ctx)
	if err != nil {
		cc.vlogf("http2: Transport health check failure: %v", err)
		cc.closeForLostPing()
	} else {
		cc.vlogf("http2: Transport health check success")
	}
}

// SetDoNotReuse marks cc as not reusable for future HTTP requests.
func (cc *ClientConn) SetDoNotReuse() {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.doNotReuse = true
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
		if streamID <= last {
			// The server's GOAWAY indicates that it received this stream.
			// It will either finish processing it, or close the connection
			// without doing so. Either way, leave the stream alone for now.
			continue
		}
		if streamID == 1 && cc.goAway.ErrCode != ErrCodeNo {
			// Don't retry the first stream on a connection if we get a non-NO error.
			// If the server is sending an error on a new connection,
			// retrying the request on a new one probably isn't going to work.
			cs.abortStreamLocked(fmt.Errorf("http2: Transport received GOAWAY from server ErrCode:%v", cc.goAway.ErrCode))
		} else {
			// Aborting the stream with errClentConnGotGoAway indicates that
			// the request should be retried on a new connection.
			cs.abortStreamLocked(errClientConnGotGoAway)
		}
	}
}

// CanTakeNewRequest reports whether the connection can take a new request,
// meaning it has not been closed or received or sent a GOAWAY.
//
// If the caller is going to immediately make a new request on this
// connection, use ReserveNewRequest instead.
func (cc *ClientConn) CanTakeNewRequest() bool {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.canTakeNewRequestLocked()
}

// ReserveNewRequest is like CanTakeNewRequest but also reserves a
// concurrent stream in cc. The reservation is decremented on the
// next call to RoundTrip.
func (cc *ClientConn) ReserveNewRequest() bool {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	if st := cc.idleStateLocked(); !st.canTakeNewRequest {
		return false
	}
	cc.streamsReserved++
	return true
}

// ClientConnState describes the state of a ClientConn.
type ClientConnState struct {
	// Closed is whether the connection is closed.
	Closed bool

	// Closing is whether the connection is in the process of
	// closing. It may be closing due to shutdown, being a
	// single-use connection, being marked as DoNotReuse, or
	// having received a GOAWAY frame.
	Closing bool

	// StreamsActive is how many streams are active.
	StreamsActive int

	// StreamsReserved is how many streams have been reserved via
	// ClientConn.ReserveNewRequest.
	StreamsReserved int

	// StreamsPending is how many requests have been sent in excess
	// of the peer's advertised MaxConcurrentStreams setting and
	// are waiting for other streams to complete.
	StreamsPending int

	// MaxConcurrentStreams is how many concurrent streams the
	// peer advertised as acceptable. Zero means no SETTINGS
	// frame has been received yet.
	MaxConcurrentStreams uint32

	// LastIdle, if non-zero, is when the connection last
	// transitioned to idle state.
	LastIdle time.Time
}

// State returns a snapshot of cc's state.
func (cc *ClientConn) State() ClientConnState {
	cc.wmu.Lock()
	maxConcurrent := cc.maxConcurrentStreams
	if !cc.seenSettings {
		maxConcurrent = 0
	}
	cc.wmu.Unlock()

	cc.mu.Lock()
	defer cc.mu.Unlock()
	return ClientConnState{
		Closed:               cc.closed,
		Closing:              cc.closing || cc.singleUse || cc.doNotReuse || cc.goAway != nil,
		StreamsActive:        len(cc.streams) + cc.pendingResets,
		StreamsReserved:      cc.streamsReserved,
		StreamsPending:       cc.pendingRequests,
		LastIdle:             cc.lastIdle,
		MaxConcurrentStreams: maxConcurrent,
	}
}

// clientConnIdleState describes the suitability of a client
// connection to initiate a new RoundTrip request.
type clientConnIdleState struct {
	canTakeNewRequest bool
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
		// We can take a new request if the total of
		//   - active streams;
		//   - reservation slots for new streams; and
		//   - streams for which we have sent a RST_STREAM and a PING,
		//     but received no subsequent frame
		// is less than the concurrency limit.
		maxConcurrentOkay = cc.currentRequestCountLocked() < int(cc.maxConcurrentStreams)
	}

	st.canTakeNewRequest = cc.goAway == nil && !cc.closed && !cc.closing && maxConcurrentOkay &&
		!cc.doNotReuse &&
		int64(cc.nextStreamID)+2*int64(cc.pendingRequests) < math.MaxInt32 &&
		!cc.tooIdleLocked()

	// If this connection has never been used for a request and is closed,
	// then let it take a request (which will fail).
	// If the conn was closed for idleness, we're racing the idle timer;
	// don't try to use the conn. (Issue #70515.)
	//
	// This avoids a situation where an error early in a connection's lifetime
	// goes unreported.
	if cc.nextStreamID == 1 && cc.streamsReserved == 0 && cc.closed && !cc.closedOnIdle {
		st.canTakeNewRequest = true
	}

	return
}

// currentRequestCountLocked reports the number of concurrency slots currently in use,
// including active streams, reserved slots, and reset streams waiting for acknowledgement.
func (cc *ClientConn) currentRequestCountLocked() int {
	return len(cc.streams) + cc.streamsReserved + cc.pendingResets
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
	return cc.idleTimeout != 0 && !cc.lastIdle.IsZero() && cc.t.timeSince(cc.lastIdle.Round(0)) > cc.idleTimeout
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

func (cc *ClientConn) closeConn() {
	t := time.AfterFunc(250*time.Millisecond, cc.forceCloseConn)
	defer t.Stop()
	cc.tconn.Close()
}

// A tls.Conn.Close can hang for a long time if the peer is unresponsive.
// Try to shut it down more aggressively.
func (cc *ClientConn) forceCloseConn() {
	tc, ok := cc.tconn.(*tls.Conn)
	if !ok {
		return
	}
	if nc := tc.NetConn(); nc != nil {
		nc.Close()
	}
}

func (cc *ClientConn) closeIfIdle() {
	cc.mu.Lock()
	if len(cc.streams) > 0 || cc.streamsReserved > 0 {
		cc.mu.Unlock()
		return
	}
	cc.closed = true
	cc.closedOnIdle = true
	nextID := cc.nextStreamID
	// TODO: do clients send GOAWAY too? maybe? Just Close:
	cc.mu.Unlock()

	if VerboseLogs {
		cc.vlogf("http2: Transport closing idle conn %p (forSingleUse=%v, maxStream=%v)", cc, cc.singleUse, nextID-2)
	}
	cc.closeConn()
}

func (cc *ClientConn) isDoNotReuseAndIdle() bool {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.doNotReuse && len(cc.streams) == 0
}

var shutdownEnterWaitStateHook = func() {}

// Shutdown gracefully closes the client connection, waiting for running streams to complete.
func (cc *ClientConn) Shutdown(ctx context.Context) error {
	if err := cc.sendGoAway(); err != nil {
		return err
	}
	// Wait for all in-flight streams to complete or connection to close
	done := make(chan struct{})
	cancelled := false // guarded by cc.mu
	go func() {
		cc.t.markNewGoroutine()
		cc.mu.Lock()
		defer cc.mu.Unlock()
		for {
			if len(cc.streams) == 0 || cc.closed {
				cc.closed = true
				close(done)
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
	case <-done:
		cc.closeConn()
		return nil
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
	closing := cc.closing
	cc.closing = true
	maxStreamID := cc.nextStreamID
	cc.mu.Unlock()
	if closing {
		// GOAWAY sent already
		return nil
	}

	cc.wmu.Lock()
	defer cc.wmu.Unlock()
	// Send a graceful shutdown frame to server
	if err := cc.fr.WriteGoAway(maxStreamID, ErrCodeNo, nil); err != nil {
		return err
	}
	if err := cc.bw.Flush(); err != nil {
		return err
	}
	// Prevent new requests
	return nil
}

// closes the client connection immediately. In-flight requests are interrupted.
// err is sent to streams.
func (cc *ClientConn) closeForError(err error) {
	cc.mu.Lock()
	cc.closed = true
	for _, cs := range cc.streams {
		cs.abortStreamLocked(err)
	}
	cc.cond.Broadcast()
	cc.mu.Unlock()
	cc.closeConn()
}

// Close closes the client connection immediately.
//
// In-flight requests are interrupted. For a graceful shutdown, use Shutdown instead.
func (cc *ClientConn) Close() error {
	err := errors.New("http2: client connection force closed via ClientConn.Close")
	cc.closeForError(err)
	return nil
}

// closes the client connection immediately. In-flight requests are interrupted.
func (cc *ClientConn) closeForLostPing() {
	err := errors.New("http2: client connection lost")
	if f := cc.t.CountError; f != nil {
		f("conn_close_lost_ping")
	}
	cc.closeForError(err)
}

// errRequestCanceled is a copy of net/http's errRequestCanceled because it's not
// exported. At least they'll be DeepEqual for h1-vs-h2 comparisons tests.
var errRequestCanceled = errors.New("net/http: request canceled")

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

func (cc *ClientConn) decrStreamReservations() {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.decrStreamReservationsLocked()
}

func (cc *ClientConn) decrStreamReservationsLocked() {
	if cc.streamsReserved > 0 {
		cc.streamsReserved--
	}
}

func (cc *ClientConn) RoundTrip(req *http.Request) (*http.Response, error) {
	return cc.roundTrip(req, nil)
}

func (cc *ClientConn) roundTrip(req *http.Request, streamf func(*clientStream)) (*http.Response, error) {
	ctx := req.Context()
	cs := &clientStream{
		cc:                   cc,
		ctx:                  ctx,
		reqCancel:            req.Cancel,
		isHead:               req.Method == "HEAD",
		reqBody:              req.Body,
		reqBodyContentLength: actualContentLength(req),
		trace:                httptrace.ContextClientTrace(ctx),
		peerClosed:           make(chan struct{}),
		abort:                make(chan struct{}),
		respHeaderRecv:       make(chan struct{}),
		donec:                make(chan struct{}),
	}

	cs.requestedGzip = httpcommon.IsRequestGzip(req.Method, req.Header, cc.t.disableCompression())

	go cs.doRequest(req, streamf)

	waitDone := func() error {
		select {
		case <-cs.donec:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		case <-cs.reqCancel:
			return errRequestCanceled
		}
	}

	handleResponseHeaders := func() (*http.Response, error) {
		res := cs.res
		if res.StatusCode > 299 {
			// On error or status code 3xx, 4xx, 5xx, etc abort any
			// ongoing write, assuming that the server doesn't care
			// about our request body. If the server replied with 1xx or
			// 2xx, however, then assume the server DOES potentially
			// want our body (e.g. full-duplex streaming:
			// golang.org/issue/13444). If it turns out the server
			// doesn't, they'll RST_STREAM us soon enough. This is a
			// heuristic to avoid adding knobs to Transport. Hopefully
			// we can keep it.
			cs.abortRequestBodyWrite()
		}
		res.Request = req
		res.TLS = cc.tlsState
		if res.Body == noBody && actualContentLength(req) == 0 {
			// If there isn't a request or response body still being
			// written, then wait for the stream to be closed before
			// RoundTrip returns.
			if err := waitDone(); err != nil {
				return nil, err
			}
		}
		return res, nil
	}

	cancelRequest := func(cs *clientStream, err error) error {
		cs.cc.mu.Lock()
		bodyClosed := cs.reqBodyClosed
		cs.cc.mu.Unlock()
		// Wait for the request body to be closed.
		//
		// If nothing closed the body before now, abortStreamLocked
		// will have started a goroutine to close it.
		//
		// Closing the body before returning avoids a race condition
		// with net/http checking its readTrackingBody to see if the
		// body was read from or closed. See golang/go#60041.
		//
		// The body is closed in a separate goroutine without the
		// connection mutex held, but dropping the mutex before waiting
		// will keep us from holding it indefinitely if the body
		// close is slow for some reason.
		if bodyClosed != nil {
			<-bodyClosed
		}
		return err
	}

	for {
		select {
		case <-cs.respHeaderRecv:
			return handleResponseHeaders()
		case <-cs.abort:
			select {
			case <-cs.respHeaderRecv:
				// If both cs.respHeaderRecv and cs.abort are signaling,
				// pick respHeaderRecv. The server probably wrote the
				// response and immediately reset the stream.
				// golang.org/issue/49645
				return handleResponseHeaders()
			default:
				waitDone()
				return nil, cs.abortErr
			}
		case <-ctx.Done():
			err := ctx.Err()
			cs.abortStream(err)
			return nil, cancelRequest(cs, err)
		case <-cs.reqCancel:
			cs.abortStream(errRequestCanceled)
			return nil, cancelRequest(cs, errRequestCanceled)
		}
	}
}

// doRequest runs for the duration of the request lifetime.
//
// It sends the request and performs post-request cleanup (closing Request.Body, etc.).
func (cs *clientStream) doRequest(req *http.Request, streamf func(*clientStream)) {
	cs.cc.t.markNewGoroutine()
	err := cs.writeRequest(req, streamf)
	cs.cleanupWriteRequest(err)
}

var errExtendedConnectNotSupported = errors.New("net/http: extended connect not supported by peer")

// writeRequest sends a request.
//
// It returns nil after the request is written, the response read,
// and the request stream is half-closed by the peer.
//
// It returns non-nil if the request ends otherwise.
// If the returned error is StreamError, the error Code may be used in resetting the stream.
func (cs *clientStream) writeRequest(req *http.Request, streamf func(*clientStream)) (err error) {
	cc := cs.cc
	ctx := cs.ctx

	// wait for setting frames to be received, a server can change this value later,
	// but we just wait for the first settings frame
	var isExtendedConnect bool
	if req.Method == "CONNECT" && req.Header.Get(":protocol") != "" {
		isExtendedConnect = true
	}

	// Acquire the new-request lock by writing to reqHeaderMu.
	// This lock guards the critical section covering allocating a new stream ID
	// (requires mu) and creating the stream (requires wmu).
	if cc.reqHeaderMu == nil {
		panic("RoundTrip on uninitialized ClientConn") // for tests
	}
	if isExtendedConnect {
		select {
		case <-cs.reqCancel:
			return errRequestCanceled
		case <-ctx.Done():
			return ctx.Err()
		case <-cc.seenSettingsChan:
			if !cc.extendedConnectAllowed {
				return errExtendedConnectNotSupported
			}
		}
	}
	select {
	case cc.reqHeaderMu <- struct{}{}:
	case <-cs.reqCancel:
		return errRequestCanceled
	case <-ctx.Done():
		return ctx.Err()
	}

	cc.mu.Lock()
	if cc.idleTimer != nil {
		cc.idleTimer.Stop()
	}
	cc.decrStreamReservationsLocked()
	if err := cc.awaitOpenSlotForStreamLocked(cs); err != nil {
		cc.mu.Unlock()
		<-cc.reqHeaderMu
		return err
	}
	cc.addStreamLocked(cs) // assigns stream ID
	if isConnectionCloseRequest(req) {
		cc.doNotReuse = true
	}
	cc.mu.Unlock()

	if streamf != nil {
		streamf(cs)
	}

	continueTimeout := cc.t.expectContinueTimeout()
	if continueTimeout != 0 {
		if !httpguts.HeaderValuesContainsToken(req.Header["Expect"], "100-continue") {
			continueTimeout = 0
		} else {
			cs.on100 = make(chan struct{}, 1)
		}
	}

	// Past this point (where we send request headers), it is possible for
	// RoundTrip to return successfully. Since the RoundTrip contract permits
	// the caller to "mutate or reuse" the Request after closing the Response's Body,
	// we must take care when referencing the Request from here on.
	err = cs.encodeAndWriteHeaders(req)
	<-cc.reqHeaderMu
	if err != nil {
		return err
	}

	hasBody := cs.reqBodyContentLength != 0
	if !hasBody {
		cs.sentEndStream = true
	} else {
		if continueTimeout != 0 {
			traceWait100Continue(cs.trace)
			timer := time.NewTimer(continueTimeout)
			select {
			case <-timer.C:
				err = nil
			case <-cs.on100:
				err = nil
			case <-cs.abort:
				err = cs.abortErr
			case <-ctx.Done():
				err = ctx.Err()
			case <-cs.reqCancel:
				err = errRequestCanceled
			}
			timer.Stop()
			if err != nil {
				traceWroteRequest(cs.trace, err)
				return err
			}
		}

		if err = cs.writeRequestBody(req); err != nil {
			if err != errStopReqBodyWrite {
				traceWroteRequest(cs.trace, err)
				return err
			}
		} else {
			cs.sentEndStream = true
		}
	}

	traceWroteRequest(cs.trace, err)

	var respHeaderTimer <-chan time.Time
	var respHeaderRecv chan struct{}
	if d := cc.responseHeaderTimeout(); d != 0 {
		timer := cc.t.newTimer(d)
		defer timer.Stop()
		respHeaderTimer = timer.C()
		respHeaderRecv = cs.respHeaderRecv
	}
	// Wait until the peer half-closes its end of the stream,
	// or until the request is aborted (via context, error, or otherwise),
	// whichever comes first.
	for {
		select {
		case <-cs.peerClosed:
			return nil
		case <-respHeaderTimer:
			return errTimeout
		case <-respHeaderRecv:
			respHeaderRecv = nil
			respHeaderTimer = nil // keep waiting for END_STREAM
		case <-cs.abort:
			return cs.abortErr
		case <-ctx.Done():
			return ctx.Err()
		case <-cs.reqCancel:
			return errRequestCanceled
		}
	}
}

func (cs *clientStream) encodeAndWriteHeaders(req *http.Request) error {
	cc := cs.cc
	ctx := cs.ctx

	cc.wmu.Lock()
	defer cc.wmu.Unlock()

	// If the request was canceled while waiting for cc.mu, just quit.
	select {
	case <-cs.abort:
		return cs.abortErr
	case <-ctx.Done():
		return ctx.Err()
	case <-cs.reqCancel:
		return errRequestCanceled
	default:
	}

	// Encode headers.
	//
	// we send: HEADERS{1}, CONTINUATION{0,} + DATA{0,} (DATA is
	// sent by writeRequestBody below, along with any Trailers,
	// again in form HEADERS{1}, CONTINUATION{0,})
	cc.hbuf.Reset()
	res, err := encodeRequestHeaders(req, cs.requestedGzip, cc.peerMaxHeaderListSize, func(name, value string) {
		cc.writeHeader(name, value)
	})
	if err != nil {
		return fmt.Errorf("http2: %w", err)
	}
	hdrs := cc.hbuf.Bytes()

	// Write the request.
	endStream := !res.HasBody && !res.HasTrailers
	cs.sentHeaders = true
	err = cc.writeHeaders(cs.ID, endStream, int(cc.maxFrameSize), hdrs)
	traceWroteHeaders(cs.trace)
	return err
}

func encodeRequestHeaders(req *http.Request, addGzipHeader bool, peerMaxHeaderListSize uint64, headerf func(name, value string)) (httpcommon.EncodeHeadersResult, error) {
	return httpcommon.EncodeHeaders(req.Context(), httpcommon.EncodeHeadersParam{
		Request: httpcommon.Request{
			Header:              req.Header,
			Trailer:             req.Trailer,
			URL:                 req.URL,
			Host:                req.Host,
			Method:              req.Method,
			ActualContentLength: actualContentLength(req),
		},
		AddGzipHeader:         addGzipHeader,
		PeerMaxHeaderListSize: peerMaxHeaderListSize,
		DefaultUserAgent:      defaultUserAgent,
	}, headerf)
}

// cleanupWriteRequest performs post-request tasks.
//
// If err (the result of writeRequest) is non-nil and the stream is not closed,
// cleanupWriteRequest will send a reset to the peer.
func (cs *clientStream) cleanupWriteRequest(err error) {
	cc := cs.cc

	if cs.ID == 0 {
		// We were canceled before creating the stream, so return our reservation.
		cc.decrStreamReservations()
	}

	// TODO: write h12Compare test showing whether
	// Request.Body is closed by the Transport,
	// and in multiple cases: server replies <=299 and >299
	// while still writing request body
	cc.mu.Lock()
	mustCloseBody := false
	if cs.reqBody != nil && cs.reqBodyClosed == nil {
		mustCloseBody = true
		cs.reqBodyClosed = make(chan struct{})
	}
	bodyClosed := cs.reqBodyClosed
	closeOnIdle := cc.singleUse || cc.doNotReuse || cc.t.disableKeepAlives() || cc.goAway != nil
	cc.mu.Unlock()
	if mustCloseBody {
		cs.reqBody.Close()
		close(bodyClosed)
	}
	if bodyClosed != nil {
		<-bodyClosed
	}

	if err != nil && cs.sentEndStream {
		// If the connection is closed immediately after the response is read,
		// we may be aborted before finishing up here. If the stream was closed
		// cleanly on both sides, there is no error.
		select {
		case <-cs.peerClosed:
			err = nil
		default:
		}
	}
	if err != nil {
		cs.abortStream(err) // possibly redundant, but harmless
		if cs.sentHeaders {
			if se, ok := err.(StreamError); ok {
				if se.Cause != errFromPeer {
					cc.writeStreamReset(cs.ID, se.Code, false, err)
				}
			} else {
				// We're cancelling an in-flight request.
				//
				// This could be due to the server becoming unresponsive.
				// To avoid sending too many requests on a dead connection,
				// we let the request continue to consume a concurrency slot
				// until we can confirm the server is still responding.
				// We do this by sending a PING frame along with the RST_STREAM
				// (unless a ping is already in flight).
				//
				// For simplicity, we don't bother tracking the PING payload:
				// We reset cc.pendingResets any time we receive a PING ACK.
				//
				// We skip this if the conn is going to be closed on idle,
				// because it's short lived and will probably be closed before
				// we get the ping response.
				ping := false
				if !closeOnIdle {
					cc.mu.Lock()
					// rstStreamPingsBlocked works around a gRPC behavior:
					// see comment on the field for details.
					if !cc.rstStreamPingsBlocked {
						if cc.pendingResets == 0 {
							ping = true
						}
						cc.pendingResets++
					}
					cc.mu.Unlock()
				}
				cc.writeStreamReset(cs.ID, ErrCodeCancel, ping, err)
			}
		}
		cs.bufPipe.CloseWithError(err) // no-op if already closed
	} else {
		if cs.sentHeaders && !cs.sentEndStream {
			cc.writeStreamReset(cs.ID, ErrCodeNo, false, nil)
		}
		cs.bufPipe.CloseWithError(errRequestCanceled)
	}
	if cs.ID != 0 {
		cc.forgetStreamID(cs.ID)
	}

	cc.wmu.Lock()
	werr := cc.werr
	cc.wmu.Unlock()
	if werr != nil {
		cc.Close()
	}

	close(cs.donec)
}

// awaitOpenSlotForStreamLocked waits until len(streams) < maxConcurrentStreams.
// Must hold cc.mu.
func (cc *ClientConn) awaitOpenSlotForStreamLocked(cs *clientStream) error {
	for {
		if cc.closed && cc.nextStreamID == 1 && cc.streamsReserved == 0 {
			// This is the very first request sent to this connection.
			// Return a fatal error which aborts the retry loop.
			return errClientConnNotEstablished
		}
		cc.lastActive = cc.t.now()
		if cc.closed || !cc.canTakeNewRequestLocked() {
			return errClientConnUnusable
		}
		cc.lastIdle = time.Time{}
		if cc.currentRequestCountLocked() < int(cc.maxConcurrentStreams) {
			return nil
		}
		cc.pendingRequests++
		cc.cond.Wait()
		cc.pendingRequests--
		select {
		case <-cs.abort:
			return cs.abortErr
		default:
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

// frameScratchBufferLen returns the length of a buffer to use for
// outgoing request bodies to read/write to/from.
//
// It returns max(1, min(peer's advertised max frame size,
// Request.ContentLength+1, 512KB)).
func (cs *clientStream) frameScratchBufferLen(maxFrameSize int) int {
	const max = 512 << 10
	n := int64(maxFrameSize)
	if n > max {
		n = max
	}
	if cl := cs.reqBodyContentLength; cl != -1 && cl+1 < n {
		// Add an extra byte past the declared content-length to
		// give the caller's Request.Body io.Reader a chance to
		// give us more bytes than they declared, so we can catch it
		// early.
		n = cl + 1
	}
	if n < 1 {
		return 1
	}
	return int(n) // doesn't truncate; max is 512K
}

// Seven bufPools manage different frame sizes. This helps to avoid scenarios where long-running
// streaming requests using small frame sizes occupy large buffers initially allocated for prior
// requests needing big buffers. The size ranges are as follows:
// {0 KB, 16 KB], {16 KB, 32 KB], {32 KB, 64 KB], {64 KB, 128 KB], {128 KB, 256 KB],
// {256 KB, 512 KB], {512 KB, infinity}
// In practice, the maximum scratch buffer size should not exceed 512 KB due to
// frameScratchBufferLen(maxFrameSize), thus the "infinity pool" should never be used.
// It exists mainly as a safety measure, for potential future increases in max buffer size.
var bufPools [7]sync.Pool // of *[]byte
func bufPoolIndex(size int) int {
	if size <= 16384 {
		return 0
	}
	size -= 1
	bits := bits.Len(uint(size))
	index := bits - 14
	if index >= len(bufPools) {
		return len(bufPools) - 1
	}
	return index
}

func (cs *clientStream) writeRequestBody(req *http.Request) (err error) {
	cc := cs.cc
	body := cs.reqBody
	sentEnd := false // whether we sent the final DATA frame w/ END_STREAM

	hasTrailers := req.Trailer != nil
	remainLen := cs.reqBodyContentLength
	hasContentLen := remainLen != -1

	cc.mu.Lock()
	maxFrameSize := int(cc.maxFrameSize)
	cc.mu.Unlock()

	// Scratch buffer for reading into & writing from.
	scratchLen := cs.frameScratchBufferLen(maxFrameSize)
	var buf []byte
	index := bufPoolIndex(scratchLen)
	if bp, ok := bufPools[index].Get().(*[]byte); ok && len(*bp) >= scratchLen {
		defer bufPools[index].Put(bp)
		buf = *bp
	} else {
		buf = make([]byte, scratchLen)
		defer bufPools[index].Put(&buf)
	}

	var sawEOF bool
	for !sawEOF {
		n, err := body.Read(buf)
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
				var scratch [1]byte
				var n1 int
				n1, err = body.Read(scratch[:])
				remainLen -= int64(n1)
			}
			if remainLen < 0 {
				err = errReqBodyTooLong
				return err
			}
		}
		if err != nil {
			cc.mu.Lock()
			bodyClosed := cs.reqBodyClosed != nil
			cc.mu.Unlock()
			switch {
			case bodyClosed:
				return errStopReqBodyWrite
			case err == io.EOF:
				sawEOF = true
				err = nil
			default:
				return err
			}
		}

		remain := buf[:n]
		for len(remain) > 0 && err == nil {
			var allowed int32
			allowed, err = cs.awaitFlowControl(len(remain))
			if err != nil {
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

	// Since the RoundTrip contract permits the caller to "mutate or reuse"
	// a request after the Response's Body is closed, verify that this hasn't
	// happened before accessing the trailers.
	cc.mu.Lock()
	trailer := req.Trailer
	err = cs.abortErr
	cc.mu.Unlock()
	if err != nil {
		return err
	}

	cc.wmu.Lock()
	defer cc.wmu.Unlock()
	var trls []byte
	if len(trailer) > 0 {
		trls, err = cc.encodeTrailers(trailer)
		if err != nil {
			return err
		}
	}

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
	ctx := cs.ctx
	cc.mu.Lock()
	defer cc.mu.Unlock()
	for {
		if cc.closed {
			return 0, errClientConnClosed
		}
		if cs.reqBodyClosed != nil {
			return 0, errStopReqBodyWrite
		}
		select {
		case <-cs.abort:
			return 0, cs.abortErr
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-cs.reqCancel:
			return 0, errRequestCanceled
		default:
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

// requires cc.wmu be held.
func (cc *ClientConn) encodeTrailers(trailer http.Header) ([]byte, error) {
	cc.hbuf.Reset()

	hlSize := uint64(0)
	for k, vv := range trailer {
		for _, v := range vv {
			hf := hpack.HeaderField{Name: k, Value: v}
			hlSize += uint64(hf.Size())
		}
	}
	if hlSize > cc.peerMaxHeaderListSize {
		return nil, errRequestHeaderListSize
	}

	for k, vv := range trailer {
		lowKey, ascii := httpcommon.LowerHeader(k)
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
func (cc *ClientConn) addStreamLocked(cs *clientStream) {
	cs.flow.add(int32(cc.initialWindowSize))
	cs.flow.setConnFlow(&cc.flow)
	cs.inflow.init(cc.initialStreamRecvWindowSize)
	cs.ID = cc.nextStreamID
	cc.nextStreamID += 2
	cc.streams[cs.ID] = cs
	if cs.ID == 0 {
		panic("assigned stream ID 0")
	}
}

func (cc *ClientConn) forgetStreamID(id uint32) {
	cc.mu.Lock()
	slen := len(cc.streams)
	delete(cc.streams, id)
	if len(cc.streams) != slen-1 {
		panic("forgetting unknown stream id")
	}
	cc.lastActive = cc.t.now()
	if len(cc.streams) == 0 && cc.idleTimer != nil {
		cc.idleTimer.Reset(cc.idleTimeout)
		cc.lastIdle = cc.t.now()
	}
	// Wake up writeRequestBody via clientStream.awaitFlowControl and
	// wake up RoundTrip if there is a pending request.
	cc.cond.Broadcast()

	closeOnIdle := cc.singleUse || cc.doNotReuse || cc.t.disableKeepAlives() || cc.goAway != nil
	if closeOnIdle && cc.streamsReserved == 0 && len(cc.streams) == 0 {
		if VerboseLogs {
			cc.vlogf("http2: Transport closing idle conn %p (forSingleUse=%v, maxStream=%v)", cc, cc.singleUse, cc.nextStreamID-2)
		}
		cc.closed = true
		defer cc.closeConn()
	}

	cc.mu.Unlock()
}

// clientConnReadLoop is the state owned by the clientConn's frame-reading readLoop.
type clientConnReadLoop struct {
	_  incomparable
	cc *ClientConn
}

// readLoop runs in its own goroutine and reads and dispatches frames.
func (cc *ClientConn) readLoop() {
	cc.t.markNewGoroutine()
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
	defer cc.closeConn()
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
	cc.closed = true

	// If the connection has never been used, and has been open for only a short time,
	// leave it in the connection pool for a little while.
	//
	// This avoids a situation where new connections are constantly created,
	// added to the pool, fail, and are removed from the pool, without any error
	// being surfaced to the user.
	unusedWaitTime := 5 * time.Second
	if cc.idleTimeout > 0 && unusedWaitTime > cc.idleTimeout {
		unusedWaitTime = cc.idleTimeout
	}
	idleTime := cc.t.now().Sub(cc.lastActive)
	if atomic.LoadUint32(&cc.atomicReused) == 0 && idleTime < unusedWaitTime && !cc.closedOnIdle {
		cc.idleTimer = cc.t.afterFunc(unusedWaitTime-idleTime, func() {
			cc.t.connPool().MarkDead(cc)
		})
	} else {
		cc.mu.Unlock() // avoid any deadlocks in MarkDead
		cc.t.connPool().MarkDead(cc)
		cc.mu.Lock()
	}

	for _, cs := range cc.streams {
		select {
		case <-cs.peerClosed:
			// The server closed the stream before closing the conn,
			// so no need to interrupt it.
		default:
			cs.abortStreamLocked(err)
		}
	}
	cc.cond.Broadcast()
	cc.mu.Unlock()

	if !cc.seenSettings {
		// If we have a pending request that wants extended CONNECT,
		// let it continue and fail with the connection error.
		cc.extendedConnectAllowed = true
		close(cc.seenSettingsChan)
	}
}

// countReadFrameError calls Transport.CountError with a string
// representing err.
func (cc *ClientConn) countReadFrameError(err error) {
	f := cc.t.CountError
	if f == nil || err == nil {
		return
	}
	if ce, ok := err.(ConnectionError); ok {
		errCode := ErrCode(ce)
		f(fmt.Sprintf("read_frame_conn_error_%s", errCode.stringToken()))
		return
	}
	if errors.Is(err, io.EOF) {
		f("read_frame_eof")
		return
	}
	if errors.Is(err, io.ErrUnexpectedEOF) {
		f("read_frame_unexpected_eof")
		return
	}
	if errors.Is(err, ErrFrameTooLarge) {
		f("read_frame_too_large")
		return
	}
	f("read_frame_other")
}

func (rl *clientConnReadLoop) run() error {
	cc := rl.cc
	gotSettings := false
	readIdleTimeout := cc.readIdleTimeout
	var t timer
	if readIdleTimeout != 0 {
		t = cc.t.afterFunc(readIdleTimeout, cc.healthCheck)
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
			if cs := rl.streamByID(se.StreamID, notHeaderOrDataFrame); cs != nil {
				if se.Cause == nil {
					se.Cause = cc.fr.errDetail
				}
				rl.endStreamError(cs, se)
			}
			continue
		} else if err != nil {
			cc.countReadFrameError(err)
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

		switch f := f.(type) {
		case *MetaHeadersFrame:
			err = rl.processHeaders(f)
		case *DataFrame:
			err = rl.processData(f)
		case *GoAwayFrame:
			err = rl.processGoAway(f)
		case *RSTStreamFrame:
			err = rl.processResetStream(f)
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
	}
}

func (rl *clientConnReadLoop) processHeaders(f *MetaHeadersFrame) error {
	cs := rl.streamByID(f.StreamID, headerOrDataFrame)
	if cs == nil {
		// We'd get here if we canceled a request while the
		// server had its response still in flight. So if this
		// was just something we canceled, ignore it.
		return nil
	}
	if cs.readClosed {
		rl.endStreamError(cs, StreamError{
			StreamID: f.StreamID,
			Code:     ErrCodeProtocol,
			Cause:    errors.New("protocol error: headers after END_STREAM"),
		})
		return nil
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
		rl.endStreamError(cs, StreamError{
			StreamID: f.StreamID,
			Code:     ErrCodeProtocol,
			Cause:    err,
		})
		return nil // return nil from process* funcs to keep conn alive
	}
	if res == nil {
		// (nil, nil) special case. See handleResponse docs.
		return nil
	}
	cs.resTrailer = &res.Trailer
	cs.res = res
	close(cs.respHeaderRecv)
	if f.StreamEnded() {
		rl.endStream(cs)
	}
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
		key := httpcommon.CanonicalHeader(hf.Name)
		if key == "Trailer" {
			t := res.Trailer
			if t == nil {
				t = make(http.Header)
				res.Trailer = t
			}
			foreachHeaderElement(hf.Value, func(v string) {
				t[httpcommon.CanonicalHeader(v)] = nil
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
		if f.StreamEnded() {
			return nil, errors.New("1xx informational response with END_STREAM flag")
		}
		if fn := cs.get1xxTraceFunc(); fn != nil {
			// If the 1xx response is being delivered to the user,
			// then they're responsible for limiting the number
			// of responses.
			if err := fn(statusCode, textproto.MIMEHeader(header)); err != nil {
				return nil, err
			}
		} else {
			// If the user didn't examine the 1xx response, then we
			// limit the size of all 1xx headers.
			//
			// This differs a bit from the HTTP/1 implementation, which
			// limits the size of all 1xx headers plus the final response.
			// Use the larger limit of MaxHeaderListSize and
			// net/http.Transport.MaxResponseHeaderBytes.
			limit := int64(cs.cc.t.maxHeaderListSize())
			if t1 := cs.cc.t.t1; t1 != nil && t1.MaxResponseHeaderBytes > limit {
				limit = t1.MaxResponseHeaderBytes
			}
			for _, h := range f.Fields {
				cs.totalHeaderSize += int64(h.Size())
			}
			if cs.totalHeaderSize > limit {
				if VerboseLogs {
					log.Printf("http2: 1xx informational responses too large")
				}
				return nil, errors.New("header list too large")
			}
		}
		if statusCode == 100 {
			traceGot100Continue(cs.trace)
			select {
			case cs.on100 <- struct{}{}:
			default:
			}
		}
		cs.pastHeaders = false // do it all again
		return nil, nil
	}

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
	} else if f.StreamEnded() && !cs.isHead {
		res.ContentLength = 0
	}

	if cs.isHead {
		res.Body = noBody
		return res, nil
	}

	if f.StreamEnded() {
		if res.ContentLength > 0 {
			res.Body = missingBody{}
		} else {
			res.Body = noBody
		}
		return res, nil
	}

	cs.bufPipe.setBuffer(&dataBuffer{expected: res.ContentLength})
	cs.bytesRemain = res.ContentLength
	res.Body = transportResponseBody{cs}

	if cs.requestedGzip && asciiEqualFold(res.Header.Get("Content-Encoding"), "gzip") {
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
		key := httpcommon.CanonicalHeader(hf.Name)
		trailer[key] = append(trailer[key], hf.Value)
	}
	cs.trailer = trailer

	rl.endStream(cs)
	return nil
}

// transportResponseBody is the concrete type of Transport.RoundTrip's
// Response.Body. It is an io.ReadCloser.
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
				cs.abortStream(err)
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
	connAdd := cc.inflow.add(n)
	var streamAdd int32
	if err == nil { // No need to refresh if the stream is over or failed.
		streamAdd = cs.inflow.add(n)
	}
	cc.mu.Unlock()

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

	cs.bufPipe.BreakWithError(errClosedResponseBody)
	cs.abortStream(errClosedResponseBody)

	unread := cs.bufPipe.Len()
	if unread > 0 {
		cc.mu.Lock()
		// Return connection-level flow control.
		connAdd := cc.inflow.add(unread)
		cc.mu.Unlock()

		// TODO(dneil): Acquiring this mutex can block indefinitely.
		// Move flow control return to a goroutine?
		cc.wmu.Lock()
		// Return connection-level flow control.
		if connAdd > 0 {
			cc.fr.WriteWindowUpdate(0, uint32(connAdd))
		}
		cc.bw.Flush()
		cc.wmu.Unlock()
	}

	select {
	case <-cs.donec:
	case <-cs.ctx.Done():
		// See golang/go#49366: The net/http package can cancel the
		// request context after the response body is fully read.
		// Don't treat this as an error.
		return nil
	case <-cs.reqCancel:
		return errRequestCanceled
	}
	return nil
}

func (rl *clientConnReadLoop) processData(f *DataFrame) error {
	cc := rl.cc
	cs := rl.streamByID(f.StreamID, headerOrDataFrame)
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
			ok := cc.inflow.take(f.Length)
			connAdd := cc.inflow.add(int(f.Length))
			cc.mu.Unlock()
			if !ok {
				return ConnectionError(ErrCodeFlowControl)
			}
			if connAdd > 0 {
				cc.wmu.Lock()
				cc.fr.WriteWindowUpdate(0, uint32(connAdd))
				cc.bw.Flush()
				cc.wmu.Unlock()
			}
		}
		return nil
	}
	if cs.readClosed {
		cc.logf("protocol error: received DATA after END_STREAM")
		rl.endStreamError(cs, StreamError{
			StreamID: f.StreamID,
			Code:     ErrCodeProtocol,
		})
		return nil
	}
	if !cs.pastHeaders {
		cc.logf("protocol error: received DATA before a HEADERS frame")
		rl.endStreamError(cs, StreamError{
			StreamID: f.StreamID,
			Code:     ErrCodeProtocol,
		})
		return nil
	}
	if f.Length > 0 {
		if cs.isHead && len(data) > 0 {
			cc.logf("protocol error: received DATA on a HEAD request")
			rl.endStreamError(cs, StreamError{
				StreamID: f.StreamID,
				Code:     ErrCodeProtocol,
			})
			return nil
		}
		// Check connection-level flow control.
		cc.mu.Lock()
		if !takeInflows(&cc.inflow, &cs.inflow, f.Length) {
			cc.mu.Unlock()
			return ConnectionError(ErrCodeFlowControl)
		}
		// Return any padded flow control now, since we won't
		// refund it later on body reads.
		var refund int
		if pad := int(f.Length) - len(data); pad > 0 {
			refund += pad
		}

		didReset := false
		var err error
		if len(data) > 0 {
			if _, err = cs.bufPipe.Write(data); err != nil {
				// Return len(data) now if the stream is already closed,
				// since data will never be read.
				didReset = true
				refund += len(data)
			}
		}

		sendConn := cc.inflow.add(refund)
		var sendStream int32
		if !didReset {
			sendStream = cs.inflow.add(refund)
		}
		cc.mu.Unlock()

		if sendConn > 0 || sendStream > 0 {
			cc.wmu.Lock()
			if sendConn > 0 {
				cc.fr.WriteWindowUpdate(0, uint32(sendConn))
			}
			if sendStream > 0 {
				cc.fr.WriteWindowUpdate(cs.ID, uint32(sendStream))
			}
			cc.bw.Flush()
			cc.wmu.Unlock()
		}

		if err != nil {
			rl.endStreamError(cs, err)
			return nil
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
	if !cs.readClosed {
		cs.readClosed = true
		// Close cs.bufPipe and cs.peerClosed with cc.mu held to avoid a
		// race condition: The caller can read io.EOF from Response.Body
		// and close the body before we close cs.peerClosed, causing
		// cleanupWriteRequest to send a RST_STREAM.
		rl.cc.mu.Lock()
		defer rl.cc.mu.Unlock()
		cs.bufPipe.closeWithErrorAndCode(io.EOF, cs.copyTrailers)
		close(cs.peerClosed)
	}
}

func (rl *clientConnReadLoop) endStreamError(cs *clientStream, err error) {
	cs.readAborted = true
	cs.abortStream(err)
}

// Constants passed to streamByID for documentation purposes.
const (
	headerOrDataFrame    = true
	notHeaderOrDataFrame = false
)

// streamByID returns the stream with the given id, or nil if no stream has that id.
// If headerOrData is true, it clears rst.StreamPingsBlocked.
func (rl *clientConnReadLoop) streamByID(id uint32, headerOrData bool) *clientStream {
	rl.cc.mu.Lock()
	defer rl.cc.mu.Unlock()
	if headerOrData {
		// Work around an unfortunate gRPC behavior.
		// See comment on ClientConn.rstStreamPingsBlocked for details.
		rl.cc.rstStreamPingsBlocked = false
	}
	cs := rl.cc.streams[id]
	if cs != nil && !cs.readAborted {
		return cs
	}
	return nil
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
		if fn := cc.t.CountError; fn != nil {
			fn("recv_goaway_" + f.ErrCode.stringToken())
		}
	}
	cc.setGoAway(f)
	return nil
}

func (rl *clientConnReadLoop) processSettings(f *SettingsFrame) error {
	cc := rl.cc
	// Locking both mu and wmu here allows frame encoding to read settings with only wmu held.
	// Acquiring wmu when f.IsAck() is unnecessary, but convenient and mostly harmless.
	cc.wmu.Lock()
	defer cc.wmu.Unlock()

	if err := rl.processSettingsNoWrite(f); err != nil {
		return err
	}
	if !f.IsAck() {
		cc.fr.WriteSettingsAck()
		cc.bw.Flush()
	}
	return nil
}

func (rl *clientConnReadLoop) processSettingsNoWrite(f *SettingsFrame) error {
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

	var seenMaxConcurrentStreams bool
	err := f.ForeachSetting(func(s Setting) error {
		switch s.ID {
		case SettingMaxFrameSize:
			cc.maxFrameSize = s.Val
		case SettingMaxConcurrentStreams:
			cc.maxConcurrentStreams = s.Val
			seenMaxConcurrentStreams = true
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
		case SettingHeaderTableSize:
			cc.henc.SetMaxDynamicTableSize(s.Val)
			cc.peerMaxHeaderTableSize = s.Val
		case SettingEnableConnectProtocol:
			if err := s.Valid(); err != nil {
				return err
			}
			// If the peer wants to send us SETTINGS_ENABLE_CONNECT_PROTOCOL,
			// we require that it do so in the first SETTINGS frame.
			//
			// When we attempt to use extended CONNECT, we wait for the first
			// SETTINGS frame to see if the server supports it. If we let the
			// server enable the feature with a later SETTINGS frame, then
			// users will see inconsistent results depending on whether we've
			// seen that frame or not.
			if !cc.seenSettings {
				cc.extendedConnectAllowed = s.Val == 1
			}
		default:
			cc.vlogf("Unhandled Setting: %v", s)
		}
		return nil
	})
	if err != nil {
		return err
	}

	if !cc.seenSettings {
		if !seenMaxConcurrentStreams {
			// This was the servers initial SETTINGS frame and it
			// didn't contain a MAX_CONCURRENT_STREAMS field so
			// increase the number of concurrent streams this
			// connection can establish to our default.
			cc.maxConcurrentStreams = defaultMaxConcurrentStreams
		}
		close(cc.seenSettingsChan)
		cc.seenSettings = true
	}

	return nil
}

func (rl *clientConnReadLoop) processWindowUpdate(f *WindowUpdateFrame) error {
	cc := rl.cc
	cs := rl.streamByID(f.StreamID, notHeaderOrDataFrame)
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
		// For stream, the sender sends RST_STREAM with an error code of FLOW_CONTROL_ERROR
		if cs != nil {
			rl.endStreamError(cs, StreamError{
				StreamID: f.StreamID,
				Code:     ErrCodeFlowControl,
			})
			return nil
		}

		return ConnectionError(ErrCodeFlowControl)
	}
	cc.cond.Broadcast()
	return nil
}

func (rl *clientConnReadLoop) processResetStream(f *RSTStreamFrame) error {
	cs := rl.streamByID(f.StreamID, notHeaderOrDataFrame)
	if cs == nil {
		// TODO: return error if server tries to RST_STREAM an idle stream
		return nil
	}
	serr := streamError(cs.ID, f.ErrCode)
	serr.Cause = errFromPeer
	if f.ErrCode == ErrCodeProtocol {
		rl.cc.SetDoNotReuse()
	}
	if fn := cs.cc.t.CountError; fn != nil {
		fn("recv_rststream_" + f.ErrCode.stringToken())
	}
	cs.abortStream(serr)

	cs.bufPipe.CloseWithError(serr)
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
	var pingError error
	errc := make(chan struct{})
	go func() {
		cc.t.markNewGoroutine()
		cc.wmu.Lock()
		defer cc.wmu.Unlock()
		if pingError = cc.fr.WritePing(false, p); pingError != nil {
			close(errc)
			return
		}
		if pingError = cc.bw.Flush(); pingError != nil {
			close(errc)
			return
		}
	}()
	select {
	case <-c:
		return nil
	case <-errc:
		return pingError
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
		if cc.pendingResets > 0 {
			// See clientStream.cleanupWriteRequest.
			cc.pendingResets = 0
			cc.rstStreamPingsBlocked = true
			cc.cond.Broadcast()
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

// writeStreamReset sends a RST_STREAM frame.
// When ping is true, it also sends a PING frame with a random payload.
func (cc *ClientConn) writeStreamReset(streamID uint32, code ErrCode, ping bool, err error) {
	// TODO: map err to more interesting error codes, once the
	// HTTP community comes up with some. But currently for
	// RST_STREAM there's no equivalent to GOAWAY frame's debug
	// data, and the error codes are all pretty vague ("cancel").
	cc.wmu.Lock()
	cc.fr.WriteRSTStream(streamID, code)
	if ping {
		var payload [8]byte
		rand.Read(payload[:])
		cc.fr.WritePing(false, payload)
	}
	cc.bw.Flush()
	cc.wmu.Unlock()
}

var (
	errResponseHeaderListSize = errors.New("http2: response header list larger than advertised limit")
	errRequestHeaderListSize  = httpcommon.ErrRequestHeaderListSize
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

var noBody io.ReadCloser = noBodyReader{}

type noBodyReader struct{}

func (noBodyReader) Close() error             { return nil }
func (noBodyReader) Read([]byte) (int, error) { return 0, io.EOF }

type missingBody struct{}

func (missingBody) Close() error             { return nil }
func (missingBody) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }

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
	if err := gz.body.Close(); err != nil {
		return err
	}
	gz.zerr = fs.ErrClosed
	return nil
}

type errorReader struct{ err error }

func (r errorReader) Read(p []byte) (int, error) { return 0, r.err }

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
	// to keep things backwards compatible, we use non-zero values of
	// IdleConnTimeout, followed by using the IdleConnTimeout on the underlying
	// http1 transport, followed by 0
	if t.IdleConnTimeout != 0 {
		return t.IdleConnTimeout
	}

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
		ci.IdleTime = cc.t.timeSince(cc.lastActive)
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

func traceGot1xxResponseFunc(trace *httptrace.ClientTrace) func(int, textproto.MIMEHeader) error {
	if trace != nil {
		return trace.Got1xxResponse
	}
	return nil
}

// dialTLSWithContext uses tls.Dialer, added in Go 1.15, to open a TLS
// connection.
func (t *Transport) dialTLSWithContext(ctx context.Context, network, addr string, cfg *tls.Config) (*tls.Conn, error) {
	dialer := &tls.Dialer{
		Config: cfg,
	}
	cn, err := dialer.DialContext(ctx, network, addr)
	if err != nil {
		return nil, err
	}
	tlsCn := cn.(*tls.Conn) // DialContext comment promises this will always succeed
	return tlsCn, nil
}
