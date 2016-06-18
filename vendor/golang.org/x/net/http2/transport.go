// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Transport code.

package http2

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/http2/hpack"
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

	// MaxHeaderListSize is the http2 SETTINGS_MAX_HEADER_LIST_SIZE to
	// send in the initial settings frame. It is how many bytes
	// of response headers are allow. Unlike the http2 spec, zero here
	// means to use a default limit (currently 10MB). If you actually
	// want to advertise an ulimited value to the peer, Transport
	// interprets the highest possible value here (0xffffffff or 1<<32-1)
	// to mean no limit.
	MaxHeaderListSize uint32

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

var errTransportVersion = errors.New("http2: ConfigureTransport is only supported starting at Go 1.6")

// ConfigureTransport configures a net/http HTTP/1 Transport to use HTTP/2.
// It requires Go 1.6 or later and returns an error if the net/http package is too old
// or if t1 has already been HTTP/2-enabled.
func ConfigureTransport(t1 *http.Transport) error {
	_, err := configureTransport(t1) // in configure_transport.go (go1.6) or not_go16.go
	return err
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
	t        *Transport
	tconn    net.Conn             // usually *tls.Conn, except specialized impls
	tlsState *tls.ConnectionState // nil only for specialized impls

	// readLoop goroutine fields:
	readerDone chan struct{} // closed on error
	readerErr  error         // set before readerDone is closed

	mu           sync.Mutex // guards following
	cond         *sync.Cond // hold mu; broadcast on flow/closed changes
	flow         flow       // our conn-level flow control quota (cs.flow is per stream)
	inflow       flow       // peer's conn-level flow control
	closed       bool
	goAway       *GoAwayFrame             // if non-nil, the GoAwayFrame we received
	streams      map[uint32]*clientStream // client-initiated
	nextStreamID uint32
	bw           *bufio.Writer
	br           *bufio.Reader
	fr           *Framer
	// Settings from peer:
	maxFrameSize         uint32
	maxConcurrentStreams uint32
	initialWindowSize    uint32
	hbuf                 bytes.Buffer // HPACK encoder writes into this
	henc                 *hpack.Encoder
	freeBuf              [][]byte

	wmu  sync.Mutex // held while writing; acquire AFTER mu if holding both
	werr error      // first write error that has occurred
}

// clientStream is the state for a single HTTP/2 stream. One of these
// is created for each Transport.RoundTrip call.
type clientStream struct {
	cc            *ClientConn
	req           *http.Request
	ID            uint32
	resc          chan resAndError
	bufPipe       pipe // buffered pipe with the flow-controlled response payload
	requestedGzip bool

	flow        flow  // guarded by cc.mu
	inflow      flow  // guarded by cc.mu
	bytesRemain int64 // -1 means unknown; owned by transportResponseBody.Read
	readErr     error // sticky read error; owned by transportResponseBody.Read
	stopReqBody error // if non-nil, stop writing req body; guarded by cc.mu

	peerReset chan struct{} // closed on peer reset
	resetErr  error         // populated before peerReset is closed

	done chan struct{} // closed when stream remove from cc.streams map; close calls guarded by cc.mu

	// owned by clientConnReadLoop:
	pastHeaders  bool // got first MetaHeadersFrame (actual headers)
	pastTrailers bool // got optional second MetaHeadersFrame (trailers)

	trailer    http.Header  // accumulated trailers
	resTrailer *http.Header // client's Response.Trailer
}

// awaitRequestCancel runs in its own goroutine and waits for the user
// to either cancel a RoundTrip request (using the provided
// Request.Cancel channel), or for the request to be done (any way it
// might be removed from the cc.streams map: peer reset, successful
// completion, TCP connection breakage, etc)
func (cs *clientStream) awaitRequestCancel(cancel <-chan struct{}) {
	if cancel == nil {
		return
	}
	select {
	case <-cancel:
		cs.bufPipe.CloseWithError(errRequestCanceled)
		cs.cc.writeStreamReset(cs.ID, ErrCodeCancel, nil)
	case <-cs.done:
	}
}

// checkReset reports any error sent in a RST_STREAM frame by the
// server.
func (cs *clientStream) checkReset() error {
	select {
	case <-cs.peerReset:
		return cs.resetErr
	default:
		return nil
	}
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

var ErrNoCachedConn = errors.New("http2: no cached connection was available")

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
func authorityAddr(authority string) (addr string) {
	if _, _, err := net.SplitHostPort(authority); err == nil {
		return authority
	}
	return net.JoinHostPort(authority, "443")
}

// RoundTripOpt is like RoundTrip, but takes options.
func (t *Transport) RoundTripOpt(req *http.Request, opt RoundTripOpt) (*http.Response, error) {
	if req.URL.Scheme != "https" {
		return nil, errors.New("http2: unsupported scheme")
	}

	addr := authorityAddr(req.URL.Host)
	for {
		cc, err := t.connPool().GetClientConn(req, addr)
		if err != nil {
			t.vlogf("http2: Transport failed to get client conn for %s: %v", addr, err)
			return nil, err
		}
		res, err := cc.RoundTrip(req)
		if shouldRetryRequest(req, err) {
			continue
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
	if cp, ok := t.connPool().(*clientConnPool); ok {
		cp.closeIdleConnections()
	}
}

var (
	errClientConnClosed   = errors.New("http2: client conn is closed")
	errClientConnUnusable = errors.New("http2: client conn not usable")
)

func shouldRetryRequest(req *http.Request, err error) bool {
	// TODO: retry GET requests (no bodies) more aggressively, if shutdown
	// before response.
	return err == errClientConnUnusable
}

func (t *Transport) dialClientConn(addr string) (*ClientConn, error) {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, err
	}
	tconn, err := t.dialTLS()("tcp", addr, t.newTLSConfig(host))
	if err != nil {
		return nil, err
	}
	return t.NewClientConn(tconn)
}

func (t *Transport) newTLSConfig(host string) *tls.Config {
	cfg := new(tls.Config)
	if t.TLSClientConfig != nil {
		*cfg = *t.TLSClientConfig
	}
	if !strSliceContains(cfg.NextProtos, NextProtoTLS) {
		cfg.NextProtos = append([]string{NextProtoTLS}, cfg.NextProtos...)
	}
	if cfg.ServerName == "" {
		cfg.ServerName = host
	}
	return cfg
}

func (t *Transport) dialTLS() func(string, string, *tls.Config) (net.Conn, error) {
	if t.DialTLS != nil {
		return t.DialTLS
	}
	return t.dialTLSDefault
}

func (t *Transport) dialTLSDefault(network, addr string, cfg *tls.Config) (net.Conn, error) {
	cn, err := tls.Dial(network, addr, cfg)
	if err != nil {
		return nil, err
	}
	if err := cn.Handshake(); err != nil {
		return nil, err
	}
	if !cfg.InsecureSkipVerify {
		if err := cn.VerifyHostname(cfg.ServerName); err != nil {
			return nil, err
		}
	}
	state := cn.ConnectionState()
	if p := state.NegotiatedProtocol; p != NextProtoTLS {
		return nil, fmt.Errorf("http2: unexpected ALPN protocol %q; want %q", p, NextProtoTLS)
	}
	if !state.NegotiatedProtocolIsMutual {
		return nil, errors.New("http2: could not negotiate protocol mutually")
	}
	return cn, nil
}

// disableKeepAlives reports whether connections should be closed as
// soon as possible after handling the first request.
func (t *Transport) disableKeepAlives() bool {
	return t.t1 != nil && t.t1.DisableKeepAlives
}

func (t *Transport) NewClientConn(c net.Conn) (*ClientConn, error) {
	if VerboseLogs {
		t.vlogf("http2: Transport creating client conn to %v", c.RemoteAddr())
	}
	if _, err := c.Write(clientPreface); err != nil {
		t.vlogf("client preface write error: %v", err)
		return nil, err
	}

	cc := &ClientConn{
		t:                    t,
		tconn:                c,
		readerDone:           make(chan struct{}),
		nextStreamID:         1,
		maxFrameSize:         16 << 10, // spec default
		initialWindowSize:    65535,    // spec default
		maxConcurrentStreams: 1000,     // "infinite", per spec. 1000 seems good enough.
		streams:              make(map[uint32]*clientStream),
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

	if cs, ok := c.(connectionStater); ok {
		state := cs.ConnectionState()
		cc.tlsState = &state
	}

	initialSettings := []Setting{
		Setting{ID: SettingEnablePush, Val: 0},
		Setting{ID: SettingInitialWindowSize, Val: transportDefaultStreamFlow},
	}
	if max := t.maxHeaderListSize(); max != 0 {
		initialSettings = append(initialSettings, Setting{ID: SettingMaxHeaderListSize, Val: max})
	}
	cc.fr.WriteSettings(initialSettings...)
	cc.fr.WriteWindowUpdate(0, transportDefaultConnFlow)
	cc.inflow.add(transportDefaultConnFlow + initialWindowSize)
	cc.bw.Flush()
	if cc.werr != nil {
		return nil, cc.werr
	}

	// Read the obligatory SETTINGS frame
	f, err := cc.fr.ReadFrame()
	if err != nil {
		return nil, err
	}
	sf, ok := f.(*SettingsFrame)
	if !ok {
		return nil, fmt.Errorf("expected settings frame, got: %T", f)
	}
	cc.fr.WriteSettingsAck()
	cc.bw.Flush()

	sf.ForeachSetting(func(s Setting) error {
		switch s.ID {
		case SettingMaxFrameSize:
			cc.maxFrameSize = s.Val
		case SettingMaxConcurrentStreams:
			cc.maxConcurrentStreams = s.Val
		case SettingInitialWindowSize:
			cc.initialWindowSize = s.Val
		default:
			// TODO(bradfitz): handle more; at least SETTINGS_HEADER_TABLE_SIZE?
			t.vlogf("Unhandled Setting: %v", s)
		}
		return nil
	})

	go cc.readLoop()
	return cc, nil
}

func (cc *ClientConn) setGoAway(f *GoAwayFrame) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.goAway = f
}

func (cc *ClientConn) CanTakeNewRequest() bool {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.canTakeNewRequestLocked()
}

func (cc *ClientConn) canTakeNewRequestLocked() bool {
	return cc.goAway == nil && !cc.closed &&
		int64(len(cc.streams)+1) < int64(cc.maxConcurrentStreams) &&
		cc.nextStreamID < 2147483647
}

func (cc *ClientConn) closeIfIdle() {
	cc.mu.Lock()
	if len(cc.streams) > 0 {
		cc.mu.Unlock()
		return
	}
	cc.closed = true
	// TODO: do clients send GOAWAY too? maybe? Just Close:
	cc.mu.Unlock()

	cc.tconn.Close()
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
			return "", &badStringError{"invalid Trailer key", k}
		}
		keys = append(keys, k)
	}
	if len(keys) > 0 {
		sort.Strings(keys)
		// TODO: could do better allocation-wise here, but trailers are rare,
		// so being lazy for now.
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
		return errors.New("http2: invalid Upgrade request header")
	}
	if v := req.Header.Get("Transfer-Encoding"); (v != "" && v != "chunked") || len(req.Header["Transfer-Encoding"]) > 1 {
		return errors.New("http2: invalid Transfer-Encoding request header")
	}
	if v := req.Header.Get("Connection"); (v != "" && v != "close" && v != "keep-alive") || len(req.Header["Connection"]) > 1 {
		return errors.New("http2: invalid Connection request header")
	}
	return nil
}

func (cc *ClientConn) RoundTrip(req *http.Request) (*http.Response, error) {
	if err := checkConnHeaders(req); err != nil {
		return nil, err
	}

	trailers, err := commaSeparatedTrailers(req)
	if err != nil {
		return nil, err
	}
	hasTrailers := trailers != ""

	var body io.Reader = req.Body
	contentLen := req.ContentLength
	if req.Body != nil && contentLen == 0 {
		// Test to see if it's actually zero or just unset.
		var buf [1]byte
		n, rerr := io.ReadFull(body, buf[:])
		if rerr != nil && rerr != io.EOF {
			contentLen = -1
			body = errorReader{rerr}
		} else if n == 1 {
			// Oh, guess there is data in this Body Reader after all.
			// The ContentLength field just wasn't set.
			// Stich the Body back together again, re-attaching our
			// consumed byte.
			contentLen = -1
			body = io.MultiReader(bytes.NewReader(buf[:]), body)
		} else {
			// Body is actually empty.
			body = nil
		}
	}

	cc.mu.Lock()
	if cc.closed || !cc.canTakeNewRequestLocked() {
		cc.mu.Unlock()
		return nil, errClientConnUnusable
	}

	cs := cc.newStream()
	cs.req = req
	hasBody := body != nil

	// TODO(bradfitz): this is a copy of the logic in net/http. Unify somewhere?
	if !cc.t.disableCompression() &&
		req.Header.Get("Accept-Encoding") == "" &&
		req.Header.Get("Range") == "" &&
		req.Method != "HEAD" {
		// Request gzip only, not deflate. Deflate is ambiguous and
		// not as universally supported anyway.
		// See: http://www.gzip.org/zlib/zlib_faq.html#faq38
		//
		// Note that we don't request this for HEAD requests,
		// due to a bug in nginx:
		//   http://trac.nginx.org/nginx/ticket/358
		//   https://golang.org/issue/5522
		//
		// We don't request gzip if the request is for a range, since
		// auto-decoding a portion of a gzipped document will just fail
		// anyway. See https://golang.org/issue/8923
		cs.requestedGzip = true
	}

	// we send: HEADERS{1}, CONTINUATION{0,} + DATA{0,} (DATA is
	// sent by writeRequestBody below, along with any Trailers,
	// again in form HEADERS{1}, CONTINUATION{0,})
	hdrs := cc.encodeHeaders(req, cs.requestedGzip, trailers, contentLen)
	cc.wmu.Lock()
	endStream := !hasBody && !hasTrailers
	werr := cc.writeHeaders(cs.ID, endStream, hdrs)
	cc.wmu.Unlock()
	cc.mu.Unlock()

	if werr != nil {
		if hasBody {
			req.Body.Close() // per RoundTripper contract
		}
		cc.forgetStreamID(cs.ID)
		// Don't bother sending a RST_STREAM (our write already failed;
		// no need to keep writing)
		return nil, werr
	}

	var respHeaderTimer <-chan time.Time
	var bodyCopyErrc chan error // result of body copy
	if hasBody {
		bodyCopyErrc = make(chan error, 1)
		go func() {
			bodyCopyErrc <- cs.writeRequestBody(body, req.Body)
		}()
	} else {
		if d := cc.responseHeaderTimeout(); d != 0 {
			timer := time.NewTimer(d)
			defer timer.Stop()
			respHeaderTimer = timer.C
		}
	}

	readLoopResCh := cs.resc
	requestCanceledCh := requestCancel(req)
	bodyWritten := false

	for {
		select {
		case re := <-readLoopResCh:
			res := re.res
			if re.err != nil || res.StatusCode > 299 {
				// On error or status code 3xx, 4xx, 5xx, etc abort any
				// ongoing write, assuming that the server doesn't care
				// about our request body. If the server replied with 1xx or
				// 2xx, however, then assume the server DOES potentially
				// want our body (e.g. full-duplex streaming:
				// golang.org/issue/13444). If it turns out the server
				// doesn't, they'll RST_STREAM us soon enough.  This is a
				// heuristic to avoid adding knobs to Transport.  Hopefully
				// we can keep it.
				cs.abortRequestBodyWrite(errStopReqBodyWrite)
			}
			if re.err != nil {
				cc.forgetStreamID(cs.ID)
				return nil, re.err
			}
			res.Request = req
			res.TLS = cc.tlsState
			return res, nil
		case <-respHeaderTimer:
			cc.forgetStreamID(cs.ID)
			if !hasBody || bodyWritten {
				cc.writeStreamReset(cs.ID, ErrCodeCancel, nil)
			} else {
				cs.abortRequestBodyWrite(errStopReqBodyWriteAndCancel)
			}
			return nil, errTimeout
		case <-requestCanceledCh:
			cc.forgetStreamID(cs.ID)
			if !hasBody || bodyWritten {
				cc.writeStreamReset(cs.ID, ErrCodeCancel, nil)
			} else {
				cs.abortRequestBodyWrite(errStopReqBodyWriteAndCancel)
			}
			return nil, errRequestCanceled
		case <-cs.peerReset:
			// processResetStream already removed the
			// stream from the streams map; no need for
			// forgetStreamID.
			return nil, cs.resetErr
		case err := <-bodyCopyErrc:
			if err != nil {
				return nil, err
			}
			bodyWritten = true
			if d := cc.responseHeaderTimeout(); d != 0 {
				timer := time.NewTimer(d)
				defer timer.Stop()
				respHeaderTimer = timer.C
			}
		}
	}
}

// requires cc.wmu be held
func (cc *ClientConn) writeHeaders(streamID uint32, endStream bool, hdrs []byte) error {
	first := true // first frame written (HEADERS is first, then CONTINUATION)
	frameSize := int(cc.maxFrameSize)
	for len(hdrs) > 0 && cc.werr == nil {
		chunk := hdrs
		if len(chunk) > frameSize {
			chunk = chunk[:frameSize]
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
)

func (cs *clientStream) writeRequestBody(body io.Reader, bodyCloser io.Closer) (err error) {
	cc := cs.cc
	sentEnd := false // whether we sent the final DATA frame w/ END_STREAM
	buf := cc.frameScratchBuffer()
	defer cc.putFrameScratchBuffer(buf)

	defer func() {
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

	var sawEOF bool
	for !sawEOF {
		n, err := body.Read(buf)
		if err == io.EOF {
			sawEOF = true
			err = nil
		} else if err != nil {
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
				// Most requests won't need this. Make this opt-in or opt-out?
				// Use some heuristic on the body type? Nagel-like timers?
				// Based on 'n'? Only last chunk of this for loop, unless flow control
				// tokens are low? For now, always:
				err = cc.bw.Flush()
			}
			cc.wmu.Unlock()
		}
		if err != nil {
			return err
		}
	}

	cc.wmu.Lock()
	if !sentEnd {
		var trls []byte
		if hasTrailers {
			cc.mu.Lock()
			trls = cc.encodeTrailers(req)
			cc.mu.Unlock()
		}

		// Avoid forgetting to send an END_STREAM if the encoded
		// trailers are 0 bytes. Both results produce and END_STREAM.
		if len(trls) > 0 {
			err = cc.writeHeaders(cs.ID, true, trls)
		} else {
			err = cc.fr.WriteData(cs.ID, true, nil)
		}
	}
	if ferr := cc.bw.Flush(); ferr != nil && err == nil {
		err = ferr
	}
	cc.wmu.Unlock()

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
		if err := cs.checkReset(); err != nil {
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

type badStringError struct {
	what string
	str  string
}

func (e *badStringError) Error() string { return fmt.Sprintf("%s %q", e.what, e.str) }

// requires cc.mu be held.
func (cc *ClientConn) encodeHeaders(req *http.Request, addGzipHeader bool, trailers string, contentLength int64) []byte {
	cc.hbuf.Reset()

	host := req.Host
	if host == "" {
		host = req.URL.Host
	}

	// 8.1.2.3 Request Pseudo-Header Fields
	// The :path pseudo-header field includes the path and query parts of the
	// target URI (the path-absolute production and optionally a '?' character
	// followed by the query production (see Sections 3.3 and 3.4 of
	// [RFC3986]).
	cc.writeHeader(":authority", host)
	cc.writeHeader(":method", req.Method)
	if req.Method != "CONNECT" {
		cc.writeHeader(":path", req.URL.RequestURI())
		cc.writeHeader(":scheme", "https")
	}
	if trailers != "" {
		cc.writeHeader("trailer", trailers)
	}

	var didUA bool
	for k, vv := range req.Header {
		lowKey := strings.ToLower(k)
		switch lowKey {
		case "host", "content-length":
			// Host is :authority, already sent.
			// Content-Length is automatic, set below.
			continue
		case "connection", "proxy-connection", "transfer-encoding", "upgrade":
			// Per 8.1.2.2 Connection-Specific Header
			// Fields, don't send connection-specific
			// fields. We deal with these earlier in
			// RoundTrip, deciding whether they're
			// error-worthy, but we don't want to mutate
			// the user's *Request so at this point, just
			// skip over them at this point.
			continue
		case "user-agent":
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
		}
		for _, v := range vv {
			cc.writeHeader(lowKey, v)
		}
	}
	if shouldSendReqContentLength(req.Method, contentLength) {
		cc.writeHeader("content-length", strconv.FormatInt(contentLength, 10))
	}
	if addGzipHeader {
		cc.writeHeader("accept-encoding", "gzip")
	}
	if !didUA {
		cc.writeHeader("user-agent", defaultUserAgent)
	}
	return cc.hbuf.Bytes()
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
func (cc *ClientConn) encodeTrailers(req *http.Request) []byte {
	cc.hbuf.Reset()
	for k, vv := range req.Trailer {
		// Transfer-Encoding, etc.. have already been filter at the
		// start of RoundTrip
		lowKey := strings.ToLower(k)
		for _, v := range vv {
			cc.writeHeader(lowKey, v)
		}
	}
	return cc.hbuf.Bytes()
}

func (cc *ClientConn) writeHeader(name, value string) {
	if VerboseLogs {
		log.Printf("http2: Transport encoding header %q = %q", name, value)
	}
	cc.henc.WriteField(hpack.HeaderField{Name: name, Value: value})
}

type resAndError struct {
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
		delete(cc.streams, id)
		close(cs.done)
	}
	return cs
}

// clientConnReadLoop is the state owned by the clientConn's frame-reading readLoop.
type clientConnReadLoop struct {
	cc            *ClientConn
	activeRes     map[uint32]*clientStream // keyed by streamID
	closeWhenIdle bool
}

// readLoop runs in its own goroutine and reads and dispatches frames.
func (cc *ClientConn) readLoop() {
	rl := &clientConnReadLoop{
		cc:        cc,
		activeRes: make(map[uint32]*clientStream),
	}

	defer rl.cleanup()
	cc.readerErr = rl.run()
	if ce, ok := cc.readerErr.(ConnectionError); ok {
		cc.wmu.Lock()
		cc.fr.WriteGoAway(0, ErrCode(ce), nil)
		cc.wmu.Unlock()
	}
}

func (rl *clientConnReadLoop) cleanup() {
	cc := rl.cc
	defer cc.tconn.Close()
	defer cc.t.connPool().MarkDead(cc)
	defer close(cc.readerDone)

	// Close any response bodies if the server closes prematurely.
	// TODO: also do this if we've written the headers but not
	// gotten a response yet.
	err := cc.readerErr
	if err == io.EOF {
		err = io.ErrUnexpectedEOF
	}
	cc.mu.Lock()
	for _, cs := range rl.activeRes {
		cs.bufPipe.CloseWithError(err)
	}
	for _, cs := range cc.streams {
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
	rl.closeWhenIdle = cc.t.disableKeepAlives()
	gotReply := false // ever saw a reply
	for {
		f, err := cc.fr.ReadFrame()
		if err != nil {
			cc.vlogf("Transport readFrame error: (%T) %v", err, err)
		}
		if se, ok := err.(StreamError); ok {
			if cs := cc.streamByID(se.StreamID, true /*ended; remove it*/); cs != nil {
				rl.endStreamError(cs, cc.fr.errDetail)
			}
			continue
		} else if err != nil {
			return err
		}
		if VerboseLogs {
			cc.vlogf("http2: Transport received %s", summarizeFrame(f))
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
			return err
		}
		if rl.closeWhenIdle && gotReply && maybeIdle && len(rl.activeRes) == 0 {
			cc.closeIfIdle()
		}
	}
}

func (rl *clientConnReadLoop) processHeaders(f *MetaHeadersFrame) error {
	cc := rl.cc
	cs := cc.streamByID(f.StreamID, f.StreamEnded())
	if cs == nil {
		// We'd get here if we canceled a request while the
		// server had its response still in flight. So if this
		// was just something we canceled, ignore it.
		return nil
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
		cs.resc <- resAndError{err: err}
		return nil // return nil from process* funcs to keep conn alive
	}
	if res == nil {
		// (nil, nil) special case. See handleResponse docs.
		return nil
	}
	if res.Body != noBody {
		rl.activeRes[cs.ID] = cs
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
// frame (currently only used for 100 expect continue). This special
// case is going away after Issue 13851 is fixed.
func (rl *clientConnReadLoop) handleResponse(cs *clientStream, f *MetaHeadersFrame) (*http.Response, error) {
	if f.Truncated {
		return nil, errResponseHeaderListSize
	}

	status := f.PseudoValue("status")
	if status == "" {
		return nil, errors.New("missing status pseudo header")
	}
	statusCode, err := strconv.Atoi(status)
	if err != nil {
		return nil, errors.New("malformed non-numeric status pseudo header")
	}

	if statusCode == 100 {
		// Just skip 100-continue response headers for now.
		// TODO: golang.org/issue/13851 for doing it properly.
		cs.pastHeaders = false // do it all again
		return nil, nil
	}

	header := make(http.Header)
	res := &http.Response{
		Proto:      "HTTP/2.0",
		ProtoMajor: 2,
		Header:     header,
		StatusCode: statusCode,
		Status:     status + " " + http.StatusText(statusCode),
	}
	for _, hf := range f.RegularFields() {
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
			header[key] = append(header[key], hf.Value)
		}
	}

	streamEnded := f.StreamEnded()
	if !streamEnded || cs.req.Method == "HEAD" {
		res.ContentLength = -1
		if clens := res.Header["Content-Length"]; len(clens) == 1 {
			if clen64, err := strconv.ParseInt(clens[0], 10, 64); err == nil {
				res.ContentLength = clen64
			} else {
				// TODO: care? unlike http/1, it won't mess up our framing, so it's
				// more safe smuggling-wise to ignore.
			}
		} else if len(clens) > 1 {
			// TODO: care? unlike http/1, it won't mess up our framing, so it's
			// more safe smuggling-wise to ignore.
		}
	}

	if streamEnded {
		res.Body = noBody
		return res, nil
	}

	buf := new(bytes.Buffer) // TODO(bradfitz): recycle this garbage
	cs.bufPipe = pipe{b: buf}
	cs.bytesRemain = res.ContentLength
	res.Body = transportResponseBody{cs}
	go cs.awaitRequestCancel(requestCancel(cs.req))

	if cs.requestedGzip && res.Header.Get("Content-Encoding") == "gzip" {
		res.Header.Del("Content-Encoding")
		res.Header.Del("Content-Length")
		res.ContentLength = -1
		res.Body = &gzipReader{body: res.Body}
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
		if v := cs.inflow.available(); v < transportDefaultStreamFlow-transportDefaultStreamMinRefresh {
			streamAdd = transportDefaultStreamFlow - v
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
	if cs.bufPipe.Err() != io.EOF {
		// TODO: write test for this
		cs.cc.writeStreamReset(cs.ID, ErrCodeCancel, nil)
	}
	cs.bufPipe.BreakWithError(errClosedResponseBody)
	return nil
}

func (rl *clientConnReadLoop) processData(f *DataFrame) error {
	cc := rl.cc
	cs := cc.streamByID(f.StreamID, f.StreamEnded())
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
		return nil
	}
	if data := f.Data(); len(data) > 0 {
		if cs.bufPipe.b == nil {
			// Data frame after it's already closed?
			cc.logf("http2: Transport received DATA frame for closed stream; closing connection")
			return ConnectionError(ErrCodeProtocol)
		}

		// Check connection-level flow control.
		cc.mu.Lock()
		if cs.inflow.available() >= int32(len(data)) {
			cs.inflow.take(int32(len(data)))
		} else {
			cc.mu.Unlock()
			return ConnectionError(ErrCodeFlowControl)
		}
		cc.mu.Unlock()

		if _, err := cs.bufPipe.Write(data); err != nil {
			rl.endStreamError(cs, err)
			return err
		}
	}

	if f.StreamEnded() {
		rl.endStream(cs)
	}
	return nil
}

var errInvalidTrailers = errors.New("http2: invalid trailers")

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
	cs.bufPipe.closeWithErrorAndCode(err, code)
	delete(rl.activeRes, cs.ID)
	if cs.req.Close || cs.req.Header.Get("Connection") == "close" {
		rl.closeWhenIdle = true
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
	return f.ForeachSetting(func(s Setting) error {
		switch s.ID {
		case SettingMaxFrameSize:
			cc.maxFrameSize = s.Val
		case SettingMaxConcurrentStreams:
			cc.maxConcurrentStreams = s.Val
		case SettingInitialWindowSize:
			// TODO: error if this is too large.

			// TODO: adjust flow control of still-open
			// frames by the difference of the old initial
			// window size and this one.
			cc.initialWindowSize = s.Val
		default:
			// TODO(bradfitz): handle more settings? SETTINGS_HEADER_TABLE_SIZE probably.
			cc.vlogf("Unhandled Setting: %v", s)
		}
		return nil
	})
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
		err := StreamError{cs.ID, f.ErrCode}
		cs.resetErr = err
		close(cs.peerReset)
		cs.bufPipe.CloseWithError(err)
		cs.cc.cond.Broadcast() // wake up checkReset via clientStream.awaitFlowControl
	}
	delete(rl.activeRes, cs.ID)
	return nil
}

func (rl *clientConnReadLoop) processPing(f *PingFrame) error {
	if f.IsAck() {
		// 6.7 PING: " An endpoint MUST NOT respond to PING frames
		// containing this flag."
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
	// TODO: do something with err? send it as a debug frame to the peer?
	// But that's only in GOAWAY. Invent a new frame type? Is there one already?
	cc.wmu.Lock()
	cc.fr.WriteRSTStream(streamID, code)
	cc.bw.Flush()
	cc.wmu.Unlock()
}

var (
	errResponseHeaderListSize = errors.New("http2: response header list larger than advertised limit")
	errPseudoTrailers         = errors.New("http2: invalid pseudo header in trailers")
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

func (rt erringRoundTripper) RoundTrip(*http.Request) (*http.Response, error) { return nil, rt.err }

// gzipReader wraps a response body so it can lazily
// call gzip.NewReader on the first call to Read
type gzipReader struct {
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
