// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: replace all <-sc.doneServing with reads from the stream's cw
// instead, and make sure that on close we close all open
// streams. then remove doneServing?

// TODO: re-audit GOAWAY support. Consider each incoming frame type and
// whether it should be ignored during graceful shutdown.

// TODO: disconnect idle clients. GFE seems to do 4 minutes. make
// configurable?  or maximum number of idle clients and remove the
// oldest?

// TODO: turn off the serve goroutine when idle, so
// an idle conn only has the readFrames goroutine active. (which could
// also be optimized probably to pin less memory in crypto/tls). This
// would involve tracking when the serve goroutine is active (atomic
// int32 read/CAS probably?) and starting it up when frames arrive,
// and shutting it down when all handlers exit. the occasional PING
// packets could use time.AfterFunc to call sc.wakeStartServeLoop()
// (which is a no-op if already running) and then queue the PING write
// as normal. The serve loop would then exit in most cases (if no
// Handlers running) and not be woken up again until the PING packet
// returns.

// TODO (maybe): add a mechanism for Handlers to going into
// half-closed-local mode (rw.(io.Closer) test?) but not exit their
// handler, and continue to be able to read from the
// Request.Body. This would be a somewhat semantic change from HTTP/1
// (or at least what we expose in net/http), so I'd probably want to
// add it there too. For now, this package says that returning from
// the Handler ServeHTTP function means you're both done reading and
// done writing, without a way to stop just one or the other.

package http2

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/textproto"
	"net/url"
	"os"
	"reflect"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/http2/hpack"
)

const (
	prefaceTimeout        = 10 * time.Second
	firstSettingsTimeout  = 2 * time.Second // should be in-flight with preface anyway
	handlerChunkWriteSize = 4 << 10
	defaultMaxStreams     = 250 // TODO: make this 100 as the GFE seems to?
)

var (
	errClientDisconnected = errors.New("client disconnected")
	errClosedBody         = errors.New("body closed by handler")
	errHandlerComplete    = errors.New("http2: request body closed due to handler exiting")
	errStreamClosed       = errors.New("http2: stream closed")
)

var responseWriterStatePool = sync.Pool{
	New: func() interface{} {
		rws := &responseWriterState{}
		rws.bw = bufio.NewWriterSize(chunkWriter{rws}, handlerChunkWriteSize)
		return rws
	},
}

// Test hooks.
var (
	testHookOnConn        func()
	testHookGetServerConn func(*serverConn)
	testHookOnPanicMu     *sync.Mutex // nil except in tests
	testHookOnPanic       func(sc *serverConn, panicVal interface{}) (rePanic bool)
)

// Server is an HTTP/2 server.
type Server struct {
	// MaxHandlers limits the number of http.Handler ServeHTTP goroutines
	// which may run at a time over all connections.
	// Negative or zero no limit.
	// TODO: implement
	MaxHandlers int

	// MaxConcurrentStreams optionally specifies the number of
	// concurrent streams that each client may have open at a
	// time. This is unrelated to the number of http.Handler goroutines
	// which may be active globally, which is MaxHandlers.
	// If zero, MaxConcurrentStreams defaults to at least 100, per
	// the HTTP/2 spec's recommendations.
	MaxConcurrentStreams uint32

	// MaxReadFrameSize optionally specifies the largest frame
	// this server is willing to read. A valid value is between
	// 16k and 16M, inclusive. If zero or otherwise invalid, a
	// default value is used.
	MaxReadFrameSize uint32

	// PermitProhibitedCipherSuites, if true, permits the use of
	// cipher suites prohibited by the HTTP/2 spec.
	PermitProhibitedCipherSuites bool
}

func (s *Server) maxReadFrameSize() uint32 {
	if v := s.MaxReadFrameSize; v >= minMaxFrameSize && v <= maxFrameSize {
		return v
	}
	return defaultMaxReadFrameSize
}

func (s *Server) maxConcurrentStreams() uint32 {
	if v := s.MaxConcurrentStreams; v > 0 {
		return v
	}
	return defaultMaxStreams
}

// ConfigureServer adds HTTP/2 support to a net/http Server.
//
// The configuration conf may be nil.
//
// ConfigureServer must be called before s begins serving.
func ConfigureServer(s *http.Server, conf *Server) error {
	if conf == nil {
		conf = new(Server)
	}

	if s.TLSConfig == nil {
		s.TLSConfig = new(tls.Config)
	} else if s.TLSConfig.CipherSuites != nil {
		// If they already provided a CipherSuite list, return
		// an error if it has a bad order or is missing
		// ECDHE_RSA_WITH_AES_128_GCM_SHA256.
		const requiredCipher = tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
		haveRequired := false
		sawBad := false
		for i, cs := range s.TLSConfig.CipherSuites {
			if cs == requiredCipher {
				haveRequired = true
			}
			if isBadCipher(cs) {
				sawBad = true
			} else if sawBad {
				return fmt.Errorf("http2: TLSConfig.CipherSuites index %d contains an HTTP/2-approved cipher suite (%#04x), but it comes after unapproved cipher suites. With this configuration, clients that don't support previous, approved cipher suites may be given an unapproved one and reject the connection.", i, cs)
			}
		}
		if !haveRequired {
			return fmt.Errorf("http2: TLSConfig.CipherSuites is missing HTTP/2-required TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256")
		}
	}

	// Note: not setting MinVersion to tls.VersionTLS12,
	// as we don't want to interfere with HTTP/1.1 traffic
	// on the user's server. We enforce TLS 1.2 later once
	// we accept a connection. Ideally this should be done
	// during next-proto selection, but using TLS <1.2 with
	// HTTP/2 is still the client's bug.

	s.TLSConfig.PreferServerCipherSuites = true

	haveNPN := false
	for _, p := range s.TLSConfig.NextProtos {
		if p == NextProtoTLS {
			haveNPN = true
			break
		}
	}
	if !haveNPN {
		s.TLSConfig.NextProtos = append(s.TLSConfig.NextProtos, NextProtoTLS)
	}
	// h2-14 is temporary (as of 2015-03-05) while we wait for all browsers
	// to switch to "h2".
	s.TLSConfig.NextProtos = append(s.TLSConfig.NextProtos, "h2-14")

	if s.TLSNextProto == nil {
		s.TLSNextProto = map[string]func(*http.Server, *tls.Conn, http.Handler){}
	}
	protoHandler := func(hs *http.Server, c *tls.Conn, h http.Handler) {
		if testHookOnConn != nil {
			testHookOnConn()
		}
		conf.ServeConn(c, &ServeConnOpts{
			Handler:    h,
			BaseConfig: hs,
		})
	}
	s.TLSNextProto[NextProtoTLS] = protoHandler
	s.TLSNextProto["h2-14"] = protoHandler // temporary; see above.
	return nil
}

// ServeConnOpts are options for the Server.ServeConn method.
type ServeConnOpts struct {
	// BaseConfig optionally sets the base configuration
	// for values. If nil, defaults are used.
	BaseConfig *http.Server

	// Handler specifies which handler to use for processing
	// requests. If nil, BaseConfig.Handler is used. If BaseConfig
	// or BaseConfig.Handler is nil, http.DefaultServeMux is used.
	Handler http.Handler
}

func (o *ServeConnOpts) baseConfig() *http.Server {
	if o != nil && o.BaseConfig != nil {
		return o.BaseConfig
	}
	return new(http.Server)
}

func (o *ServeConnOpts) handler() http.Handler {
	if o != nil {
		if o.Handler != nil {
			return o.Handler
		}
		if o.BaseConfig != nil && o.BaseConfig.Handler != nil {
			return o.BaseConfig.Handler
		}
	}
	return http.DefaultServeMux
}

// ServeConn serves HTTP/2 requests on the provided connection and
// blocks until the connection is no longer readable.
//
// ServeConn starts speaking HTTP/2 assuming that c has not had any
// reads or writes. It writes its initial settings frame and expects
// to be able to read the preface and settings frame from the
// client. If c has a ConnectionState method like a *tls.Conn, the
// ConnectionState is used to verify the TLS ciphersuite and to set
// the Request.TLS field in Handlers.
//
// ServeConn does not support h2c by itself. Any h2c support must be
// implemented in terms of providing a suitably-behaving net.Conn.
//
// The opts parameter is optional. If nil, default values are used.
func (s *Server) ServeConn(c net.Conn, opts *ServeConnOpts) {
	sc := &serverConn{
		srv:              s,
		hs:               opts.baseConfig(),
		conn:             c,
		remoteAddrStr:    c.RemoteAddr().String(),
		bw:               newBufferedWriter(c),
		handler:          opts.handler(),
		streams:          make(map[uint32]*stream),
		readFrameCh:      make(chan readFrameResult),
		wantWriteFrameCh: make(chan frameWriteMsg, 8),
		wroteFrameCh:     make(chan frameWriteResult, 1), // buffered; one send in writeFrameAsync
		bodyReadCh:       make(chan bodyReadMsg),         // buffering doesn't matter either way
		doneServing:      make(chan struct{}),
		advMaxStreams:    s.maxConcurrentStreams(),
		writeSched: writeScheduler{
			maxFrameSize: initialMaxFrameSize,
		},
		initialWindowSize: initialWindowSize,
		headerTableSize:   initialHeaderTableSize,
		serveG:            newGoroutineLock(),
		pushEnabled:       true,
	}
	sc.flow.add(initialWindowSize)
	sc.inflow.add(initialWindowSize)
	sc.hpackEncoder = hpack.NewEncoder(&sc.headerWriteBuf)
	sc.hpackDecoder = hpack.NewDecoder(initialHeaderTableSize, nil)
	sc.hpackDecoder.SetMaxStringLength(sc.maxHeaderStringLen())

	fr := NewFramer(sc.bw, c)
	fr.SetMaxReadFrameSize(s.maxReadFrameSize())
	sc.framer = fr

	if tc, ok := c.(connectionStater); ok {
		sc.tlsState = new(tls.ConnectionState)
		*sc.tlsState = tc.ConnectionState()
		// 9.2 Use of TLS Features
		// An implementation of HTTP/2 over TLS MUST use TLS
		// 1.2 or higher with the restrictions on feature set
		// and cipher suite described in this section. Due to
		// implementation limitations, it might not be
		// possible to fail TLS negotiation. An endpoint MUST
		// immediately terminate an HTTP/2 connection that
		// does not meet the TLS requirements described in
		// this section with a connection error (Section
		// 5.4.1) of type INADEQUATE_SECURITY.
		if sc.tlsState.Version < tls.VersionTLS12 {
			sc.rejectConn(ErrCodeInadequateSecurity, "TLS version too low")
			return
		}

		if sc.tlsState.ServerName == "" {
			// Client must use SNI, but we don't enforce that anymore,
			// since it was causing problems when connecting to bare IP
			// addresses during development.
			//
			// TODO: optionally enforce? Or enforce at the time we receive
			// a new request, and verify the the ServerName matches the :authority?
			// But that precludes proxy situations, perhaps.
			//
			// So for now, do nothing here again.
		}

		if !s.PermitProhibitedCipherSuites && isBadCipher(sc.tlsState.CipherSuite) {
			// "Endpoints MAY choose to generate a connection error
			// (Section 5.4.1) of type INADEQUATE_SECURITY if one of
			// the prohibited cipher suites are negotiated."
			//
			// We choose that. In my opinion, the spec is weak
			// here. It also says both parties must support at least
			// TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 so there's no
			// excuses here. If we really must, we could allow an
			// "AllowInsecureWeakCiphers" option on the server later.
			// Let's see how it plays out first.
			sc.rejectConn(ErrCodeInadequateSecurity, fmt.Sprintf("Prohibited TLS 1.2 Cipher Suite: %x", sc.tlsState.CipherSuite))
			return
		}
	}

	if hook := testHookGetServerConn; hook != nil {
		hook(sc)
	}
	sc.serve()
}

// isBadCipher reports whether the cipher is blacklisted by the HTTP/2 spec.
func isBadCipher(cipher uint16) bool {
	switch cipher {
	case tls.TLS_RSA_WITH_RC4_128_SHA,
		tls.TLS_RSA_WITH_3DES_EDE_CBC_SHA,
		tls.TLS_RSA_WITH_AES_128_CBC_SHA,
		tls.TLS_RSA_WITH_AES_256_CBC_SHA,
		tls.TLS_ECDHE_ECDSA_WITH_RC4_128_SHA,
		tls.TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA,
		tls.TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA,
		tls.TLS_ECDHE_RSA_WITH_RC4_128_SHA,
		tls.TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA,
		tls.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA,
		tls.TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA:
		// Reject cipher suites from Appendix A.
		// "This list includes those cipher suites that do not
		// offer an ephemeral key exchange and those that are
		// based on the TLS null, stream or block cipher type"
		return true
	default:
		return false
	}
}

func (sc *serverConn) rejectConn(err ErrCode, debug string) {
	sc.vlogf("http2: server rejecting conn: %v, %s", err, debug)
	// ignoring errors. hanging up anyway.
	sc.framer.WriteGoAway(0, err, []byte(debug))
	sc.bw.Flush()
	sc.conn.Close()
}

type serverConn struct {
	// Immutable:
	srv              *Server
	hs               *http.Server
	conn             net.Conn
	bw               *bufferedWriter // writing to conn
	handler          http.Handler
	framer           *Framer
	hpackDecoder     *hpack.Decoder
	doneServing      chan struct{}         // closed when serverConn.serve ends
	readFrameCh      chan readFrameResult  // written by serverConn.readFrames
	wantWriteFrameCh chan frameWriteMsg    // from handlers -> serve
	wroteFrameCh     chan frameWriteResult // from writeFrameAsync -> serve, tickles more frame writes
	bodyReadCh       chan bodyReadMsg      // from handlers -> serve
	testHookCh       chan func(int)        // code to run on the serve loop
	flow             flow                  // conn-wide (not stream-specific) outbound flow control
	inflow           flow                  // conn-wide inbound flow control
	tlsState         *tls.ConnectionState  // shared by all handlers, like net/http
	remoteAddrStr    string

	// Everything following is owned by the serve loop; use serveG.check():
	serveG                goroutineLock // used to verify funcs are on serve()
	pushEnabled           bool
	sawFirstSettings      bool // got the initial SETTINGS frame after the preface
	needToSendSettingsAck bool
	unackedSettings       int    // how many SETTINGS have we sent without ACKs?
	clientMaxStreams      uint32 // SETTINGS_MAX_CONCURRENT_STREAMS from client (our PUSH_PROMISE limit)
	advMaxStreams         uint32 // our SETTINGS_MAX_CONCURRENT_STREAMS advertised the client
	curOpenStreams        uint32 // client's number of open streams
	maxStreamID           uint32 // max ever seen
	streams               map[uint32]*stream
	initialWindowSize     int32
	headerTableSize       uint32
	peerMaxHeaderListSize uint32            // zero means unknown (default)
	canonHeader           map[string]string // http2-lower-case -> Go-Canonical-Case
	req                   requestParam      // non-zero while reading request headers
	writingFrame          bool              // started write goroutine but haven't heard back on wroteFrameCh
	needsFrameFlush       bool              // last frame write wasn't a flush
	writeSched            writeScheduler
	inGoAway              bool // we've started to or sent GOAWAY
	needToSendGoAway      bool // we need to schedule a GOAWAY frame write
	goAwayCode            ErrCode
	shutdownTimerCh       <-chan time.Time // nil until used
	shutdownTimer         *time.Timer      // nil until used

	// Owned by the writeFrameAsync goroutine:
	headerWriteBuf bytes.Buffer
	hpackEncoder   *hpack.Encoder
}

func (sc *serverConn) maxHeaderStringLen() int {
	v := sc.maxHeaderListSize()
	if uint32(int(v)) == v {
		return int(v)
	}
	// They had a crazy big number for MaxHeaderBytes anyway,
	// so give them unlimited header lengths:
	return 0
}

func (sc *serverConn) maxHeaderListSize() uint32 {
	n := sc.hs.MaxHeaderBytes
	if n <= 0 {
		n = http.DefaultMaxHeaderBytes
	}
	// http2's count is in a slightly different unit and includes 32 bytes per pair.
	// So, take the net/http.Server value and pad it up a bit, assuming 10 headers.
	const perFieldOverhead = 32 // per http2 spec
	const typicalHeaders = 10   // conservative
	return uint32(n + typicalHeaders*perFieldOverhead)
}

// requestParam is the state of the next request, initialized over
// potentially several frames HEADERS + zero or more CONTINUATION
// frames.
type requestParam struct {
	// stream is non-nil if we're reading (HEADER or CONTINUATION)
	// frames for a request (but not DATA).
	stream            *stream
	header            http.Header
	method, path      string
	scheme, authority string
	sawRegularHeader  bool  // saw a non-pseudo header already
	invalidHeader     bool  // an invalid header was seen
	headerListSize    int64 // actually uint32, but easier math this way
}

// stream represents a stream. This is the minimal metadata needed by
// the serve goroutine. Most of the actual stream state is owned by
// the http.Handler's goroutine in the responseWriter. Because the
// responseWriter's responseWriterState is recycled at the end of a
// handler, this struct intentionally has no pointer to the
// *responseWriter{,State} itself, as the Handler ending nils out the
// responseWriter's state field.
type stream struct {
	// immutable:
	sc   *serverConn
	id   uint32
	body *pipe       // non-nil if expecting DATA frames
	cw   closeWaiter // closed wait stream transitions to closed state

	// owned by serverConn's serve loop:
	bodyBytes        int64   // body bytes seen so far
	declBodyBytes    int64   // or -1 if undeclared
	flow             flow    // limits writing from Handler to client
	inflow           flow    // what the client is allowed to POST/etc to us
	parent           *stream // or nil
	numTrailerValues int64
	weight           uint8
	state            streamState
	sentReset        bool // only true once detached from streams map
	gotReset         bool // only true once detacted from streams map
	gotTrailerHeader bool // HEADER frame for trailers was seen

	trailer    http.Header // accumulated trailers
	reqTrailer http.Header // handler's Request.Trailer
}

func (sc *serverConn) Framer() *Framer  { return sc.framer }
func (sc *serverConn) CloseConn() error { return sc.conn.Close() }
func (sc *serverConn) Flush() error     { return sc.bw.Flush() }
func (sc *serverConn) HeaderEncoder() (*hpack.Encoder, *bytes.Buffer) {
	return sc.hpackEncoder, &sc.headerWriteBuf
}

func (sc *serverConn) state(streamID uint32) (streamState, *stream) {
	sc.serveG.check()
	// http://http2.github.io/http2-spec/#rfc.section.5.1
	if st, ok := sc.streams[streamID]; ok {
		return st.state, st
	}
	// "The first use of a new stream identifier implicitly closes all
	// streams in the "idle" state that might have been initiated by
	// that peer with a lower-valued stream identifier. For example, if
	// a client sends a HEADERS frame on stream 7 without ever sending a
	// frame on stream 5, then stream 5 transitions to the "closed"
	// state when the first frame for stream 7 is sent or received."
	if streamID <= sc.maxStreamID {
		return stateClosed, nil
	}
	return stateIdle, nil
}

// setConnState calls the net/http ConnState hook for this connection, if configured.
// Note that the net/http package does StateNew and StateClosed for us.
// There is currently no plan for StateHijacked or hijacking HTTP/2 connections.
func (sc *serverConn) setConnState(state http.ConnState) {
	if sc.hs.ConnState != nil {
		sc.hs.ConnState(sc.conn, state)
	}
}

func (sc *serverConn) vlogf(format string, args ...interface{}) {
	if VerboseLogs {
		sc.logf(format, args...)
	}
}

func (sc *serverConn) logf(format string, args ...interface{}) {
	if lg := sc.hs.ErrorLog; lg != nil {
		lg.Printf(format, args...)
	} else {
		log.Printf(format, args...)
	}
}

// errno returns v's underlying uintptr, else 0.
//
// TODO: remove this helper function once http2 can use build
// tags. See comment in isClosedConnError.
func errno(v error) uintptr {
	if rv := reflect.ValueOf(v); rv.Kind() == reflect.Uintptr {
		return uintptr(rv.Uint())
	}
	return 0
}

// isClosedConnError reports whether err is an error from use of a closed
// network connection.
func isClosedConnError(err error) bool {
	if err == nil {
		return false
	}

	// TODO: remove this string search and be more like the Windows
	// case below. That might involve modifying the standard library
	// to return better error types.
	str := err.Error()
	if strings.Contains(str, "use of closed network connection") {
		return true
	}

	// TODO(bradfitz): x/tools/cmd/bundle doesn't really support
	// build tags, so I can't make an http2_windows.go file with
	// Windows-specific stuff. Fix that and move this, once we
	// have a way to bundle this into std's net/http somehow.
	if runtime.GOOS == "windows" {
		if oe, ok := err.(*net.OpError); ok && oe.Op == "read" {
			if se, ok := oe.Err.(*os.SyscallError); ok && se.Syscall == "wsarecv" {
				const WSAECONNABORTED = 10053
				const WSAECONNRESET = 10054
				if n := errno(se.Err); n == WSAECONNRESET || n == WSAECONNABORTED {
					return true
				}
			}
		}
	}
	return false
}

func (sc *serverConn) condlogf(err error, format string, args ...interface{}) {
	if err == nil {
		return
	}
	if err == io.EOF || err == io.ErrUnexpectedEOF || isClosedConnError(err) {
		// Boring, expected errors.
		sc.vlogf(format, args...)
	} else {
		sc.logf(format, args...)
	}
}

func (sc *serverConn) onNewHeaderField(f hpack.HeaderField) {
	sc.serveG.check()
	if VerboseLogs {
		sc.vlogf("http2: server decoded %v", f)
	}
	switch {
	case !validHeaderFieldValue(f.Value): // f.Name checked _after_ pseudo check, since ':' is invalid
		sc.req.invalidHeader = true
	case strings.HasPrefix(f.Name, ":"):
		if sc.req.sawRegularHeader {
			sc.logf("pseudo-header after regular header")
			sc.req.invalidHeader = true
			return
		}
		var dst *string
		switch f.Name {
		case ":method":
			dst = &sc.req.method
		case ":path":
			dst = &sc.req.path
		case ":scheme":
			dst = &sc.req.scheme
		case ":authority":
			dst = &sc.req.authority
		default:
			// 8.1.2.1 Pseudo-Header Fields
			// "Endpoints MUST treat a request or response
			// that contains undefined or invalid
			// pseudo-header fields as malformed (Section
			// 8.1.2.6)."
			sc.logf("invalid pseudo-header %q", f.Name)
			sc.req.invalidHeader = true
			return
		}
		if *dst != "" {
			sc.logf("duplicate pseudo-header %q sent", f.Name)
			sc.req.invalidHeader = true
			return
		}
		*dst = f.Value
	case !validHeaderFieldName(f.Name):
		sc.req.invalidHeader = true
	default:
		sc.req.sawRegularHeader = true
		sc.req.header.Add(sc.canonicalHeader(f.Name), f.Value)
		const headerFieldOverhead = 32 // per spec
		sc.req.headerListSize += int64(len(f.Name)) + int64(len(f.Value)) + headerFieldOverhead
		if sc.req.headerListSize > int64(sc.maxHeaderListSize()) {
			sc.hpackDecoder.SetEmitEnabled(false)
		}
	}
}

func (st *stream) onNewTrailerField(f hpack.HeaderField) {
	sc := st.sc
	sc.serveG.check()
	if VerboseLogs {
		sc.vlogf("http2: server decoded trailer %v", f)
	}
	switch {
	case strings.HasPrefix(f.Name, ":"):
		sc.req.invalidHeader = true
		return
	case !validHeaderFieldName(f.Name) || !validHeaderFieldValue(f.Value):
		sc.req.invalidHeader = true
		return
	default:
		key := sc.canonicalHeader(f.Name)
		if st.trailer != nil {
			vv := append(st.trailer[key], f.Value)
			st.trailer[key] = vv

			// arbitrary; TODO: read spec about header list size limits wrt trailers
			const tooBig = 1000
			if len(vv) >= tooBig {
				sc.hpackDecoder.SetEmitEnabled(false)
			}
		}
	}
}

func (sc *serverConn) canonicalHeader(v string) string {
	sc.serveG.check()
	cv, ok := commonCanonHeader[v]
	if ok {
		return cv
	}
	cv, ok = sc.canonHeader[v]
	if ok {
		return cv
	}
	if sc.canonHeader == nil {
		sc.canonHeader = make(map[string]string)
	}
	cv = http.CanonicalHeaderKey(v)
	sc.canonHeader[v] = cv
	return cv
}

type readFrameResult struct {
	f   Frame // valid until readMore is called
	err error

	// readMore should be called once the consumer no longer needs or
	// retains f. After readMore, f is invalid and more frames can be
	// read.
	readMore func()
}

// readFrames is the loop that reads incoming frames.
// It takes care to only read one frame at a time, blocking until the
// consumer is done with the frame.
// It's run on its own goroutine.
func (sc *serverConn) readFrames() {
	gate := make(gate)
	for {
		f, err := sc.framer.ReadFrame()
		select {
		case sc.readFrameCh <- readFrameResult{f, err, gate.Done}:
		case <-sc.doneServing:
			return
		}
		select {
		case <-gate:
		case <-sc.doneServing:
			return
		}
		if terminalReadFrameError(err) {
			return
		}
	}
}

// frameWriteResult is the message passed from writeFrameAsync to the serve goroutine.
type frameWriteResult struct {
	wm  frameWriteMsg // what was written (or attempted)
	err error         // result of the writeFrame call
}

// writeFrameAsync runs in its own goroutine and writes a single frame
// and then reports when it's done.
// At most one goroutine can be running writeFrameAsync at a time per
// serverConn.
func (sc *serverConn) writeFrameAsync(wm frameWriteMsg) {
	err := wm.write.writeFrame(sc)
	sc.wroteFrameCh <- frameWriteResult{wm, err}
}

func (sc *serverConn) closeAllStreamsOnConnClose() {
	sc.serveG.check()
	for _, st := range sc.streams {
		sc.closeStream(st, errClientDisconnected)
	}
}

func (sc *serverConn) stopShutdownTimer() {
	sc.serveG.check()
	if t := sc.shutdownTimer; t != nil {
		t.Stop()
	}
}

func (sc *serverConn) notePanic() {
	// Note: this is for serverConn.serve panicking, not http.Handler code.
	if testHookOnPanicMu != nil {
		testHookOnPanicMu.Lock()
		defer testHookOnPanicMu.Unlock()
	}
	if testHookOnPanic != nil {
		if e := recover(); e != nil {
			if testHookOnPanic(sc, e) {
				panic(e)
			}
		}
	}
}

func (sc *serverConn) serve() {
	sc.serveG.check()
	defer sc.notePanic()
	defer sc.conn.Close()
	defer sc.closeAllStreamsOnConnClose()
	defer sc.stopShutdownTimer()
	defer close(sc.doneServing) // unblocks handlers trying to send

	if VerboseLogs {
		sc.vlogf("http2: server connection from %v on %p", sc.conn.RemoteAddr(), sc.hs)
	}

	sc.writeFrame(frameWriteMsg{
		write: writeSettings{
			{SettingMaxFrameSize, sc.srv.maxReadFrameSize()},
			{SettingMaxConcurrentStreams, sc.advMaxStreams},
			{SettingMaxHeaderListSize, sc.maxHeaderListSize()},

			// TODO: more actual settings, notably
			// SettingInitialWindowSize, but then we also
			// want to bump up the conn window size the
			// same amount here right after the settings
		},
	})
	sc.unackedSettings++

	if err := sc.readPreface(); err != nil {
		sc.condlogf(err, "http2: server: error reading preface from client %v: %v", sc.conn.RemoteAddr(), err)
		return
	}
	// Now that we've got the preface, get us out of the
	// "StateNew" state.  We can't go directly to idle, though.
	// Active means we read some data and anticipate a request. We'll
	// do another Active when we get a HEADERS frame.
	sc.setConnState(http.StateActive)
	sc.setConnState(http.StateIdle)

	go sc.readFrames() // closed by defer sc.conn.Close above

	settingsTimer := time.NewTimer(firstSettingsTimeout)
	loopNum := 0
	for {
		loopNum++
		select {
		case wm := <-sc.wantWriteFrameCh:
			sc.writeFrame(wm)
		case res := <-sc.wroteFrameCh:
			sc.wroteFrame(res)
		case res := <-sc.readFrameCh:
			if !sc.processFrameFromReader(res) {
				return
			}
			res.readMore()
			if settingsTimer.C != nil {
				settingsTimer.Stop()
				settingsTimer.C = nil
			}
		case m := <-sc.bodyReadCh:
			sc.noteBodyRead(m.st, m.n)
		case <-settingsTimer.C:
			sc.logf("timeout waiting for SETTINGS frames from %v", sc.conn.RemoteAddr())
			return
		case <-sc.shutdownTimerCh:
			sc.vlogf("GOAWAY close timer fired; closing conn from %v", sc.conn.RemoteAddr())
			return
		case fn := <-sc.testHookCh:
			fn(loopNum)
		}
	}
}

// readPreface reads the ClientPreface greeting from the peer
// or returns an error on timeout or an invalid greeting.
func (sc *serverConn) readPreface() error {
	errc := make(chan error, 1)
	go func() {
		// Read the client preface
		buf := make([]byte, len(ClientPreface))
		if _, err := io.ReadFull(sc.conn, buf); err != nil {
			errc <- err
		} else if !bytes.Equal(buf, clientPreface) {
			errc <- fmt.Errorf("bogus greeting %q", buf)
		} else {
			errc <- nil
		}
	}()
	timer := time.NewTimer(prefaceTimeout) // TODO: configurable on *Server?
	defer timer.Stop()
	select {
	case <-timer.C:
		return errors.New("timeout waiting for client preface")
	case err := <-errc:
		if err == nil {
			if VerboseLogs {
				sc.vlogf("http2: server: client %v said hello", sc.conn.RemoteAddr())
			}
		}
		return err
	}
}

var errChanPool = sync.Pool{
	New: func() interface{} { return make(chan error, 1) },
}

var writeDataPool = sync.Pool{
	New: func() interface{} { return new(writeData) },
}

// writeDataFromHandler writes DATA response frames from a handler on
// the given stream.
func (sc *serverConn) writeDataFromHandler(stream *stream, data []byte, endStream bool) error {
	ch := errChanPool.Get().(chan error)
	writeArg := writeDataPool.Get().(*writeData)
	*writeArg = writeData{stream.id, data, endStream}
	err := sc.writeFrameFromHandler(frameWriteMsg{
		write:  writeArg,
		stream: stream,
		done:   ch,
	})
	if err != nil {
		return err
	}
	var frameWriteDone bool // the frame write is done (successfully or not)
	select {
	case err = <-ch:
		frameWriteDone = true
	case <-sc.doneServing:
		return errClientDisconnected
	case <-stream.cw:
		// If both ch and stream.cw were ready (as might
		// happen on the final Write after an http.Handler
		// ends), prefer the write result. Otherwise this
		// might just be us successfully closing the stream.
		// The writeFrameAsync and serve goroutines guarantee
		// that the ch send will happen before the stream.cw
		// close.
		select {
		case err = <-ch:
			frameWriteDone = true
		default:
			return errStreamClosed
		}
	}
	errChanPool.Put(ch)
	if frameWriteDone {
		writeDataPool.Put(writeArg)
	}
	return err
}

// writeFrameFromHandler sends wm to sc.wantWriteFrameCh, but aborts
// if the connection has gone away.
//
// This must not be run from the serve goroutine itself, else it might
// deadlock writing to sc.wantWriteFrameCh (which is only mildly
// buffered and is read by serve itself). If you're on the serve
// goroutine, call writeFrame instead.
func (sc *serverConn) writeFrameFromHandler(wm frameWriteMsg) error {
	sc.serveG.checkNotOn() // NOT
	select {
	case sc.wantWriteFrameCh <- wm:
		return nil
	case <-sc.doneServing:
		// Serve loop is gone.
		// Client has closed their connection to the server.
		return errClientDisconnected
	}
}

// writeFrame schedules a frame to write and sends it if there's nothing
// already being written.
//
// There is no pushback here (the serve goroutine never blocks). It's
// the http.Handlers that block, waiting for their previous frames to
// make it onto the wire
//
// If you're not on the serve goroutine, use writeFrameFromHandler instead.
func (sc *serverConn) writeFrame(wm frameWriteMsg) {
	sc.serveG.check()
	sc.writeSched.add(wm)
	sc.scheduleFrameWrite()
}

// startFrameWrite starts a goroutine to write wm (in a separate
// goroutine since that might block on the network), and updates the
// serve goroutine's state about the world, updated from info in wm.
func (sc *serverConn) startFrameWrite(wm frameWriteMsg) {
	sc.serveG.check()
	if sc.writingFrame {
		panic("internal error: can only be writing one frame at a time")
	}

	st := wm.stream
	if st != nil {
		switch st.state {
		case stateHalfClosedLocal:
			panic("internal error: attempt to send frame on half-closed-local stream")
		case stateClosed:
			if st.sentReset || st.gotReset {
				// Skip this frame.
				sc.scheduleFrameWrite()
				return
			}
			panic(fmt.Sprintf("internal error: attempt to send a write %v on a closed stream", wm))
		}
	}

	sc.writingFrame = true
	sc.needsFrameFlush = true
	go sc.writeFrameAsync(wm)
}

// errHandlerPanicked is the error given to any callers blocked in a read from
// Request.Body when the main goroutine panics. Since most handlers read in the
// the main ServeHTTP goroutine, this will show up rarely.
var errHandlerPanicked = errors.New("http2: handler panicked")

// wroteFrame is called on the serve goroutine with the result of
// whatever happened on writeFrameAsync.
func (sc *serverConn) wroteFrame(res frameWriteResult) {
	sc.serveG.check()
	if !sc.writingFrame {
		panic("internal error: expected to be already writing a frame")
	}
	sc.writingFrame = false

	wm := res.wm
	st := wm.stream

	closeStream := endsStream(wm.write)

	if _, ok := wm.write.(handlerPanicRST); ok {
		sc.closeStream(st, errHandlerPanicked)
	}

	// Reply (if requested) to the blocked ServeHTTP goroutine.
	if ch := wm.done; ch != nil {
		select {
		case ch <- res.err:
		default:
			panic(fmt.Sprintf("unbuffered done channel passed in for type %T", wm.write))
		}
	}
	wm.write = nil // prevent use (assume it's tainted after wm.done send)

	if closeStream {
		if st == nil {
			panic("internal error: expecting non-nil stream")
		}
		switch st.state {
		case stateOpen:
			// Here we would go to stateHalfClosedLocal in
			// theory, but since our handler is done and
			// the net/http package provides no mechanism
			// for finishing writing to a ResponseWriter
			// while still reading data (see possible TODO
			// at top of this file), we go into closed
			// state here anyway, after telling the peer
			// we're hanging up on them.
			st.state = stateHalfClosedLocal // won't last long, but necessary for closeStream via resetStream
			errCancel := StreamError{st.id, ErrCodeCancel}
			sc.resetStream(errCancel)
		case stateHalfClosedRemote:
			sc.closeStream(st, errHandlerComplete)
		}
	}

	sc.scheduleFrameWrite()
}

// scheduleFrameWrite tickles the frame writing scheduler.
//
// If a frame is already being written, nothing happens. This will be called again
// when the frame is done being written.
//
// If a frame isn't being written we need to send one, the best frame
// to send is selected, preferring first things that aren't
// stream-specific (e.g. ACKing settings), and then finding the
// highest priority stream.
//
// If a frame isn't being written and there's nothing else to send, we
// flush the write buffer.
func (sc *serverConn) scheduleFrameWrite() {
	sc.serveG.check()
	if sc.writingFrame {
		return
	}
	if sc.needToSendGoAway {
		sc.needToSendGoAway = false
		sc.startFrameWrite(frameWriteMsg{
			write: &writeGoAway{
				maxStreamID: sc.maxStreamID,
				code:        sc.goAwayCode,
			},
		})
		return
	}
	if sc.needToSendSettingsAck {
		sc.needToSendSettingsAck = false
		sc.startFrameWrite(frameWriteMsg{write: writeSettingsAck{}})
		return
	}
	if !sc.inGoAway {
		if wm, ok := sc.writeSched.take(); ok {
			sc.startFrameWrite(wm)
			return
		}
	}
	if sc.needsFrameFlush {
		sc.startFrameWrite(frameWriteMsg{write: flushFrameWriter{}})
		sc.needsFrameFlush = false // after startFrameWrite, since it sets this true
		return
	}
}

func (sc *serverConn) goAway(code ErrCode) {
	sc.serveG.check()
	if sc.inGoAway {
		return
	}
	if code != ErrCodeNo {
		sc.shutDownIn(250 * time.Millisecond)
	} else {
		// TODO: configurable
		sc.shutDownIn(1 * time.Second)
	}
	sc.inGoAway = true
	sc.needToSendGoAway = true
	sc.goAwayCode = code
	sc.scheduleFrameWrite()
}

func (sc *serverConn) shutDownIn(d time.Duration) {
	sc.serveG.check()
	sc.shutdownTimer = time.NewTimer(d)
	sc.shutdownTimerCh = sc.shutdownTimer.C
}

func (sc *serverConn) resetStream(se StreamError) {
	sc.serveG.check()
	sc.writeFrame(frameWriteMsg{write: se})
	if st, ok := sc.streams[se.StreamID]; ok {
		st.sentReset = true
		sc.closeStream(st, se)
	}
}

// processFrameFromReader processes the serve loop's read from readFrameCh from the
// frame-reading goroutine.
// processFrameFromReader returns whether the connection should be kept open.
func (sc *serverConn) processFrameFromReader(res readFrameResult) bool {
	sc.serveG.check()
	err := res.err
	if err != nil {
		if err == ErrFrameTooLarge {
			sc.goAway(ErrCodeFrameSize)
			return true // goAway will close the loop
		}
		clientGone := err == io.EOF || err == io.ErrUnexpectedEOF || isClosedConnError(err)
		if clientGone {
			// TODO: could we also get into this state if
			// the peer does a half close
			// (e.g. CloseWrite) because they're done
			// sending frames but they're still wanting
			// our open replies?  Investigate.
			// TODO: add CloseWrite to crypto/tls.Conn first
			// so we have a way to test this? I suppose
			// just for testing we could have a non-TLS mode.
			return false
		}
	} else {
		f := res.f
		if VerboseLogs {
			sc.vlogf("http2: server read frame %v", summarizeFrame(f))
		}
		err = sc.processFrame(f)
		if err == nil {
			return true
		}
	}

	switch ev := err.(type) {
	case StreamError:
		sc.resetStream(ev)
		return true
	case goAwayFlowError:
		sc.goAway(ErrCodeFlowControl)
		return true
	case ConnectionError:
		sc.logf("http2: server connection error from %v: %v", sc.conn.RemoteAddr(), ev)
		sc.goAway(ErrCode(ev))
		return true // goAway will handle shutdown
	default:
		if res.err != nil {
			sc.vlogf("http2: server closing client connection; error reading frame from client %s: %v", sc.conn.RemoteAddr(), err)
		} else {
			sc.logf("http2: server closing client connection: %v", err)
		}
		return false
	}
}

func (sc *serverConn) processFrame(f Frame) error {
	sc.serveG.check()

	// First frame received must be SETTINGS.
	if !sc.sawFirstSettings {
		if _, ok := f.(*SettingsFrame); !ok {
			return ConnectionError(ErrCodeProtocol)
		}
		sc.sawFirstSettings = true
	}

	switch f := f.(type) {
	case *SettingsFrame:
		return sc.processSettings(f)
	case *HeadersFrame:
		return sc.processHeaders(f)
	case *ContinuationFrame:
		return sc.processContinuation(f)
	case *WindowUpdateFrame:
		return sc.processWindowUpdate(f)
	case *PingFrame:
		return sc.processPing(f)
	case *DataFrame:
		return sc.processData(f)
	case *RSTStreamFrame:
		return sc.processResetStream(f)
	case *PriorityFrame:
		return sc.processPriority(f)
	case *PushPromiseFrame:
		// A client cannot push. Thus, servers MUST treat the receipt of a PUSH_PROMISE
		// frame as a connection error (Section 5.4.1) of type PROTOCOL_ERROR.
		return ConnectionError(ErrCodeProtocol)
	default:
		sc.vlogf("http2: server ignoring frame: %v", f.Header())
		return nil
	}
}

func (sc *serverConn) processPing(f *PingFrame) error {
	sc.serveG.check()
	if f.IsAck() {
		// 6.7 PING: " An endpoint MUST NOT respond to PING frames
		// containing this flag."
		return nil
	}
	if f.StreamID != 0 {
		// "PING frames are not associated with any individual
		// stream. If a PING frame is received with a stream
		// identifier field value other than 0x0, the recipient MUST
		// respond with a connection error (Section 5.4.1) of type
		// PROTOCOL_ERROR."
		return ConnectionError(ErrCodeProtocol)
	}
	sc.writeFrame(frameWriteMsg{write: writePingAck{f}})
	return nil
}

func (sc *serverConn) processWindowUpdate(f *WindowUpdateFrame) error {
	sc.serveG.check()
	switch {
	case f.StreamID != 0: // stream-level flow control
		st := sc.streams[f.StreamID]
		if st == nil {
			// "WINDOW_UPDATE can be sent by a peer that has sent a
			// frame bearing the END_STREAM flag. This means that a
			// receiver could receive a WINDOW_UPDATE frame on a "half
			// closed (remote)" or "closed" stream. A receiver MUST
			// NOT treat this as an error, see Section 5.1."
			return nil
		}
		if !st.flow.add(int32(f.Increment)) {
			return StreamError{f.StreamID, ErrCodeFlowControl}
		}
	default: // connection-level flow control
		if !sc.flow.add(int32(f.Increment)) {
			return goAwayFlowError{}
		}
	}
	sc.scheduleFrameWrite()
	return nil
}

func (sc *serverConn) processResetStream(f *RSTStreamFrame) error {
	sc.serveG.check()

	state, st := sc.state(f.StreamID)
	if state == stateIdle {
		// 6.4 "RST_STREAM frames MUST NOT be sent for a
		// stream in the "idle" state. If a RST_STREAM frame
		// identifying an idle stream is received, the
		// recipient MUST treat this as a connection error
		// (Section 5.4.1) of type PROTOCOL_ERROR.
		return ConnectionError(ErrCodeProtocol)
	}
	if st != nil {
		st.gotReset = true
		sc.closeStream(st, StreamError{f.StreamID, f.ErrCode})
	}
	return nil
}

func (sc *serverConn) closeStream(st *stream, err error) {
	sc.serveG.check()
	if st.state == stateIdle || st.state == stateClosed {
		panic(fmt.Sprintf("invariant; can't close stream in state %v", st.state))
	}
	st.state = stateClosed
	sc.curOpenStreams--
	if sc.curOpenStreams == 0 {
		sc.setConnState(http.StateIdle)
	}
	delete(sc.streams, st.id)
	if p := st.body; p != nil {
		p.CloseWithError(err)
	}
	st.cw.Close() // signals Handler's CloseNotifier, unblocks writes, etc
	sc.writeSched.forgetStream(st.id)
}

func (sc *serverConn) processSettings(f *SettingsFrame) error {
	sc.serveG.check()
	if f.IsAck() {
		sc.unackedSettings--
		if sc.unackedSettings < 0 {
			// Why is the peer ACKing settings we never sent?
			// The spec doesn't mention this case, but
			// hang up on them anyway.
			return ConnectionError(ErrCodeProtocol)
		}
		return nil
	}
	if err := f.ForeachSetting(sc.processSetting); err != nil {
		return err
	}
	sc.needToSendSettingsAck = true
	sc.scheduleFrameWrite()
	return nil
}

func (sc *serverConn) processSetting(s Setting) error {
	sc.serveG.check()
	if err := s.Valid(); err != nil {
		return err
	}
	if VerboseLogs {
		sc.vlogf("http2: server processing setting %v", s)
	}
	switch s.ID {
	case SettingHeaderTableSize:
		sc.headerTableSize = s.Val
		sc.hpackEncoder.SetMaxDynamicTableSize(s.Val)
	case SettingEnablePush:
		sc.pushEnabled = s.Val != 0
	case SettingMaxConcurrentStreams:
		sc.clientMaxStreams = s.Val
	case SettingInitialWindowSize:
		return sc.processSettingInitialWindowSize(s.Val)
	case SettingMaxFrameSize:
		sc.writeSched.maxFrameSize = s.Val
	case SettingMaxHeaderListSize:
		sc.peerMaxHeaderListSize = s.Val
	default:
		// Unknown setting: "An endpoint that receives a SETTINGS
		// frame with any unknown or unsupported identifier MUST
		// ignore that setting."
		if VerboseLogs {
			sc.vlogf("http2: server ignoring unknown setting %v", s)
		}
	}
	return nil
}

func (sc *serverConn) processSettingInitialWindowSize(val uint32) error {
	sc.serveG.check()
	// Note: val already validated to be within range by
	// processSetting's Valid call.

	// "A SETTINGS frame can alter the initial flow control window
	// size for all current streams. When the value of
	// SETTINGS_INITIAL_WINDOW_SIZE changes, a receiver MUST
	// adjust the size of all stream flow control windows that it
	// maintains by the difference between the new value and the
	// old value."
	old := sc.initialWindowSize
	sc.initialWindowSize = int32(val)
	growth := sc.initialWindowSize - old // may be negative
	for _, st := range sc.streams {
		if !st.flow.add(growth) {
			// 6.9.2 Initial Flow Control Window Size
			// "An endpoint MUST treat a change to
			// SETTINGS_INITIAL_WINDOW_SIZE that causes any flow
			// control window to exceed the maximum size as a
			// connection error (Section 5.4.1) of type
			// FLOW_CONTROL_ERROR."
			return ConnectionError(ErrCodeFlowControl)
		}
	}
	return nil
}

func (sc *serverConn) processData(f *DataFrame) error {
	sc.serveG.check()
	// "If a DATA frame is received whose stream is not in "open"
	// or "half closed (local)" state, the recipient MUST respond
	// with a stream error (Section 5.4.2) of type STREAM_CLOSED."
	id := f.Header().StreamID
	st, ok := sc.streams[id]
	if !ok || st.state != stateOpen || st.gotTrailerHeader {
		// This includes sending a RST_STREAM if the stream is
		// in stateHalfClosedLocal (which currently means that
		// the http.Handler returned, so it's done reading &
		// done writing). Try to stop the client from sending
		// more DATA.
		return StreamError{id, ErrCodeStreamClosed}
	}
	if st.body == nil {
		panic("internal error: should have a body in this state")
	}
	data := f.Data()

	// Sender sending more than they'd declared?
	if st.declBodyBytes != -1 && st.bodyBytes+int64(len(data)) > st.declBodyBytes {
		st.body.CloseWithError(fmt.Errorf("sender tried to send more than declared Content-Length of %d bytes", st.declBodyBytes))
		return StreamError{id, ErrCodeStreamClosed}
	}
	if len(data) > 0 {
		// Check whether the client has flow control quota.
		if int(st.inflow.available()) < len(data) {
			return StreamError{id, ErrCodeFlowControl}
		}
		st.inflow.take(int32(len(data)))
		wrote, err := st.body.Write(data)
		if err != nil {
			return StreamError{id, ErrCodeStreamClosed}
		}
		if wrote != len(data) {
			panic("internal error: bad Writer")
		}
		st.bodyBytes += int64(len(data))
	}
	if f.StreamEnded() {
		st.endStream()
	}
	return nil
}

// endStream closes a Request.Body's pipe. It is called when a DATA
// frame says a request body is over (or after trailers).
func (st *stream) endStream() {
	sc := st.sc
	sc.serveG.check()

	if st.declBodyBytes != -1 && st.declBodyBytes != st.bodyBytes {
		st.body.CloseWithError(fmt.Errorf("request declared a Content-Length of %d but only wrote %d bytes",
			st.declBodyBytes, st.bodyBytes))
	} else {
		st.body.closeWithErrorAndCode(io.EOF, st.copyTrailersToHandlerRequest)
		st.body.CloseWithError(io.EOF)
	}
	st.state = stateHalfClosedRemote
}

// copyTrailersToHandlerRequest is run in the Handler's goroutine in
// its Request.Body.Read just before it gets io.EOF.
func (st *stream) copyTrailersToHandlerRequest() {
	for k, vv := range st.trailer {
		if _, ok := st.reqTrailer[k]; ok {
			// Only copy it over it was pre-declared.
			st.reqTrailer[k] = vv
		}
	}
}

func (sc *serverConn) processHeaders(f *HeadersFrame) error {
	sc.serveG.check()
	id := f.Header().StreamID
	if sc.inGoAway {
		// Ignore.
		return nil
	}
	// http://http2.github.io/http2-spec/#rfc.section.5.1.1
	// Streams initiated by a client MUST use odd-numbered stream
	// identifiers. [...] An endpoint that receives an unexpected
	// stream identifier MUST respond with a connection error
	// (Section 5.4.1) of type PROTOCOL_ERROR.
	if id%2 != 1 {
		return ConnectionError(ErrCodeProtocol)
	}
	// A HEADERS frame can be used to create a new stream or
	// send a trailer for an open one. If we already have a stream
	// open, let it process its own HEADERS frame (trailers at this
	// point, if it's valid).
	st := sc.streams[f.Header().StreamID]
	if st != nil {
		return st.processTrailerHeaders(f)
	}

	// [...] The identifier of a newly established stream MUST be
	// numerically greater than all streams that the initiating
	// endpoint has opened or reserved. [...]  An endpoint that
	// receives an unexpected stream identifier MUST respond with
	// a connection error (Section 5.4.1) of type PROTOCOL_ERROR.
	if id <= sc.maxStreamID || sc.req.stream != nil {
		return ConnectionError(ErrCodeProtocol)
	}

	if id > sc.maxStreamID {
		sc.maxStreamID = id
	}
	st = &stream{
		sc:    sc,
		id:    id,
		state: stateOpen,
	}
	if f.StreamEnded() {
		st.state = stateHalfClosedRemote
	}
	st.cw.Init()

	st.flow.conn = &sc.flow // link to conn-level counter
	st.flow.add(sc.initialWindowSize)
	st.inflow.conn = &sc.inflow      // link to conn-level counter
	st.inflow.add(initialWindowSize) // TODO: update this when we send a higher initial window size in the initial settings

	sc.streams[id] = st
	if f.HasPriority() {
		adjustStreamPriority(sc.streams, st.id, f.Priority)
	}
	sc.curOpenStreams++
	if sc.curOpenStreams == 1 {
		sc.setConnState(http.StateActive)
	}
	sc.req = requestParam{
		stream: st,
		header: make(http.Header),
	}
	sc.hpackDecoder.SetEmitFunc(sc.onNewHeaderField)
	sc.hpackDecoder.SetEmitEnabled(true)
	return sc.processHeaderBlockFragment(st, f.HeaderBlockFragment(), f.HeadersEnded())
}

func (st *stream) processTrailerHeaders(f *HeadersFrame) error {
	sc := st.sc
	sc.serveG.check()
	if st.gotTrailerHeader {
		return ConnectionError(ErrCodeProtocol)
	}
	st.gotTrailerHeader = true
	if !f.StreamEnded() {
		return StreamError{st.id, ErrCodeProtocol}
	}
	sc.resetPendingRequest() // we use invalidHeader from it for trailers
	return st.processTrailerHeaderBlockFragment(f.HeaderBlockFragment(), f.HeadersEnded())
}

func (sc *serverConn) processContinuation(f *ContinuationFrame) error {
	sc.serveG.check()
	st := sc.streams[f.Header().StreamID]
	if st.gotTrailerHeader {
		return st.processTrailerHeaderBlockFragment(f.HeaderBlockFragment(), f.HeadersEnded())
	}
	return sc.processHeaderBlockFragment(st, f.HeaderBlockFragment(), f.HeadersEnded())
}

func (sc *serverConn) processHeaderBlockFragment(st *stream, frag []byte, end bool) error {
	sc.serveG.check()
	if _, err := sc.hpackDecoder.Write(frag); err != nil {
		return ConnectionError(ErrCodeCompression)
	}
	if !end {
		return nil
	}
	if err := sc.hpackDecoder.Close(); err != nil {
		return ConnectionError(ErrCodeCompression)
	}
	defer sc.resetPendingRequest()
	if sc.curOpenStreams > sc.advMaxStreams {
		// "Endpoints MUST NOT exceed the limit set by their
		// peer. An endpoint that receives a HEADERS frame
		// that causes their advertised concurrent stream
		// limit to be exceeded MUST treat this as a stream
		// error (Section 5.4.2) of type PROTOCOL_ERROR or
		// REFUSED_STREAM."
		if sc.unackedSettings == 0 {
			// They should know better.
			return StreamError{st.id, ErrCodeProtocol}
		}
		// Assume it's a network race, where they just haven't
		// received our last SETTINGS update. But actually
		// this can't happen yet, because we don't yet provide
		// a way for users to adjust server parameters at
		// runtime.
		return StreamError{st.id, ErrCodeRefusedStream}
	}

	rw, req, err := sc.newWriterAndRequest()
	if err != nil {
		return err
	}
	st.reqTrailer = req.Trailer
	if st.reqTrailer != nil {
		st.trailer = make(http.Header)
	}
	st.body = req.Body.(*requestBody).pipe // may be nil
	st.declBodyBytes = req.ContentLength

	handler := sc.handler.ServeHTTP
	if !sc.hpackDecoder.EmitEnabled() {
		// Their header list was too long. Send a 431 error.
		handler = handleHeaderListTooLong
	}

	go sc.runHandler(rw, req, handler)
	return nil
}

func (st *stream) processTrailerHeaderBlockFragment(frag []byte, end bool) error {
	sc := st.sc
	sc.serveG.check()
	sc.hpackDecoder.SetEmitFunc(st.onNewTrailerField)
	if _, err := sc.hpackDecoder.Write(frag); err != nil {
		return ConnectionError(ErrCodeCompression)
	}
	if !end {
		return nil
	}

	rp := &sc.req
	if rp.invalidHeader {
		return StreamError{rp.stream.id, ErrCodeProtocol}
	}

	err := sc.hpackDecoder.Close()
	st.endStream()
	if err != nil {
		return ConnectionError(ErrCodeCompression)
	}
	return nil
}

func (sc *serverConn) processPriority(f *PriorityFrame) error {
	adjustStreamPriority(sc.streams, f.StreamID, f.PriorityParam)
	return nil
}

func adjustStreamPriority(streams map[uint32]*stream, streamID uint32, priority PriorityParam) {
	st, ok := streams[streamID]
	if !ok {
		// TODO: not quite correct (this streamID might
		// already exist in the dep tree, but be closed), but
		// close enough for now.
		return
	}
	st.weight = priority.Weight
	parent := streams[priority.StreamDep] // might be nil
	if parent == st {
		// if client tries to set this stream to be the parent of itself
		// ignore and keep going
		return
	}

	// section 5.3.3: If a stream is made dependent on one of its
	// own dependencies, the formerly dependent stream is first
	// moved to be dependent on the reprioritized stream's previous
	// parent. The moved dependency retains its weight.
	for piter := parent; piter != nil; piter = piter.parent {
		if piter == st {
			parent.parent = st.parent
			break
		}
	}
	st.parent = parent
	if priority.Exclusive && (st.parent != nil || priority.StreamDep == 0) {
		for _, openStream := range streams {
			if openStream != st && openStream.parent == st.parent {
				openStream.parent = st
			}
		}
	}
}

// resetPendingRequest zeros out all state related to a HEADERS frame
// and its zero or more CONTINUATION frames sent to start a new
// request.
func (sc *serverConn) resetPendingRequest() {
	sc.serveG.check()
	sc.req = requestParam{}
}

func (sc *serverConn) newWriterAndRequest() (*responseWriter, *http.Request, error) {
	sc.serveG.check()
	rp := &sc.req

	if rp.invalidHeader {
		return nil, nil, StreamError{rp.stream.id, ErrCodeProtocol}
	}

	isConnect := rp.method == "CONNECT"
	if isConnect {
		if rp.path != "" || rp.scheme != "" || rp.authority == "" {
			return nil, nil, StreamError{rp.stream.id, ErrCodeProtocol}
		}
	} else if rp.method == "" || rp.path == "" ||
		(rp.scheme != "https" && rp.scheme != "http") {
		// See 8.1.2.6 Malformed Requests and Responses:
		//
		// Malformed requests or responses that are detected
		// MUST be treated as a stream error (Section 5.4.2)
		// of type PROTOCOL_ERROR."
		//
		// 8.1.2.3 Request Pseudo-Header Fields
		// "All HTTP/2 requests MUST include exactly one valid
		// value for the :method, :scheme, and :path
		// pseudo-header fields"
		return nil, nil, StreamError{rp.stream.id, ErrCodeProtocol}
	}

	bodyOpen := rp.stream.state == stateOpen
	if rp.method == "HEAD" && bodyOpen {
		// HEAD requests can't have bodies
		return nil, nil, StreamError{rp.stream.id, ErrCodeProtocol}
	}
	var tlsState *tls.ConnectionState // nil if not scheme https

	if rp.scheme == "https" {
		tlsState = sc.tlsState
	}
	authority := rp.authority
	if authority == "" {
		authority = rp.header.Get("Host")
	}
	needsContinue := rp.header.Get("Expect") == "100-continue"
	if needsContinue {
		rp.header.Del("Expect")
	}
	// Merge Cookie headers into one "; "-delimited value.
	if cookies := rp.header["Cookie"]; len(cookies) > 1 {
		rp.header.Set("Cookie", strings.Join(cookies, "; "))
	}

	// Setup Trailers
	var trailer http.Header
	for _, v := range rp.header["Trailer"] {
		for _, key := range strings.Split(v, ",") {
			key = http.CanonicalHeaderKey(strings.TrimSpace(key))
			switch key {
			case "Transfer-Encoding", "Trailer", "Content-Length":
				// Bogus. (copy of http1 rules)
				// Ignore.
			default:
				if trailer == nil {
					trailer = make(http.Header)
				}
				trailer[key] = nil
			}
		}
	}
	delete(rp.header, "Trailer")

	body := &requestBody{
		conn:          sc,
		stream:        rp.stream,
		needsContinue: needsContinue,
	}
	var url_ *url.URL
	var requestURI string
	if isConnect {
		url_ = &url.URL{Host: rp.authority}
		requestURI = rp.authority // mimic HTTP/1 server behavior
	} else {
		var err error
		url_, err = url.ParseRequestURI(rp.path)
		if err != nil {
			return nil, nil, StreamError{rp.stream.id, ErrCodeProtocol}
		}
		requestURI = rp.path
	}
	req := &http.Request{
		Method:     rp.method,
		URL:        url_,
		RemoteAddr: sc.remoteAddrStr,
		Header:     rp.header,
		RequestURI: requestURI,
		Proto:      "HTTP/2.0",
		ProtoMajor: 2,
		ProtoMinor: 0,
		TLS:        tlsState,
		Host:       authority,
		Body:       body,
		Trailer:    trailer,
	}
	if bodyOpen {
		body.pipe = &pipe{
			b: &fixedBuffer{buf: make([]byte, initialWindowSize)}, // TODO: garbage
		}

		if vv, ok := rp.header["Content-Length"]; ok {
			req.ContentLength, _ = strconv.ParseInt(vv[0], 10, 64)
		} else {
			req.ContentLength = -1
		}
	}

	rws := responseWriterStatePool.Get().(*responseWriterState)
	bwSave := rws.bw
	*rws = responseWriterState{} // zero all the fields
	rws.conn = sc
	rws.bw = bwSave
	rws.bw.Reset(chunkWriter{rws})
	rws.stream = rp.stream
	rws.req = req
	rws.body = body

	rw := &responseWriter{rws: rws}
	return rw, req, nil
}

// Run on its own goroutine.
func (sc *serverConn) runHandler(rw *responseWriter, req *http.Request, handler func(http.ResponseWriter, *http.Request)) {
	didPanic := true
	defer func() {
		if didPanic {
			e := recover()
			// Same as net/http:
			const size = 64 << 10
			buf := make([]byte, size)
			buf = buf[:runtime.Stack(buf, false)]
			sc.writeFrameFromHandler(frameWriteMsg{
				write:  handlerPanicRST{rw.rws.stream.id},
				stream: rw.rws.stream,
			})
			sc.logf("http2: panic serving %v: %v\n%s", sc.conn.RemoteAddr(), e, buf)
			return
		}
		rw.handlerDone()
	}()
	handler(rw, req)
	didPanic = false
}

func handleHeaderListTooLong(w http.ResponseWriter, r *http.Request) {
	// 10.5.1 Limits on Header Block Size:
	// .. "A server that receives a larger header block than it is
	// willing to handle can send an HTTP 431 (Request Header Fields Too
	// Large) status code"
	const statusRequestHeaderFieldsTooLarge = 431 // only in Go 1.6+
	w.WriteHeader(statusRequestHeaderFieldsTooLarge)
	io.WriteString(w, "<h1>HTTP Error 431</h1><p>Request Header Field(s) Too Large</p>")
}

// called from handler goroutines.
// h may be nil.
func (sc *serverConn) writeHeaders(st *stream, headerData *writeResHeaders) error {
	sc.serveG.checkNotOn() // NOT on
	var errc chan error
	if headerData.h != nil {
		// If there's a header map (which we don't own), so we have to block on
		// waiting for this frame to be written, so an http.Flush mid-handler
		// writes out the correct value of keys, before a handler later potentially
		// mutates it.
		errc = errChanPool.Get().(chan error)
	}
	if err := sc.writeFrameFromHandler(frameWriteMsg{
		write:  headerData,
		stream: st,
		done:   errc,
	}); err != nil {
		return err
	}
	if errc != nil {
		select {
		case err := <-errc:
			errChanPool.Put(errc)
			return err
		case <-sc.doneServing:
			return errClientDisconnected
		case <-st.cw:
			return errStreamClosed
		}
	}
	return nil
}

// called from handler goroutines.
func (sc *serverConn) write100ContinueHeaders(st *stream) {
	sc.writeFrameFromHandler(frameWriteMsg{
		write:  write100ContinueHeadersFrame{st.id},
		stream: st,
	})
}

// A bodyReadMsg tells the server loop that the http.Handler read n
// bytes of the DATA from the client on the given stream.
type bodyReadMsg struct {
	st *stream
	n  int
}

// called from handler goroutines.
// Notes that the handler for the given stream ID read n bytes of its body
// and schedules flow control tokens to be sent.
func (sc *serverConn) noteBodyReadFromHandler(st *stream, n int) {
	sc.serveG.checkNotOn() // NOT on
	select {
	case sc.bodyReadCh <- bodyReadMsg{st, n}:
	case <-sc.doneServing:
	}
}

func (sc *serverConn) noteBodyRead(st *stream, n int) {
	sc.serveG.check()
	sc.sendWindowUpdate(nil, n) // conn-level
	if st.state != stateHalfClosedRemote && st.state != stateClosed {
		// Don't send this WINDOW_UPDATE if the stream is closed
		// remotely.
		sc.sendWindowUpdate(st, n)
	}
}

// st may be nil for conn-level
func (sc *serverConn) sendWindowUpdate(st *stream, n int) {
	sc.serveG.check()
	// "The legal range for the increment to the flow control
	// window is 1 to 2^31-1 (2,147,483,647) octets."
	// A Go Read call on 64-bit machines could in theory read
	// a larger Read than this. Very unlikely, but we handle it here
	// rather than elsewhere for now.
	const maxUint31 = 1<<31 - 1
	for n >= maxUint31 {
		sc.sendWindowUpdate32(st, maxUint31)
		n -= maxUint31
	}
	sc.sendWindowUpdate32(st, int32(n))
}

// st may be nil for conn-level
func (sc *serverConn) sendWindowUpdate32(st *stream, n int32) {
	sc.serveG.check()
	if n == 0 {
		return
	}
	if n < 0 {
		panic("negative update")
	}
	var streamID uint32
	if st != nil {
		streamID = st.id
	}
	sc.writeFrame(frameWriteMsg{
		write:  writeWindowUpdate{streamID: streamID, n: uint32(n)},
		stream: st,
	})
	var ok bool
	if st == nil {
		ok = sc.inflow.add(n)
	} else {
		ok = st.inflow.add(n)
	}
	if !ok {
		panic("internal error; sent too many window updates without decrements?")
	}
}

type requestBody struct {
	stream        *stream
	conn          *serverConn
	closed        bool
	pipe          *pipe // non-nil if we have a HTTP entity message body
	needsContinue bool  // need to send a 100-continue
}

func (b *requestBody) Close() error {
	if b.pipe != nil {
		b.pipe.CloseWithError(errClosedBody)
	}
	b.closed = true
	return nil
}

func (b *requestBody) Read(p []byte) (n int, err error) {
	if b.needsContinue {
		b.needsContinue = false
		b.conn.write100ContinueHeaders(b.stream)
	}
	if b.pipe == nil {
		return 0, io.EOF
	}
	n, err = b.pipe.Read(p)
	if n > 0 {
		b.conn.noteBodyReadFromHandler(b.stream, n)
	}
	return
}

// responseWriter is the http.ResponseWriter implementation.  It's
// intentionally small (1 pointer wide) to minimize garbage.  The
// responseWriterState pointer inside is zeroed at the end of a
// request (in handlerDone) and calls on the responseWriter thereafter
// simply crash (caller's mistake), but the much larger responseWriterState
// and buffers are reused between multiple requests.
type responseWriter struct {
	rws *responseWriterState
}

// Optional http.ResponseWriter interfaces implemented.
var (
	_ http.CloseNotifier = (*responseWriter)(nil)
	_ http.Flusher       = (*responseWriter)(nil)
	_ stringWriter       = (*responseWriter)(nil)
)

type responseWriterState struct {
	// immutable within a request:
	stream *stream
	req    *http.Request
	body   *requestBody // to close at end of request, if DATA frames didn't
	conn   *serverConn

	// TODO: adjust buffer writing sizes based on server config, frame size updates from peer, etc
	bw *bufio.Writer // writing to a chunkWriter{this *responseWriterState}

	// mutated by http.Handler goroutine:
	handlerHeader http.Header // nil until called
	snapHeader    http.Header // snapshot of handlerHeader at WriteHeader time
	trailers      []string    // set in writeChunk
	status        int         // status code passed to WriteHeader
	wroteHeader   bool        // WriteHeader called (explicitly or implicitly). Not necessarily sent to user yet.
	sentHeader    bool        // have we sent the header frame?
	handlerDone   bool        // handler has finished

	sentContentLen int64 // non-zero if handler set a Content-Length header
	wroteBytes     int64

	closeNotifierMu sync.Mutex // guards closeNotifierCh
	closeNotifierCh chan bool  // nil until first used
}

type chunkWriter struct{ rws *responseWriterState }

func (cw chunkWriter) Write(p []byte) (n int, err error) { return cw.rws.writeChunk(p) }

func (rws *responseWriterState) hasTrailers() bool { return len(rws.trailers) != 0 }

// declareTrailer is called for each Trailer header when the
// response header is written. It notes that a header will need to be
// written in the trailers at the end of the response.
func (rws *responseWriterState) declareTrailer(k string) {
	k = http.CanonicalHeaderKey(k)
	switch k {
	case "Transfer-Encoding", "Content-Length", "Trailer":
		// Forbidden by RFC 2616 14.40.
		return
	}
	if !strSliceContains(rws.trailers, k) {
		rws.trailers = append(rws.trailers, k)
	}
}

// writeChunk writes chunks from the bufio.Writer. But because
// bufio.Writer may bypass its chunking, sometimes p may be
// arbitrarily large.
//
// writeChunk is also responsible (on the first chunk) for sending the
// HEADER response.
func (rws *responseWriterState) writeChunk(p []byte) (n int, err error) {
	if !rws.wroteHeader {
		rws.writeHeader(200)
	}

	isHeadResp := rws.req.Method == "HEAD"
	if !rws.sentHeader {
		rws.sentHeader = true
		var ctype, clen string
		if clen = rws.snapHeader.Get("Content-Length"); clen != "" {
			rws.snapHeader.Del("Content-Length")
			clen64, err := strconv.ParseInt(clen, 10, 64)
			if err == nil && clen64 >= 0 {
				rws.sentContentLen = clen64
			} else {
				clen = ""
			}
		}
		if clen == "" && rws.handlerDone && bodyAllowedForStatus(rws.status) && (len(p) > 0 || !isHeadResp) {
			clen = strconv.Itoa(len(p))
		}
		_, hasContentType := rws.snapHeader["Content-Type"]
		if !hasContentType && bodyAllowedForStatus(rws.status) {
			ctype = http.DetectContentType(p)
		}
		var date string
		if _, ok := rws.snapHeader["Date"]; !ok {
			// TODO(bradfitz): be faster here, like net/http? measure.
			date = time.Now().UTC().Format(http.TimeFormat)
		}

		for _, v := range rws.snapHeader["Trailer"] {
			foreachHeaderElement(v, rws.declareTrailer)
		}

		endStream := (rws.handlerDone && !rws.hasTrailers() && len(p) == 0) || isHeadResp
		err = rws.conn.writeHeaders(rws.stream, &writeResHeaders{
			streamID:      rws.stream.id,
			httpResCode:   rws.status,
			h:             rws.snapHeader,
			endStream:     endStream,
			contentType:   ctype,
			contentLength: clen,
			date:          date,
		})
		if err != nil {
			return 0, err
		}
		if endStream {
			return 0, nil
		}
	}
	if isHeadResp {
		return len(p), nil
	}
	if len(p) == 0 && !rws.handlerDone {
		return 0, nil
	}

	if rws.handlerDone {
		rws.promoteUndeclaredTrailers()
	}

	endStream := rws.handlerDone && !rws.hasTrailers()
	if len(p) > 0 || endStream {
		// only send a 0 byte DATA frame if we're ending the stream.
		if err := rws.conn.writeDataFromHandler(rws.stream, p, endStream); err != nil {
			return 0, err
		}
	}

	if rws.handlerDone && rws.hasTrailers() {
		err = rws.conn.writeHeaders(rws.stream, &writeResHeaders{
			streamID:  rws.stream.id,
			h:         rws.handlerHeader,
			trailers:  rws.trailers,
			endStream: true,
		})
		return len(p), err
	}
	return len(p), nil
}

// TrailerPrefix is a magic prefix for ResponseWriter.Header map keys
// that, if present, signals that the map entry is actually for
// the response trailers, and not the response headers. The prefix
// is stripped after the ServeHTTP call finishes and the values are
// sent in the trailers.
//
// This mechanism is intended only for trailers that are not known
// prior to the headers being written. If the set of trailers is fixed
// or known before the header is written, the normal Go trailers mechanism
// is preferred:
//    https://golang.org/pkg/net/http/#ResponseWriter
//    https://golang.org/pkg/net/http/#example_ResponseWriter_trailers
const TrailerPrefix = "Trailer:"

// promoteUndeclaredTrailers permits http.Handlers to set trailers
// after the header has already been flushed. Because the Go
// ResponseWriter interface has no way to set Trailers (only the
// Header), and because we didn't want to expand the ResponseWriter
// interface, and because nobody used trailers, and because RFC 2616
// says you SHOULD (but not must) predeclare any trailers in the
// header, the official ResponseWriter rules said trailers in Go must
// be predeclared, and then we reuse the same ResponseWriter.Header()
// map to mean both Headers and Trailers.  When it's time to write the
// Trailers, we pick out the fields of Headers that were declared as
// trailers. That worked for a while, until we found the first major
// user of Trailers in the wild: gRPC (using them only over http2),
// and gRPC libraries permit setting trailers mid-stream without
// predeclarnig them. So: change of plans. We still permit the old
// way, but we also permit this hack: if a Header() key begins with
// "Trailer:", the suffix of that key is a Trailer. Because ':' is an
// invalid token byte anyway, there is no ambiguity. (And it's already
// filtered out) It's mildly hacky, but not terrible.
//
// This method runs after the Handler is done and promotes any Header
// fields to be trailers.
func (rws *responseWriterState) promoteUndeclaredTrailers() {
	for k, vv := range rws.handlerHeader {
		if !strings.HasPrefix(k, TrailerPrefix) {
			continue
		}
		trailerKey := strings.TrimPrefix(k, TrailerPrefix)
		rws.declareTrailer(trailerKey)
		rws.handlerHeader[http.CanonicalHeaderKey(trailerKey)] = vv
	}
	sort.Strings(rws.trailers)
}

func (w *responseWriter) Flush() {
	rws := w.rws
	if rws == nil {
		panic("Header called after Handler finished")
	}
	if rws.bw.Buffered() > 0 {
		if err := rws.bw.Flush(); err != nil {
			// Ignore the error. The frame writer already knows.
			return
		}
	} else {
		// The bufio.Writer won't call chunkWriter.Write
		// (writeChunk with zero bytes, so we have to do it
		// ourselves to force the HTTP response header and/or
		// final DATA frame (with END_STREAM) to be sent.
		rws.writeChunk(nil)
	}
}

func (w *responseWriter) CloseNotify() <-chan bool {
	rws := w.rws
	if rws == nil {
		panic("CloseNotify called after Handler finished")
	}
	rws.closeNotifierMu.Lock()
	ch := rws.closeNotifierCh
	if ch == nil {
		ch = make(chan bool, 1)
		rws.closeNotifierCh = ch
		go func() {
			rws.stream.cw.Wait() // wait for close
			ch <- true
		}()
	}
	rws.closeNotifierMu.Unlock()
	return ch
}

func (w *responseWriter) Header() http.Header {
	rws := w.rws
	if rws == nil {
		panic("Header called after Handler finished")
	}
	if rws.handlerHeader == nil {
		rws.handlerHeader = make(http.Header)
	}
	return rws.handlerHeader
}

func (w *responseWriter) WriteHeader(code int) {
	rws := w.rws
	if rws == nil {
		panic("WriteHeader called after Handler finished")
	}
	rws.writeHeader(code)
}

func (rws *responseWriterState) writeHeader(code int) {
	if !rws.wroteHeader {
		rws.wroteHeader = true
		rws.status = code
		if len(rws.handlerHeader) > 0 {
			rws.snapHeader = cloneHeader(rws.handlerHeader)
		}
	}
}

func cloneHeader(h http.Header) http.Header {
	h2 := make(http.Header, len(h))
	for k, vv := range h {
		vv2 := make([]string, len(vv))
		copy(vv2, vv)
		h2[k] = vv2
	}
	return h2
}

// The Life Of A Write is like this:
//
// * Handler calls w.Write or w.WriteString ->
// * -> rws.bw (*bufio.Writer) ->
// * (Handler migth call Flush)
// * -> chunkWriter{rws}
// * -> responseWriterState.writeChunk(p []byte)
// * -> responseWriterState.writeChunk (most of the magic; see comment there)
func (w *responseWriter) Write(p []byte) (n int, err error) {
	return w.write(len(p), p, "")
}

func (w *responseWriter) WriteString(s string) (n int, err error) {
	return w.write(len(s), nil, s)
}

// either dataB or dataS is non-zero.
func (w *responseWriter) write(lenData int, dataB []byte, dataS string) (n int, err error) {
	rws := w.rws
	if rws == nil {
		panic("Write called after Handler finished")
	}
	if !rws.wroteHeader {
		w.WriteHeader(200)
	}
	if !bodyAllowedForStatus(rws.status) {
		return 0, http.ErrBodyNotAllowed
	}
	rws.wroteBytes += int64(len(dataB)) + int64(len(dataS)) // only one can be set
	if rws.sentContentLen != 0 && rws.wroteBytes > rws.sentContentLen {
		// TODO: send a RST_STREAM
		return 0, errors.New("http2: handler wrote more than declared Content-Length")
	}

	if dataB != nil {
		return rws.bw.Write(dataB)
	} else {
		return rws.bw.WriteString(dataS)
	}
}

func (w *responseWriter) handlerDone() {
	rws := w.rws
	rws.handlerDone = true
	w.Flush()
	w.rws = nil
	responseWriterStatePool.Put(rws)
}

// foreachHeaderElement splits v according to the "#rule" construction
// in RFC 2616 section 2.1 and calls fn for each non-empty element.
func foreachHeaderElement(v string, fn func(string)) {
	v = textproto.TrimString(v)
	if v == "" {
		return
	}
	if !strings.Contains(v, ",") {
		fn(v)
		return
	}
	for _, f := range strings.Split(v, ",") {
		if f = textproto.TrimString(f); f != "" {
			fn(f)
		}
	}
}
