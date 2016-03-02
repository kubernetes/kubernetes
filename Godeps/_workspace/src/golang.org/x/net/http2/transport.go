// Copyright 2015 The Go Authors.
// See https://go.googlesource.com/go/+/master/CONTRIBUTORS
// Licensed under the same terms as Go itself:
// https://go.googlesource.com/go/+/master/LICENSE

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
	"strconv"
	"strings"
	"sync"

	"golang.org/x/net/http2/hpack"
)

type Transport struct {
	Fallback http.RoundTripper

	// TODO: remove this and make more general with a TLS dial hook, like http
	InsecureTLSDial bool

	connMu sync.Mutex
	conns  map[string][]*clientConn // key is host:port
}

type clientConn struct {
	t        *Transport
	tconn    *tls.Conn
	tlsState *tls.ConnectionState
	connKey  []string // key(s) this connection is cached in, in t.conns

	readerDone chan struct{} // closed on error
	readerErr  error         // set before readerDone is closed
	hdec       *hpack.Decoder
	nextRes    *http.Response

	mu           sync.Mutex
	closed       bool
	goAway       *GoAwayFrame // if non-nil, the GoAwayFrame we received
	streams      map[uint32]*clientStream
	nextStreamID uint32
	bw           *bufio.Writer
	werr         error // first write error that has occurred
	br           *bufio.Reader
	fr           *Framer
	// Settings from peer:
	maxFrameSize         uint32
	maxConcurrentStreams uint32
	initialWindowSize    uint32
	hbuf                 bytes.Buffer // HPACK encoder writes into this
	henc                 *hpack.Encoder
}

type clientStream struct {
	ID   uint32
	resc chan resAndError
	pw   *io.PipeWriter
	pr   *io.PipeReader
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

func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.Scheme != "https" {
		if t.Fallback == nil {
			return nil, errors.New("http2: unsupported scheme and no Fallback")
		}
		return t.Fallback.RoundTrip(req)
	}

	host, port, err := net.SplitHostPort(req.URL.Host)
	if err != nil {
		host = req.URL.Host
		port = "443"
	}

	for {
		cc, err := t.getClientConn(host, port)
		if err != nil {
			return nil, err
		}
		res, err := cc.roundTrip(req)
		if shouldRetryRequest(err) { // TODO: or clientconn is overloaded (too many outstanding requests)?
			continue
		}
		if err != nil {
			return nil, err
		}
		return res, nil
	}
}

// CloseIdleConnections closes any connections which were previously
// connected from previous requests but are now sitting idle.
// It does not interrupt any connections currently in use.
func (t *Transport) CloseIdleConnections() {
	t.connMu.Lock()
	defer t.connMu.Unlock()
	for _, vv := range t.conns {
		for _, cc := range vv {
			cc.closeIfIdle()
		}
	}
}

var errClientConnClosed = errors.New("http2: client conn is closed")

func shouldRetryRequest(err error) bool {
	// TODO: or GOAWAY graceful shutdown stuff
	return err == errClientConnClosed
}

func (t *Transport) removeClientConn(cc *clientConn) {
	t.connMu.Lock()
	defer t.connMu.Unlock()
	for _, key := range cc.connKey {
		vv, ok := t.conns[key]
		if !ok {
			continue
		}
		newList := filterOutClientConn(vv, cc)
		if len(newList) > 0 {
			t.conns[key] = newList
		} else {
			delete(t.conns, key)
		}
	}
}

func filterOutClientConn(in []*clientConn, exclude *clientConn) []*clientConn {
	out := in[:0]
	for _, v := range in {
		if v != exclude {
			out = append(out, v)
		}
	}
	return out
}

func (t *Transport) getClientConn(host, port string) (*clientConn, error) {
	t.connMu.Lock()
	defer t.connMu.Unlock()

	key := net.JoinHostPort(host, port)

	for _, cc := range t.conns[key] {
		if cc.canTakeNewRequest() {
			return cc, nil
		}
	}
	if t.conns == nil {
		t.conns = make(map[string][]*clientConn)
	}
	cc, err := t.newClientConn(host, port, key)
	if err != nil {
		return nil, err
	}
	t.conns[key] = append(t.conns[key], cc)
	return cc, nil
}

func (t *Transport) newClientConn(host, port, key string) (*clientConn, error) {
	cfg := &tls.Config{
		ServerName:         host,
		NextProtos:         []string{NextProtoTLS},
		InsecureSkipVerify: t.InsecureTLSDial,
	}
	tconn, err := tls.Dial("tcp", net.JoinHostPort(host, port), cfg)
	if err != nil {
		return nil, err
	}
	if err := tconn.Handshake(); err != nil {
		return nil, err
	}
	if !t.InsecureTLSDial {
		if err := tconn.VerifyHostname(cfg.ServerName); err != nil {
			return nil, err
		}
	}
	state := tconn.ConnectionState()
	if p := state.NegotiatedProtocol; p != NextProtoTLS {
		// TODO(bradfitz): fall back to Fallback
		return nil, fmt.Errorf("bad protocol: %v", p)
	}
	if !state.NegotiatedProtocolIsMutual {
		return nil, errors.New("could not negotiate protocol mutually")
	}
	if _, err := tconn.Write(clientPreface); err != nil {
		return nil, err
	}

	cc := &clientConn{
		t:                    t,
		tconn:                tconn,
		connKey:              []string{key}, // TODO: cert's validated hostnames too
		tlsState:             &state,
		readerDone:           make(chan struct{}),
		nextStreamID:         1,
		maxFrameSize:         16 << 10, // spec default
		initialWindowSize:    65535,    // spec default
		maxConcurrentStreams: 1000,     // "infinite", per spec. 1000 seems good enough.
		streams:              make(map[uint32]*clientStream),
	}
	cc.bw = bufio.NewWriter(stickyErrWriter{tconn, &cc.werr})
	cc.br = bufio.NewReader(tconn)
	cc.fr = NewFramer(cc.bw, cc.br)
	cc.henc = hpack.NewEncoder(&cc.hbuf)

	cc.fr.WriteSettings()
	// TODO: re-send more conn-level flow control tokens when server uses all these.
	cc.fr.WriteWindowUpdate(0, 1<<30) // um, 0x7fffffff doesn't work to Google? it hangs?
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
			// TODO(bradfitz): handle more
			log.Printf("Unhandled Setting: %v", s)
		}
		return nil
	})
	// TODO: figure out henc size
	cc.hdec = hpack.NewDecoder(initialHeaderTableSize, cc.onNewHeaderField)

	go cc.readLoop()
	return cc, nil
}

func (cc *clientConn) setGoAway(f *GoAwayFrame) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.goAway = f
}

func (cc *clientConn) canTakeNewRequest() bool {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.goAway == nil &&
		int64(len(cc.streams)+1) < int64(cc.maxConcurrentStreams) &&
		cc.nextStreamID < 2147483647
}

func (cc *clientConn) closeIfIdle() {
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

func (cc *clientConn) roundTrip(req *http.Request) (*http.Response, error) {
	cc.mu.Lock()

	if cc.closed {
		cc.mu.Unlock()
		return nil, errClientConnClosed
	}

	cs := cc.newStream()
	hasBody := false // TODO

	// we send: HEADERS[+CONTINUATION] + (DATA?)
	hdrs := cc.encodeHeaders(req)
	first := true
	for len(hdrs) > 0 {
		chunk := hdrs
		if len(chunk) > int(cc.maxFrameSize) {
			chunk = chunk[:cc.maxFrameSize]
		}
		hdrs = hdrs[len(chunk):]
		endHeaders := len(hdrs) == 0
		if first {
			cc.fr.WriteHeaders(HeadersFrameParam{
				StreamID:      cs.ID,
				BlockFragment: chunk,
				EndStream:     !hasBody,
				EndHeaders:    endHeaders,
			})
			first = false
		} else {
			cc.fr.WriteContinuation(cs.ID, endHeaders, chunk)
		}
	}
	cc.bw.Flush()
	werr := cc.werr
	cc.mu.Unlock()

	if hasBody {
		// TODO: write data. and it should probably be interleaved:
		//   go ... io.Copy(dataFrameWriter{cc, cs, ...}, req.Body) ... etc
	}

	if werr != nil {
		return nil, werr
	}

	re := <-cs.resc
	if re.err != nil {
		return nil, re.err
	}
	res := re.res
	res.Request = req
	res.TLS = cc.tlsState
	return res, nil
}

// requires cc.mu be held.
func (cc *clientConn) encodeHeaders(req *http.Request) []byte {
	cc.hbuf.Reset()

	// TODO(bradfitz): figure out :authority-vs-Host stuff between http2 and Go
	host := req.Host
	if host == "" {
		host = req.URL.Host
	}

	path := req.URL.Path
	if path == "" {
		path = "/"
	}

	cc.writeHeader(":authority", host) // probably not right for all sites
	cc.writeHeader(":method", req.Method)
	cc.writeHeader(":path", path)
	cc.writeHeader(":scheme", "https")

	for k, vv := range req.Header {
		lowKey := strings.ToLower(k)
		if lowKey == "host" {
			continue
		}
		for _, v := range vv {
			cc.writeHeader(lowKey, v)
		}
	}
	return cc.hbuf.Bytes()
}

func (cc *clientConn) writeHeader(name, value string) {
	log.Printf("sending %q = %q", name, value)
	cc.henc.WriteField(hpack.HeaderField{Name: name, Value: value})
}

type resAndError struct {
	res *http.Response
	err error
}

// requires cc.mu be held.
func (cc *clientConn) newStream() *clientStream {
	cs := &clientStream{
		ID:   cc.nextStreamID,
		resc: make(chan resAndError, 1),
	}
	cc.nextStreamID += 2
	cc.streams[cs.ID] = cs
	return cs
}

func (cc *clientConn) streamByID(id uint32, andRemove bool) *clientStream {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cs := cc.streams[id]
	if andRemove {
		delete(cc.streams, id)
	}
	return cs
}

// runs in its own goroutine.
func (cc *clientConn) readLoop() {
	defer cc.t.removeClientConn(cc)
	defer close(cc.readerDone)

	activeRes := map[uint32]*clientStream{} // keyed by streamID
	// Close any response bodies if the server closes prematurely.
	// TODO: also do this if we've written the headers but not
	// gotten a response yet.
	defer func() {
		err := cc.readerErr
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		for _, cs := range activeRes {
			cs.pw.CloseWithError(err)
		}
	}()

	// continueStreamID is the stream ID we're waiting for
	// continuation frames for.
	var continueStreamID uint32

	for {
		f, err := cc.fr.ReadFrame()
		if err != nil {
			cc.readerErr = err
			return
		}
		log.Printf("Transport received %v: %#v", f.Header(), f)

		streamID := f.Header().StreamID

		_, isContinue := f.(*ContinuationFrame)
		if isContinue {
			if streamID != continueStreamID {
				log.Printf("Protocol violation: got CONTINUATION with id %d; want %d", streamID, continueStreamID)
				cc.readerErr = ConnectionError(ErrCodeProtocol)
				return
			}
		} else if continueStreamID != 0 {
			// Continue frames need to be adjacent in the stream
			// and we were in the middle of headers.
			log.Printf("Protocol violation: got %T for stream %d, want CONTINUATION for %d", f, streamID, continueStreamID)
			cc.readerErr = ConnectionError(ErrCodeProtocol)
			return
		}

		if streamID%2 == 0 {
			// Ignore streams pushed from the server for now.
			// These always have an even stream id.
			continue
		}
		streamEnded := false
		if ff, ok := f.(streamEnder); ok {
			streamEnded = ff.StreamEnded()
		}

		cs := cc.streamByID(streamID, streamEnded)
		if cs == nil {
			log.Printf("Received frame for untracked stream ID %d", streamID)
			continue
		}

		switch f := f.(type) {
		case *HeadersFrame:
			cc.nextRes = &http.Response{
				Proto:      "HTTP/2.0",
				ProtoMajor: 2,
				Header:     make(http.Header),
			}
			cs.pr, cs.pw = io.Pipe()
			cc.hdec.Write(f.HeaderBlockFragment())
		case *ContinuationFrame:
			cc.hdec.Write(f.HeaderBlockFragment())
		case *DataFrame:
			log.Printf("DATA: %q", f.Data())
			cs.pw.Write(f.Data())
		case *GoAwayFrame:
			cc.t.removeClientConn(cc)
			if f.ErrCode != 0 {
				// TODO: deal with GOAWAY more. particularly the error code
				log.Printf("transport got GOAWAY with error code = %v", f.ErrCode)
			}
			cc.setGoAway(f)
		default:
			log.Printf("Transport: unhandled response frame type %T", f)
		}
		headersEnded := false
		if he, ok := f.(headersEnder); ok {
			headersEnded = he.HeadersEnded()
			if headersEnded {
				continueStreamID = 0
			} else {
				continueStreamID = streamID
			}
		}

		if streamEnded {
			cs.pw.Close()
			delete(activeRes, streamID)
		}
		if headersEnded {
			if cs == nil {
				panic("couldn't find stream") // TODO be graceful
			}
			// TODO: set the Body to one which notes the
			// Close and also sends the server a
			// RST_STREAM
			cc.nextRes.Body = cs.pr
			res := cc.nextRes
			activeRes[streamID] = cs
			cs.resc <- resAndError{res: res}
		}
	}
}

func (cc *clientConn) onNewHeaderField(f hpack.HeaderField) {
	// TODO: verifiy pseudo headers come before non-pseudo headers
	// TODO: verifiy the status is set
	log.Printf("Header field: %+v", f)
	if f.Name == ":status" {
		code, err := strconv.Atoi(f.Value)
		if err != nil {
			panic("TODO: be graceful")
		}
		cc.nextRes.Status = f.Value + " " + http.StatusText(code)
		cc.nextRes.StatusCode = code
		return
	}
	if strings.HasPrefix(f.Name, ":") {
		// "Endpoints MUST NOT generate pseudo-header fields other than those defined in this document."
		// TODO: treat as invalid?
		return
	}
	cc.nextRes.Header.Add(http.CanonicalHeaderKey(f.Name), f.Value)
}
