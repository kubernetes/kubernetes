package messenger

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"net"
	"net/http"
	"net/textproto"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	log "github.com/golang/glog"
)

const (
	DefaultReadTimeout  = 5 * time.Second
	DefaultWriteTimeout = 5 * time.Second

	// writeFlushPeriod is the amount of time we're willing to wait for a single
	// response buffer to be fully written to the underlying TCP connection; after
	// this amount of time the remaining bytes of the response are discarded. see
	// responseWriter().
	writeFlushPeriod = 30 * time.Second
)

type decoderID int32

func (did decoderID) String() string {
	return "[" + strconv.Itoa(int(did)) + "]"
}

func (did *decoderID) next() decoderID {
	return decoderID(atomic.AddInt32((*int32)(did), 1))
}

var (
	errHijackFailed = errors.New("failed to hijack http connection")
	did             decoderID // decoder ID counter
	closedChan      = make(chan struct{})
)

func init() {
	close(closedChan)
}

type Decoder interface {
	Requests() <-chan *Request
	Err() <-chan error
	Cancel(bool)
}

type Request struct {
	*http.Request
	response chan<- Response // callers that are finished with a Request should ensure that response is *always* closed, regardless of whether a Response has been written.
}

type Response struct {
	code   int
	reason string
}

type httpDecoder struct {
	req          *http.Request // original request
	kalive       bool          // keepalive
	chunked      bool          // chunked
	msg          chan *Request
	con          net.Conn
	rw           *bufio.ReadWriter
	errCh        chan error
	buf          *bytes.Buffer
	lrc          *io.LimitedReader
	shouldQuit   chan struct{} // signal chan, closes upon calls to Cancel(...)
	forceQuit    chan struct{} // signal chan, indicates that quit is NOT graceful; closes upon Cancel(false)
	cancelGuard  sync.Mutex
	readTimeout  time.Duration
	writeTimeout time.Duration
	idtag        string             // useful for debugging
	sendError    func(err error)    // abstraction for error handling
	outCh        chan *bytes.Buffer // chan of responses to be written to the connection
}

// DecodeHTTP hijacks an HTTP server connection and generates mesos libprocess HTTP
// requests via the returned chan. Upon generation of an error in the error chan the
// decoder's internal goroutine will terminate. This func returns immediately.
// The caller should immediately *stop* using the ResponseWriter and Request that were
// passed as parameters; the decoder assumes full control of the HTTP transport.
func DecodeHTTP(w http.ResponseWriter, r *http.Request) Decoder {
	id := did.next()
	d := &httpDecoder{
		msg:          make(chan *Request),
		errCh:        make(chan error, 1),
		req:          r,
		shouldQuit:   make(chan struct{}),
		forceQuit:    make(chan struct{}),
		readTimeout:  DefaultReadTimeout,
		writeTimeout: DefaultWriteTimeout,
		idtag:        id.String(),
		outCh:        make(chan *bytes.Buffer),
	}
	d.sendError = d.defaultSendError
	go d.run(w)
	return d
}

func (d *httpDecoder) Requests() <-chan *Request {
	return d.msg
}

func (d *httpDecoder) Err() <-chan error {
	return d.errCh
}

// Cancel the decoding process; if graceful then process pending responses before terminating
func (d *httpDecoder) Cancel(graceful bool) {
	log.V(2).Infof("%scancel:%t", d.idtag, graceful)
	d.cancelGuard.Lock()
	defer d.cancelGuard.Unlock()
	select {
	case <-d.shouldQuit:
		// already quitting, but perhaps gracefully?
	default:
		close(d.shouldQuit)
	}
	// allow caller to "upgrade" from a graceful cancel to a forced one
	if !graceful {
		select {
		case <-d.forceQuit:
			// already forcefully quitting
		default:
			close(d.forceQuit) // push it!
		}
	}
}

func (d *httpDecoder) run(res http.ResponseWriter) {
	defer func() {
		close(d.outCh) // we're finished generating response objects
		log.V(2).Infoln(d.idtag + "run: terminating")
	}()

	for state := d.bootstrapState(res); state != nil; {
		next := state(d)
		state = next
	}
}

// tryFlushResponse flushes the response buffer (if not empty); returns true if flush succeeded
func (d *httpDecoder) tryFlushResponse(out *bytes.Buffer) {
	log.V(2).Infof(d.idtag+"try-flush-responses: %d bytes to flush", out.Len())
	// set a write deadline here so that we don't block for very long.
	err := d.setWriteTimeout()
	if err != nil {
		// this is a problem because if we can't set the timeout then we can't guarantee
		// how long a write op might block for. Log the error and skip this response.
		log.Errorln("failed to set write deadline, aborting response:", err.Error())
	} else {
		_, err = out.WriteTo(d.rw.Writer)
		if err != nil {
			if neterr, ok := err.(net.Error); ok && neterr.Timeout() && out.Len() > 0 {
				// we couldn't fully write before timing out, return rch and hope that
				// we have better luck next time.
				return
			}
			// we don't really know how to deal with other kinds of errors, so
			// log it and skip the rest of the response.
			log.Errorln("failed to write response buffer:", err.Error())
		}
		err = d.rw.Flush()
		if err != nil {
			if neterr, ok := err.(net.Error); ok && neterr.Timeout() && out.Len() > 0 {
				return
			}
			log.Errorln("failed to flush response buffer:", err.Error())
		}
	}
}

// TODO(jdef) make this a func on Response, to write its contents to a *bytes.Buffer
func (d *httpDecoder) buildResponseEntity(resp *Response) *bytes.Buffer {
	log.V(2).Infoln(d.idtag + "build-response-entity")

	out := &bytes.Buffer{}

	// generate new response buffer content and continue; buffer should have
	// at least a response status-line w/ Content-Length: 0
	out.WriteString("HTTP/1.1 ")
	out.WriteString(strconv.Itoa(resp.code))
	out.WriteString(" ")
	out.WriteString(resp.reason)
	out.WriteString(crlf + "Content-Length: 0" + crlf)

	select {
	case <-d.shouldQuit:
		// this is the last request in the pipeline and we've been told to quit, so
		// indicate that the server will close the connection.
		out.WriteString("Connection: Close" + crlf)
	default:
	}
	out.WriteString(crlf) // this ends the HTTP response entity
	return out
}

// updateForRequest updates the chunked and kalive fields of the decoder to align
// with the header values of the request
func (d *httpDecoder) updateForRequest() {
	// check "Transfer-Encoding" for "chunked"
	d.chunked = false
	for _, v := range d.req.Header["Transfer-Encoding"] {
		if v == "chunked" {
			d.chunked = true
			break
		}
	}
	if !d.chunked && d.req.ContentLength < 0 {
		// strongly suspect that Go's internal net/http lib is stripping
		// the Transfer-Encoding header from the initial request, so this
		// workaround makes a very mesos-specific assumption: an unknown
		// Content-Length indicates a chunked stream.
		d.chunked = true
	}

	// check "Connection" for "Keep-Alive"
	d.kalive = d.req.Header.Get("Connection") == "Keep-Alive"

	log.V(2).Infof(d.idtag+"update-for-request: chunked %v keep-alive %v", d.chunked, d.kalive)
}

func (d *httpDecoder) readBodyContent() httpState {
	log.V(2).Info(d.idtag + "read-body-content")
	if d.chunked {
		d.buf = &bytes.Buffer{}
		return readChunkHeaderState
	} else {
		d.lrc = limit(d.rw.Reader, d.req.ContentLength)
		d.buf = &bytes.Buffer{}
		return readBodyState
	}
}

const http202response = "HTTP/1.1 202 OK\r\nContent-Length: 0\r\n\r\n"

func (d *httpDecoder) generateRequest() httpState {
	log.V(2).Infof(d.idtag + "generate-request")
	// send a Request to msg
	b := d.buf.Bytes()
	rch := make(chan Response, 1)
	r := &Request{
		Request: &http.Request{
			Method:        d.req.Method,
			URL:           d.req.URL,
			Proto:         d.req.Proto,
			ProtoMajor:    d.req.ProtoMajor,
			ProtoMinor:    d.req.ProtoMinor,
			Header:        d.req.Header,
			Close:         !d.kalive,
			Host:          d.req.Host,
			RequestURI:    d.req.RequestURI,
			Body:          &body{bytes.NewBuffer(b)},
			ContentLength: int64(len(b)),
		},
		response: rch,
	}

	select {
	case d.msg <- r:
	case <-d.forceQuit:
		return terminateState
	}

	select {
	case <-d.forceQuit:
		return terminateState
	case resp, ok := <-rch:
		if ok {
			// response required, so build it and ship it
			out := d.buildResponseEntity(&resp)
			select {
			case <-d.forceQuit:
				return terminateState
			case d.outCh <- out:
			}
		}
	}

	if d.kalive {
		d.req = &http.Request{
			ContentLength: -1,
			Header:        make(http.Header),
		}
		return awaitRequestState
	} else {
		return gracefulTerminateState
	}
}

func (d *httpDecoder) defaultSendError(err error) {
	d.errCh <- err
}

type httpState func(d *httpDecoder) httpState

// terminateState forcefully shuts down the state machine
func terminateState(d *httpDecoder) httpState {
	log.V(2).Infoln(d.idtag + "terminate-state")
	// closing these chans tells Decoder users that it's wrapping up
	close(d.msg)
	close(d.errCh)

	// attempt to forcefully close the connection and signal response handlers that
	// no further responses should be written
	d.Cancel(false)

	if d.con != nil {
		d.con.Close()
	}

	// there is no spoon
	return nil
}

func gracefulTerminateState(d *httpDecoder) httpState {
	log.V(2).Infoln(d.idtag + "gracefully-terminate-state")
	// closing these chans tells Decoder users that it's wrapping up
	close(d.msg)
	close(d.errCh)

	// gracefully terminate the connection; signal that we should flush pending
	// responses before closing the connection.
	d.Cancel(true)

	return nil
}

func limit(r *bufio.Reader, limit int64) *io.LimitedReader {
	return &io.LimitedReader{
		R: r,
		N: limit,
	}
}

// bootstrapState expects to be called when the standard net/http lib has already
// read the initial request query line + headers from a connection. the request
// is ready to be hijacked at this point.
func (d *httpDecoder) bootstrapState(res http.ResponseWriter) httpState {
	log.V(2).Infoln(d.idtag + "bootstrap-state")

	d.updateForRequest()

	// hijack
	hj, ok := res.(http.Hijacker)
	if !ok {
		http.Error(res, "server does not support hijack", http.StatusInternalServerError)
		d.sendError(errHijackFailed)
		return terminateState
	}
	c, rw, err := hj.Hijack()
	if err != nil {
		http.Error(res, "failed to hijack the connection", http.StatusInternalServerError)
		d.sendError(errHijackFailed)
		return terminateState
	}

	d.rw = rw
	d.con = c

	go d.responseWriter()
	return d.readBodyContent()
}

func (d *httpDecoder) responseWriter() {
	defer func() {
		log.V(3).Infoln(d.idtag + "response-writer: closing connection")
		d.con.Close()
	}()
	for buf := range d.outCh {
		//TODO(jdef) I worry about this busy-looping

		// write & flush the buffer until there's nothing left in it, or else
		// we exceed the write/flush period.
		now := time.Now()
		for buf.Len() > 0 && time.Since(now) < writeFlushPeriod {
			select {
			case <-d.forceQuit:
				return
			default:
			}
			d.tryFlushResponse(buf)
		}
		if buf.Len() > 0 {
			//TODO(jdef) should we abort the entire connection instead? a partially written
			// response doesn't do anyone any good. That said, real libprocess agents don't
			// really care about the response channel anyway - the entire system is fire and
			// forget. So I've decided to err on the side that we might lose response bytes
			// in favor of completely reading the connection request stream before we terminate.
			log.Errorln(d.idtag + "failed to fully flush output buffer within write-flush period")
		}
	}
}

type body struct {
	*bytes.Buffer
}

func (b *body) Close() error { return nil }

// checkTimeoutOrFail tests whether the given error is related to a timeout condition.
// returns true if the caller should advance to the returned state.
func (d *httpDecoder) checkTimeoutOrFail(err error, stateContinue httpState) (httpState, bool) {
	if err != nil {
		if neterr, ok := err.(net.Error); ok && neterr.Timeout() {
			select {
			case <-d.forceQuit:
				return terminateState, true
			case <-d.shouldQuit:
				return gracefulTerminateState, true
			default:
				return stateContinue, true
			}
		}
		d.sendError(err)
		return terminateState, true
	}
	return nil, false
}

func (d *httpDecoder) setReadTimeoutOrFail() bool {
	if d.readTimeout > 0 {
		err := d.con.SetReadDeadline(time.Now().Add(d.readTimeout))
		if err != nil {
			d.sendError(err)
			return false
		}
	}
	return true
}

func (d *httpDecoder) setWriteTimeout() error {
	if d.writeTimeout > 0 {
		return d.con.SetWriteDeadline(time.Now().Add(d.writeTimeout))
	}
	return nil
}

func readChunkHeaderState(d *httpDecoder) httpState {
	log.V(2).Infoln(d.idtag + "read-chunk-header-state")
	tr := textproto.NewReader(d.rw.Reader)
	if !d.setReadTimeoutOrFail() {
		return terminateState
	}
	hexlen, err := tr.ReadLine()
	if next, ok := d.checkTimeoutOrFail(err, readChunkHeaderState); ok {
		return next
	}

	clen, err := strconv.ParseInt(hexlen, 16, 64)
	if err != nil {
		d.sendError(err)
		return terminateState
	}

	if clen == 0 {
		return readEndOfChunkStreamState
	}

	d.lrc = limit(d.rw.Reader, clen)
	return readChunkState
}

func readChunkState(d *httpDecoder) httpState {
	log.V(2).Infoln(d.idtag+"read-chunk-state, bytes remaining:", d.lrc.N)
	if !d.setReadTimeoutOrFail() {
		return terminateState
	}
	_, err := d.buf.ReadFrom(d.lrc)
	if next, ok := d.checkTimeoutOrFail(err, readChunkState); ok {
		return next
	}
	return readEndOfChunkState
}

const crlf = "\r\n"

func readEndOfChunkState(d *httpDecoder) httpState {
	log.V(2).Infoln(d.idtag + "read-end-of-chunk-state")
	if !d.setReadTimeoutOrFail() {
		return terminateState
	}
	b, err := d.rw.Reader.Peek(2)
	if len(b) == 2 {
		if string(b) == crlf {
			d.rw.ReadByte()
			d.rw.ReadByte()
			return readChunkHeaderState
		}
		d.sendError(errors.New(d.idtag + "unexpected data at end-of-chunk marker"))
		return terminateState
	}
	// less than two bytes avail
	if next, ok := d.checkTimeoutOrFail(err, readEndOfChunkState); ok {
		return next
	}
	panic("couldn't peek 2 bytes, but didn't get an error?!")
}

func readEndOfChunkStreamState(d *httpDecoder) httpState {
	log.V(2).Infoln(d.idtag + "read-end-of-chunk-stream-state")
	if !d.setReadTimeoutOrFail() {
		return terminateState
	}
	b, err := d.rw.Reader.Peek(2)
	if len(b) == 2 {
		if string(b) == crlf {
			d.rw.ReadByte()
			d.rw.ReadByte()
			return d.generateRequest()
		}
		d.sendError(errors.New(d.idtag + "unexpected data at end-of-chunk marker"))
		return terminateState
	}
	// less than 2 bytes avail
	if next, ok := d.checkTimeoutOrFail(err, readEndOfChunkStreamState); ok {
		return next
	}
	panic("couldn't peek 2 bytes, but didn't get an error?!")
}

func readBodyState(d *httpDecoder) httpState {
	log.V(2).Infof(d.idtag+"read-body-state: %d bytes remaining", d.lrc.N)
	// read remaining bytes into the buffer
	var err error
	if d.lrc.N > 0 {
		if !d.setReadTimeoutOrFail() {
			return terminateState
		}
		_, err = d.buf.ReadFrom(d.lrc)
	}
	if d.lrc.N <= 0 {
		return d.generateRequest()
	}
	if next, ok := d.checkTimeoutOrFail(err, readBodyState); ok {
		return next
	}
	return readBodyState
}

func isGracefulTermSignal(err error) bool {
	if err == io.EOF {
		return true
	}
	if operr, ok := err.(*net.OpError); ok {
		return operr.Op == "read" && err == syscall.ECONNRESET
	}
	return false
}

func awaitRequestState(d *httpDecoder) httpState {
	log.V(2).Infoln(d.idtag + "await-request-state")
	tr := textproto.NewReader(d.rw.Reader)
	if !d.setReadTimeoutOrFail() {
		return terminateState
	}
	requestLine, err := tr.ReadLine()
	if requestLine == "" && isGracefulTermSignal(err) {
		// we're actually expecting this at some point, so don't react poorly
		return gracefulTerminateState
	}
	if next, ok := d.checkTimeoutOrFail(err, awaitRequestState); ok {
		return next
	}
	ss := strings.SplitN(requestLine, " ", 3)
	if len(ss) < 3 {
		if err == io.EOF {
			return gracefulTerminateState
		}
		d.sendError(errors.New(d.idtag + "illegal request line"))
		return terminateState
	}
	r := d.req
	r.Method = ss[0]
	r.RequestURI = ss[1]
	r.URL, err = url.ParseRequestURI(ss[1])
	if err != nil {
		d.sendError(err)
		return terminateState
	}
	major, minor, ok := http.ParseHTTPVersion(ss[2])
	if !ok {
		d.sendError(errors.New(d.idtag + "malformed HTTP version"))
		return terminateState
	}
	r.ProtoMajor = major
	r.ProtoMinor = minor
	r.Proto = ss[2]
	return readHeaderState
}

func readHeaderState(d *httpDecoder) httpState {
	log.V(2).Infoln(d.idtag + "read-header-state")
	if !d.setReadTimeoutOrFail() {
		return terminateState
	}
	r := d.req
	tr := textproto.NewReader(d.rw.Reader)
	h, err := tr.ReadMIMEHeader()
	// merge any headers that were read successfully (before a possible error)
	for k, v := range h {
		if rh, exists := r.Header[k]; exists {
			r.Header[k] = append(rh, v...)
		} else {
			r.Header[k] = v
		}
		log.V(2).Infoln(d.idtag+"request header", k, v)
	}
	if next, ok := d.checkTimeoutOrFail(err, readHeaderState); ok {
		return next
	}

	// special headers: Host, Content-Length, Transfer-Encoding
	r.Host = r.Header.Get("Host")
	r.TransferEncoding = r.Header["Transfer-Encoding"]
	if cl := r.Header.Get("Content-Length"); cl != "" {
		l, err := strconv.ParseInt(cl, 10, 64)
		if err != nil {
			d.sendError(err)
			return terminateState
		}
		if l > -1 {
			r.ContentLength = l
			log.V(2).Infoln(d.idtag+"set content length", r.ContentLength)
		}
	}
	d.updateForRequest()
	return d.readBodyContent()
}
