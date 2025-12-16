package websocket // import "github.com/coder/websocket"

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall/js"

	"github.com/coder/websocket/internal/bpool"
	"github.com/coder/websocket/internal/wsjs"
)

// opcode represents a WebSocket opcode.
type opcode int

// https://tools.ietf.org/html/rfc6455#section-11.8.
const (
	opContinuation opcode = iota
	opText
	opBinary
	// 3 - 7 are reserved for further non-control frames.
	_
	_
	_
	_
	_
	opClose
	opPing
	opPong
	// 11-16 are reserved for further control frames.
)

// Conn provides a wrapper around the browser WebSocket API.
type Conn struct {
	noCopy noCopy
	ws     wsjs.WebSocket

	// read limit for a message in bytes.
	msgReadLimit atomic.Int64

	closeReadMu  sync.Mutex
	closeReadCtx context.Context

	closingMu     sync.Mutex
	closeOnce     sync.Once
	closed        chan struct{}
	closeErrOnce  sync.Once
	closeErr      error
	closeWasClean bool

	releaseOnClose   func()
	releaseOnError   func()
	releaseOnMessage func()

	readSignal chan struct{}
	readBufMu  sync.Mutex
	readBuf    []wsjs.MessageEvent
}

func (c *Conn) close(err error, wasClean bool) {
	c.closeOnce.Do(func() {
		runtime.SetFinalizer(c, nil)

		if !wasClean {
			err = fmt.Errorf("unclean connection close: %w", err)
		}
		c.setCloseErr(err)
		c.closeWasClean = wasClean
		close(c.closed)
	})
}

func (c *Conn) init() {
	c.closed = make(chan struct{})
	c.readSignal = make(chan struct{}, 1)

	c.msgReadLimit.Store(32768)

	c.releaseOnClose = c.ws.OnClose(func(e wsjs.CloseEvent) {
		err := CloseError{
			Code:   StatusCode(e.Code),
			Reason: e.Reason,
		}
		// We do not know if we sent or received this close as
		// its possible the browser triggered it without us
		// explicitly sending it.
		c.close(err, e.WasClean)

		c.releaseOnClose()
		c.releaseOnError()
		c.releaseOnMessage()
	})

	c.releaseOnError = c.ws.OnError(func(v js.Value) {
		c.setCloseErr(errors.New(v.Get("message").String()))
		c.closeWithInternal()
	})

	c.releaseOnMessage = c.ws.OnMessage(func(e wsjs.MessageEvent) {
		c.readBufMu.Lock()
		defer c.readBufMu.Unlock()

		c.readBuf = append(c.readBuf, e)

		// Lets the read goroutine know there is definitely something in readBuf.
		select {
		case c.readSignal <- struct{}{}:
		default:
		}
	})

	runtime.SetFinalizer(c, func(c *Conn) {
		c.setCloseErr(errors.New("connection garbage collected"))
		c.closeWithInternal()
	})
}

func (c *Conn) closeWithInternal() {
	c.Close(StatusInternalError, "something went wrong")
}

// Read attempts to read a message from the connection.
// The maximum time spent waiting is bounded by the context.
func (c *Conn) Read(ctx context.Context) (MessageType, []byte, error) {
	c.closeReadMu.Lock()
	closedRead := c.closeReadCtx != nil
	c.closeReadMu.Unlock()
	if closedRead {
		return 0, nil, errors.New("WebSocket connection read closed")
	}

	typ, p, err := c.read(ctx)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to read: %w", err)
	}
	readLimit := c.msgReadLimit.Load()
	if readLimit >= 0 && int64(len(p)) > readLimit {
		reason := fmt.Errorf("read limited at %d bytes", c.msgReadLimit.Load())
		c.Close(StatusMessageTooBig, reason.Error())
		return 0, nil, fmt.Errorf("%w: %v", ErrMessageTooBig, reason)
	}
	return typ, p, nil
}

func (c *Conn) read(ctx context.Context) (MessageType, []byte, error) {
	select {
	case <-ctx.Done():
		c.Close(StatusPolicyViolation, "read timed out")
		return 0, nil, ctx.Err()
	case <-c.readSignal:
	case <-c.closed:
		return 0, nil, net.ErrClosed
	}

	c.readBufMu.Lock()
	defer c.readBufMu.Unlock()

	me := c.readBuf[0]
	// We copy the messages forward and decrease the size
	// of the slice to avoid reallocating.
	copy(c.readBuf, c.readBuf[1:])
	c.readBuf = c.readBuf[:len(c.readBuf)-1]

	if len(c.readBuf) > 0 {
		// Next time we read, we'll grab the message.
		select {
		case c.readSignal <- struct{}{}:
		default:
		}
	}

	switch p := me.Data.(type) {
	case string:
		return MessageText, []byte(p), nil
	case []byte:
		return MessageBinary, p, nil
	default:
		panic("websocket: unexpected data type from wsjs OnMessage: " + reflect.TypeOf(me.Data).String())
	}
}

// Ping is mocked out for Wasm.
func (c *Conn) Ping(ctx context.Context) error {
	return nil
}

// Write writes a message of the given type to the connection.
// Always non blocking.
func (c *Conn) Write(ctx context.Context, typ MessageType, p []byte) error {
	err := c.write(typ, p)
	if err != nil {
		// Have to ensure the WebSocket is closed after a write error
		// to match the Go API. It can only error if the message type
		// is unexpected or the passed bytes contain invalid UTF-8 for
		// MessageText.
		err := fmt.Errorf("failed to write: %w", err)
		c.setCloseErr(err)
		c.closeWithInternal()
		return err
	}
	return nil
}

func (c *Conn) write(typ MessageType, p []byte) error {
	if c.isClosed() {
		return net.ErrClosed
	}
	switch typ {
	case MessageBinary:
		return c.ws.SendBytes(p)
	case MessageText:
		return c.ws.SendText(string(p))
	default:
		return fmt.Errorf("unexpected message type: %v", typ)
	}
}

// Close closes the WebSocket with the given code and reason.
// It will wait until the peer responds with a close frame
// or the connection is closed.
// It thus performs the full WebSocket close handshake.
func (c *Conn) Close(code StatusCode, reason string) error {
	err := c.exportedClose(code, reason)
	if err != nil {
		return fmt.Errorf("failed to close WebSocket: %w", err)
	}
	return nil
}

// CloseNow closes the WebSocket connection without attempting a close handshake.
// Use when you do not want the overhead of the close handshake.
//
// note: No different from Close(StatusGoingAway, "") in WASM as there is no way to close
// a WebSocket without the close handshake.
func (c *Conn) CloseNow() error {
	return c.Close(StatusGoingAway, "")
}

func (c *Conn) exportedClose(code StatusCode, reason string) error {
	c.closingMu.Lock()
	defer c.closingMu.Unlock()

	if c.isClosed() {
		return net.ErrClosed
	}

	ce := fmt.Errorf("sent close: %w", CloseError{
		Code:   code,
		Reason: reason,
	})

	c.setCloseErr(ce)
	err := c.ws.Close(int(code), reason)
	if err != nil {
		return err
	}

	<-c.closed
	if !c.closeWasClean {
		return c.closeErr
	}
	return nil
}

// Subprotocol returns the negotiated subprotocol.
// An empty string means the default protocol.
func (c *Conn) Subprotocol() string {
	return c.ws.Subprotocol()
}

// DialOptions represents the options available to pass to Dial.
type DialOptions struct {
	// Subprotocols lists the subprotocols to negotiate with the server.
	Subprotocols []string
}

// Dial creates a new WebSocket connection to the given url with the given options.
// The passed context bounds the maximum time spent waiting for the connection to open.
// The returned *http.Response is always nil or a mock. It's only in the signature
// to match the core API.
func Dial(ctx context.Context, url string, opts *DialOptions) (*Conn, *http.Response, error) {
	c, resp, err := dial(ctx, url, opts)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to WebSocket dial %q: %w", url, err)
	}
	return c, resp, nil
}

func dial(ctx context.Context, url string, opts *DialOptions) (*Conn, *http.Response, error) {
	if opts == nil {
		opts = &DialOptions{}
	}

	url = strings.Replace(url, "http://", "ws://", 1)
	url = strings.Replace(url, "https://", "wss://", 1)

	ws, err := wsjs.New(url, opts.Subprotocols)
	if err != nil {
		return nil, nil, err
	}

	c := &Conn{
		ws: ws,
	}
	c.init()

	opench := make(chan struct{})
	releaseOpen := ws.OnOpen(func(e js.Value) {
		close(opench)
	})
	defer releaseOpen()

	select {
	case <-ctx.Done():
		c.Close(StatusPolicyViolation, "dial timed out")
		return nil, nil, ctx.Err()
	case <-opench:
		return c, &http.Response{
			StatusCode: http.StatusSwitchingProtocols,
		}, nil
	case <-c.closed:
		return nil, nil, net.ErrClosed
	}
}

// Reader attempts to read a message from the connection.
// The maximum time spent waiting is bounded by the context.
func (c *Conn) Reader(ctx context.Context) (MessageType, io.Reader, error) {
	typ, p, err := c.Read(ctx)
	if err != nil {
		return 0, nil, err
	}
	return typ, bytes.NewReader(p), nil
}

// Writer returns a writer to write a WebSocket data message to the connection.
// It buffers the entire message in memory and then sends it when the writer
// is closed.
func (c *Conn) Writer(ctx context.Context, typ MessageType) (io.WriteCloser, error) {
	return &writer{
		c:   c,
		ctx: ctx,
		typ: typ,
		b:   bpool.Get(),
	}, nil
}

type writer struct {
	closed bool

	c   *Conn
	ctx context.Context
	typ MessageType

	b *bytes.Buffer
}

func (w *writer) Write(p []byte) (int, error) {
	if w.closed {
		return 0, errors.New("cannot write to closed writer")
	}
	n, err := w.b.Write(p)
	if err != nil {
		return n, fmt.Errorf("failed to write message: %w", err)
	}
	return n, nil
}

func (w *writer) Close() error {
	if w.closed {
		return errors.New("cannot close closed writer")
	}
	w.closed = true
	defer bpool.Put(w.b)

	err := w.c.Write(w.ctx, w.typ, w.b.Bytes())
	if err != nil {
		return fmt.Errorf("failed to close writer: %w", err)
	}
	return nil
}

// CloseRead implements *Conn.CloseRead for wasm.
func (c *Conn) CloseRead(ctx context.Context) context.Context {
	c.closeReadMu.Lock()
	ctx2 := c.closeReadCtx
	if ctx2 != nil {
		c.closeReadMu.Unlock()
		return ctx2
	}
	ctx, cancel := context.WithCancel(ctx)
	c.closeReadCtx = ctx
	c.closeReadMu.Unlock()

	go func() {
		defer cancel()
		defer c.CloseNow()
		_, _, err := c.read(ctx)
		if err != nil {
			c.Close(StatusPolicyViolation, "unexpected data message")
		}
	}()
	return ctx
}

// SetReadLimit implements *Conn.SetReadLimit for wasm.
func (c *Conn) SetReadLimit(n int64) {
	c.msgReadLimit.Store(n)
}

func (c *Conn) setCloseErr(err error) {
	c.closeErrOnce.Do(func() {
		c.closeErr = fmt.Errorf("WebSocket closed: %w", err)
	})
}

func (c *Conn) isClosed() bool {
	select {
	case <-c.closed:
		return true
	default:
		return false
	}
}

// AcceptOptions represents Accept's options.
type AcceptOptions struct {
	Subprotocols         []string
	InsecureSkipVerify   bool
	OriginPatterns       []string
	CompressionMode      CompressionMode
	CompressionThreshold int
}

// Accept is stubbed out for Wasm.
func Accept(w http.ResponseWriter, r *http.Request, opts *AcceptOptions) (*Conn, error) {
	return nil, errors.New("unimplemented")
}

// StatusCode represents a WebSocket status code.
// https://tools.ietf.org/html/rfc6455#section-7.4
type StatusCode int

// https://www.iana.org/assignments/websocket/websocket.xhtml#close-code-number
//
// These are only the status codes defined by the protocol.
//
// You can define custom codes in the 3000-4999 range.
// The 3000-3999 range is reserved for use by libraries, frameworks and applications.
// The 4000-4999 range is reserved for private use.
const (
	StatusNormalClosure   StatusCode = 1000
	StatusGoingAway       StatusCode = 1001
	StatusProtocolError   StatusCode = 1002
	StatusUnsupportedData StatusCode = 1003

	// 1004 is reserved and so unexported.
	statusReserved StatusCode = 1004

	// StatusNoStatusRcvd cannot be sent in a close message.
	// It is reserved for when a close message is received without
	// a status code.
	StatusNoStatusRcvd StatusCode = 1005

	// StatusAbnormalClosure is exported for use only with Wasm.
	// In non Wasm Go, the returned error will indicate whether the
	// connection was closed abnormally.
	StatusAbnormalClosure StatusCode = 1006

	StatusInvalidFramePayloadData StatusCode = 1007
	StatusPolicyViolation         StatusCode = 1008
	StatusMessageTooBig           StatusCode = 1009
	StatusMandatoryExtension      StatusCode = 1010
	StatusInternalError           StatusCode = 1011
	StatusServiceRestart          StatusCode = 1012
	StatusTryAgainLater           StatusCode = 1013
	StatusBadGateway              StatusCode = 1014

	// StatusTLSHandshake is only exported for use with Wasm.
	// In non Wasm Go, the returned error will indicate whether there was
	// a TLS handshake failure.
	StatusTLSHandshake StatusCode = 1015
)

// CloseError is returned when the connection is closed with a status and reason.
//
// Use Go 1.13's errors.As to check for this error.
// Also see the CloseStatus helper.
type CloseError struct {
	Code   StatusCode
	Reason string
}

func (ce CloseError) Error() string {
	return fmt.Sprintf("status = %v and reason = %q", ce.Code, ce.Reason)
}

// CloseStatus is a convenience wrapper around Go 1.13's errors.As to grab
// the status code from a CloseError.
//
// -1 will be returned if the passed error is nil or not a CloseError.
func CloseStatus(err error) StatusCode {
	var ce CloseError
	if errors.As(err, &ce) {
		return ce.Code
	}
	return -1
}

// CompressionMode represents the modes available to the deflate extension.
// See https://tools.ietf.org/html/rfc7692
// Works in all browsers except Safari which does not implement the deflate extension.
type CompressionMode int

const (
	// CompressionNoContextTakeover grabs a new flate.Reader and flate.Writer as needed
	// for every message. This applies to both server and client side.
	//
	// This means less efficient compression as the sliding window from previous messages
	// will not be used but the memory overhead will be lower if the connections
	// are long lived and seldom used.
	//
	// The message will only be compressed if greater than 512 bytes.
	CompressionNoContextTakeover CompressionMode = iota

	// CompressionContextTakeover uses a flate.Reader and flate.Writer per connection.
	// This enables reusing the sliding window from previous messages.
	// As most WebSocket protocols are repetitive, this can be very efficient.
	// It carries an overhead of 8 kB for every connection compared to CompressionNoContextTakeover.
	//
	// If the peer negotiates NoContextTakeover on the client or server side, it will be
	// used instead as this is required by the RFC.
	CompressionContextTakeover

	// CompressionDisabled disables the deflate extension.
	//
	// Use this if you are using a predominantly binary protocol with very
	// little duplication in between messages or CPU and memory are more
	// important than bandwidth.
	CompressionDisabled
)

// MessageType represents the type of a WebSocket message.
// See https://tools.ietf.org/html/rfc6455#section-5.6
type MessageType int

// MessageType constants.
const (
	// MessageText is for UTF-8 encoded text messages like JSON.
	MessageText MessageType = iota + 1
	// MessageBinary is for binary messages like protobufs.
	MessageBinary
)

type mu struct {
	c  *Conn
	ch chan struct{}
}

func newMu(c *Conn) *mu {
	return &mu{
		c:  c,
		ch: make(chan struct{}, 1),
	}
}

func (m *mu) forceLock() {
	m.ch <- struct{}{}
}

func (m *mu) tryLock() bool {
	select {
	case m.ch <- struct{}{}:
		return true
	default:
		return false
	}
}

func (m *mu) unlock() {
	select {
	case <-m.ch:
	default:
	}
}

type noCopy struct{}

func (*noCopy) Lock() {}
