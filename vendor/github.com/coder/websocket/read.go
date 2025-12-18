//go:build !js

package websocket

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"sync/atomic"
	"time"

	"github.com/coder/websocket/internal/errd"
	"github.com/coder/websocket/internal/util"
)

// Reader reads from the connection until there is a WebSocket
// data message to be read. It will handle ping, pong and close frames as appropriate.
//
// It returns the type of the message and an io.Reader to read it.
// The passed context will also bound the reader.
// Ensure you read to EOF otherwise the connection will hang.
//
// Call CloseRead if you do not expect any data messages from the peer.
//
// Only one Reader may be open at a time.
//
// If you need a separate timeout on the Reader call and the Read itself,
// use time.AfterFunc to cancel the context passed in.
// See https://github.com/nhooyr/websocket/issues/87#issue-451703332
// Most users should not need this.
func (c *Conn) Reader(ctx context.Context) (MessageType, io.Reader, error) {
	return c.reader(ctx)
}

// Read is a convenience method around Reader to read a single message
// from the connection.
func (c *Conn) Read(ctx context.Context) (MessageType, []byte, error) {
	typ, r, err := c.Reader(ctx)
	if err != nil {
		return 0, nil, err
	}

	b, err := io.ReadAll(r)
	return typ, b, err
}

// CloseRead starts a goroutine to read from the connection until it is closed
// or a data message is received.
//
// Once CloseRead is called you cannot read any messages from the connection.
// The returned context will be cancelled when the connection is closed.
//
// If a data message is received, the connection will be closed with StatusPolicyViolation.
//
// Call CloseRead when you do not expect to read any more messages.
// Since it actively reads from the connection, it will ensure that ping, pong and close
// frames are responded to. This means c.Ping and c.Close will still work as expected.
//
// This function is idempotent.
func (c *Conn) CloseRead(ctx context.Context) context.Context {
	c.closeReadMu.Lock()
	ctx2 := c.closeReadCtx
	if ctx2 != nil {
		c.closeReadMu.Unlock()
		return ctx2
	}
	ctx, cancel := context.WithCancel(ctx)
	c.closeReadCtx = ctx
	c.closeReadDone = make(chan struct{})
	c.closeReadMu.Unlock()

	go func() {
		defer close(c.closeReadDone)
		defer cancel()
		defer c.close()
		_, _, err := c.Reader(ctx)
		if err == nil {
			c.Close(StatusPolicyViolation, "unexpected data message")
		}
	}()
	return ctx
}

// SetReadLimit sets the max number of bytes to read for a single message.
// It applies to the Reader and Read methods.
//
// By default, the connection has a message read limit of 32768 bytes.
//
// When the limit is hit, reads return an error wrapping ErrMessageTooBig and
// the connection is closed with StatusMessageTooBig.
//
// Set to -1 to disable.
func (c *Conn) SetReadLimit(n int64) {
	if n >= 0 {
		// We read one more byte than the limit in case
		// there is a fin frame that needs to be read.
		n++
	}

	c.msgReader.limitReader.limit.Store(n)
}

const defaultReadLimit = 32768

func newMsgReader(c *Conn) *msgReader {
	mr := &msgReader{
		c:   c,
		fin: true,
	}
	mr.readFunc = mr.read

	mr.limitReader = newLimitReader(c, mr.readFunc, defaultReadLimit+1)
	return mr
}

func (mr *msgReader) resetFlate() {
	if mr.flateContextTakeover() {
		if mr.dict == nil {
			mr.dict = &slidingWindow{}
		}
		mr.dict.init(32768)
	}
	if mr.flateBufio == nil {
		mr.flateBufio = getBufioReader(mr.readFunc)
	}

	if mr.flateContextTakeover() {
		mr.flateReader = getFlateReader(mr.flateBufio, mr.dict.buf)
	} else {
		mr.flateReader = getFlateReader(mr.flateBufio, nil)
	}
	mr.limitReader.r = mr.flateReader
	mr.flateTail.Reset(deflateMessageTail)
}

func (mr *msgReader) putFlateReader() {
	if mr.flateReader != nil {
		putFlateReader(mr.flateReader)
		mr.flateReader = nil
	}
}

func (mr *msgReader) close() {
	mr.c.readMu.forceLock()
	mr.putFlateReader()
	if mr.dict != nil {
		mr.dict.close()
		mr.dict = nil
	}
	if mr.flateBufio != nil {
		putBufioReader(mr.flateBufio)
	}

	if mr.c.client {
		putBufioReader(mr.c.br)
		mr.c.br = nil
	}
}

func (mr *msgReader) flateContextTakeover() bool {
	if mr.c.client {
		return !mr.c.copts.serverNoContextTakeover
	}
	return !mr.c.copts.clientNoContextTakeover
}

func (c *Conn) readRSV1Illegal(h header) bool {
	// If compression is disabled, rsv1 is illegal.
	if !c.flate() {
		return true
	}
	// rsv1 is only allowed on data frames beginning messages.
	if h.opcode != opText && h.opcode != opBinary {
		return true
	}
	return false
}

func (c *Conn) readLoop(ctx context.Context) (header, error) {
	for {
		h, err := c.readFrameHeader(ctx)
		if err != nil {
			return header{}, err
		}

		if h.rsv1 && c.readRSV1Illegal(h) || h.rsv2 || h.rsv3 {
			err := fmt.Errorf("received header with unexpected rsv bits set: %v:%v:%v", h.rsv1, h.rsv2, h.rsv3)
			c.writeError(StatusProtocolError, err)
			return header{}, err
		}

		if !c.client && !h.masked {
			return header{}, errors.New("received unmasked frame from client")
		}

		switch h.opcode {
		case opClose, opPing, opPong:
			err = c.handleControl(ctx, h)
			if err != nil {
				// Pass through CloseErrors when receiving a close frame.
				if h.opcode == opClose && CloseStatus(err) != -1 {
					return header{}, err
				}
				return header{}, fmt.Errorf("failed to handle control frame %v: %w", h.opcode, err)
			}
		case opContinuation, opText, opBinary:
			return h, nil
		default:
			err := fmt.Errorf("received unknown opcode %v", h.opcode)
			c.writeError(StatusProtocolError, err)
			return header{}, err
		}
	}
}

// prepareRead sets the readTimeout context and returns a done function
// to be called after the read is done. It also returns an error if the
// connection is closed. The reference to the error is used to assign
// an error depending on if the connection closed or the context timed
// out during use. Typically, the referenced error is a named return
// variable of the function calling this method.
func (c *Conn) prepareRead(ctx context.Context, err *error) (func(), error) {
	select {
	case <-c.closed:
		return nil, net.ErrClosed
	default:
	}
	c.setupReadTimeout(ctx)

	done := func() {
		c.clearReadTimeout()
		select {
		case <-c.closed:
			if *err != nil {
				*err = net.ErrClosed
			}
		default:
		}
		if *err != nil && ctx.Err() != nil {
			*err = ctx.Err()
		}
	}

	c.closeStateMu.Lock()
	closeReceivedErr := c.closeReceivedErr
	c.closeStateMu.Unlock()
	if closeReceivedErr != nil {
		defer done()
		return nil, closeReceivedErr
	}

	return done, nil
}

func (c *Conn) readFrameHeader(ctx context.Context) (_ header, err error) {
	readDone, err := c.prepareRead(ctx, &err)
	if err != nil {
		return header{}, err
	}
	defer readDone()

	h, err := readFrameHeader(c.br, c.readHeaderBuf[:])
	if err != nil {
		return header{}, err
	}

	return h, nil
}

func (c *Conn) readFramePayload(ctx context.Context, p []byte) (_ int, err error) {
	readDone, err := c.prepareRead(ctx, &err)
	if err != nil {
		return 0, err
	}
	defer readDone()

	n, err := io.ReadFull(c.br, p)
	if err != nil {
		return n, fmt.Errorf("failed to read frame payload: %w", err)
	}

	return n, nil
}

func (c *Conn) handleControl(ctx context.Context, h header) (err error) {
	if h.payloadLength < 0 || h.payloadLength > maxControlPayload {
		err := fmt.Errorf("received control frame payload with invalid length: %d", h.payloadLength)
		c.writeError(StatusProtocolError, err)
		return err
	}

	if !h.fin {
		err := errors.New("received fragmented control frame")
		c.writeError(StatusProtocolError, err)
		return err
	}

	ctx, cancel := context.WithTimeout(ctx, time.Second*5)
	defer cancel()

	b := c.readControlBuf[:h.payloadLength]
	_, err = c.readFramePayload(ctx, b)
	if err != nil {
		return err
	}

	if h.masked {
		mask(b, h.maskKey)
	}

	switch h.opcode {
	case opPing:
		if c.onPingReceived != nil {
			if !c.onPingReceived(ctx, b) {
				return nil
			}
		}
		return c.writeControl(ctx, opPong, b)
	case opPong:
		if c.onPongReceived != nil {
			c.onPongReceived(ctx, b)
		}
		c.activePingsMu.Lock()
		pong, ok := c.activePings[string(b)]
		c.activePingsMu.Unlock()
		if ok {
			select {
			case pong <- struct{}{}:
			default:
			}
		}
		return nil
	}

	// opClose

	ce, err := parseClosePayload(b)
	if err != nil {
		err = fmt.Errorf("received invalid close payload: %w", err)
		c.writeError(StatusProtocolError, err)
		return err
	}

	err = fmt.Errorf("received close frame: %w", ce)
	c.closeStateMu.Lock()
	c.closeReceivedErr = err
	closeSent := c.closeSentErr != nil
	c.closeStateMu.Unlock()

	// Only unlock readMu if this connection is being closed becaue
	// c.close will try to acquire the readMu lock. We unlock for
	// writeClose as well because it may also call c.close.
	if !closeSent {
		c.readMu.unlock()
		_ = c.writeClose(ce.Code, ce.Reason)
	}
	if !c.casClosing() {
		c.readMu.unlock()
		_ = c.close()
	}
	return err
}

func (c *Conn) reader(ctx context.Context) (_ MessageType, _ io.Reader, err error) {
	defer errd.Wrap(&err, "failed to get reader")

	err = c.readMu.lock(ctx)
	if err != nil {
		return 0, nil, err
	}
	defer c.readMu.unlock()

	if !c.msgReader.fin {
		return 0, nil, errors.New("previous message not read to completion")
	}

	h, err := c.readLoop(ctx)
	if err != nil {
		return 0, nil, err
	}

	if h.opcode == opContinuation {
		err := errors.New("received continuation frame without text or binary frame")
		c.writeError(StatusProtocolError, err)
		return 0, nil, err
	}

	c.msgReader.reset(ctx, h)

	return MessageType(h.opcode), c.msgReader, nil
}

type msgReader struct {
	c *Conn

	ctx         context.Context
	flate       bool
	flateReader io.Reader
	flateBufio  *bufio.Reader
	flateTail   strings.Reader
	limitReader *limitReader
	dict        *slidingWindow

	fin           bool
	payloadLength int64
	maskKey       uint32

	// util.ReaderFunc(mr.Read) to avoid continuous allocations.
	readFunc util.ReaderFunc
}

func (mr *msgReader) reset(ctx context.Context, h header) {
	mr.ctx = ctx
	mr.flate = h.rsv1
	mr.limitReader.reset(mr.readFunc)

	if mr.flate {
		mr.resetFlate()
	}

	mr.setFrame(h)
}

func (mr *msgReader) setFrame(h header) {
	mr.fin = h.fin
	mr.payloadLength = h.payloadLength
	mr.maskKey = h.maskKey
}

func (mr *msgReader) Read(p []byte) (n int, err error) {
	err = mr.c.readMu.lock(mr.ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to read: %w", err)
	}
	defer mr.c.readMu.unlock()

	n, err = mr.limitReader.Read(p)
	if mr.flate && mr.flateContextTakeover() {
		p = p[:n]
		mr.dict.write(p)
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) && mr.fin && mr.flate {
		mr.putFlateReader()
		return n, io.EOF
	}
	if err != nil {
		return n, fmt.Errorf("failed to read: %w", err)
	}
	return n, nil
}

func (mr *msgReader) read(p []byte) (int, error) {
	for {
		if mr.payloadLength == 0 {
			if mr.fin {
				if mr.flate {
					return mr.flateTail.Read(p)
				}
				return 0, io.EOF
			}

			h, err := mr.c.readLoop(mr.ctx)
			if err != nil {
				return 0, err
			}
			if h.opcode != opContinuation {
				err := errors.New("received new data message without finishing the previous message")
				mr.c.writeError(StatusProtocolError, err)
				return 0, err
			}
			mr.setFrame(h)

			continue
		}

		if int64(len(p)) > mr.payloadLength {
			p = p[:mr.payloadLength]
		}

		n, err := mr.c.readFramePayload(mr.ctx, p)
		if err != nil {
			return n, err
		}

		mr.payloadLength -= int64(n)

		if !mr.c.client {
			mr.maskKey = mask(p, mr.maskKey)
		}

		return n, nil
	}
}

type limitReader struct {
	c     *Conn
	r     io.Reader
	limit atomic.Int64
	n     int64
}

func newLimitReader(c *Conn, r io.Reader, limit int64) *limitReader {
	lr := &limitReader{
		c: c,
	}
	lr.limit.Store(limit)
	lr.reset(r)
	return lr
}

func (lr *limitReader) reset(r io.Reader) {
	lr.n = lr.limit.Load()
	lr.r = r
}

func (lr *limitReader) Read(p []byte) (int, error) {
	if lr.n < 0 {
		return lr.r.Read(p)
	}

	if lr.n == 0 {
		reason := fmt.Errorf("read limited at %d bytes", lr.limit.Load())
		lr.c.writeError(StatusMessageTooBig, reason)
		return 0, fmt.Errorf("%w: %v", ErrMessageTooBig, reason)
	}

	if int64(len(p)) > lr.n {
		p = p[:lr.n]
	}
	n, err := lr.r.Read(p)
	lr.n -= int64(n)
	if lr.n < 0 {
		lr.n = 0
	}
	return n, err
}
