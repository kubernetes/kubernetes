// Copyright 2013 The Gorilla WebSocket Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bufio"
	"encoding/binary"
	"errors"
	"io"
	"io/ioutil"
	"math/rand"
	"net"
	"strconv"
	"time"
)

const (
	maxFrameHeaderSize         = 2 + 8 + 4 // Fixed header + length + mask
	maxControlFramePayloadSize = 125
	finalBit                   = 1 << 7
	maskBit                    = 1 << 7
	writeWait                  = time.Second

	defaultReadBufferSize  = 4096
	defaultWriteBufferSize = 4096

	continuationFrame = 0
	noFrame           = -1
)

// Close codes defined in RFC 6455, section 11.7.
const (
	CloseNormalClosure           = 1000
	CloseGoingAway               = 1001
	CloseProtocolError           = 1002
	CloseUnsupportedData         = 1003
	CloseNoStatusReceived        = 1005
	CloseAbnormalClosure         = 1006
	CloseInvalidFramePayloadData = 1007
	ClosePolicyViolation         = 1008
	CloseMessageTooBig           = 1009
	CloseMandatoryExtension      = 1010
	CloseInternalServerErr       = 1011
	CloseTLSHandshake            = 1015
)

// The message types are defined in RFC 6455, section 11.8.
const (
	// TextMessage denotes a text data message. The text message payload is
	// interpreted as UTF-8 encoded text data.
	TextMessage = 1

	// BinaryMessage denotes a binary data message.
	BinaryMessage = 2

	// CloseMessage denotes a close control message. The optional message
	// payload contains a numeric code and text. Use the FormatCloseMessage
	// function to format a close message payload.
	CloseMessage = 8

	// PingMessage denotes a ping control message. The optional message payload
	// is UTF-8 encoded text.
	PingMessage = 9

	// PongMessage denotes a ping control message. The optional message payload
	// is UTF-8 encoded text.
	PongMessage = 10
)

// ErrCloseSent is returned when the application writes a message to the
// connection after sending a close message.
var ErrCloseSent = errors.New("websocket: close sent")

// ErrReadLimit is returned when reading a message that is larger than the
// read limit set for the connection.
var ErrReadLimit = errors.New("websocket: read limit exceeded")

// netError satisfies the net Error interface.
type netError struct {
	msg       string
	temporary bool
	timeout   bool
}

func (e *netError) Error() string   { return e.msg }
func (e *netError) Temporary() bool { return e.temporary }
func (e *netError) Timeout() bool   { return e.timeout }

// closeError represents close frame.
type closeError struct {
	code int
	text string
}

func (e *closeError) Error() string {
	return "websocket: close " + strconv.Itoa(e.code) + " " + e.text
}

var (
	errWriteTimeout        = &netError{msg: "websocket: write timeout", timeout: true}
	errUnexpectedEOF       = &closeError{code: CloseAbnormalClosure, text: io.ErrUnexpectedEOF.Error()}
	errBadWriteOpCode      = errors.New("websocket: bad write message type")
	errWriteClosed         = errors.New("websocket: write closed")
	errInvalidControlFrame = errors.New("websocket: invalid control frame")
)

func hideTempErr(err error) error {
	if e, ok := err.(net.Error); ok && e.Temporary() {
		err = &netError{msg: e.Error(), timeout: e.Timeout()}
	}
	return err
}

func isControl(frameType int) bool {
	return frameType == CloseMessage || frameType == PingMessage || frameType == PongMessage
}

func isData(frameType int) bool {
	return frameType == TextMessage || frameType == BinaryMessage
}

func maskBytes(key [4]byte, pos int, b []byte) int {
	for i := range b {
		b[i] ^= key[pos&3]
		pos++
	}
	return pos & 3
}

func newMaskKey() [4]byte {
	n := rand.Uint32()
	return [4]byte{byte(n), byte(n >> 8), byte(n >> 16), byte(n >> 24)}
}

// Conn represents a WebSocket connection.
type Conn struct {
	conn        net.Conn
	isServer    bool
	subprotocol string

	// Write fields
	mu        chan bool // used as mutex to protect write to conn and closeSent
	closeSent bool      // true if close message was sent

	// Message writer fields.
	writeErr       error
	writeBuf       []byte // frame is constructed in this buffer.
	writePos       int    // end of data in writeBuf.
	writeFrameType int    // type of the current frame.
	writeSeq       int    // incremented to invalidate message writers.
	writeDeadline  time.Time

	// Read fields
	readErr       error
	br            *bufio.Reader
	readRemaining int64 // bytes remaining in current frame.
	readFinal     bool  // true the current message has more frames.
	readSeq       int   // incremented to invalidate message readers.
	readLength    int64 // Message size.
	readLimit     int64 // Maximum message size.
	readMaskPos   int
	readMaskKey   [4]byte
	handlePong    func(string) error
	handlePing    func(string) error
}

func newConn(conn net.Conn, isServer bool, readBufferSize, writeBufferSize int) *Conn {
	mu := make(chan bool, 1)
	mu <- true

	if readBufferSize == 0 {
		readBufferSize = defaultReadBufferSize
	}
	if writeBufferSize == 0 {
		writeBufferSize = defaultWriteBufferSize
	}

	c := &Conn{
		isServer:       isServer,
		br:             bufio.NewReaderSize(conn, readBufferSize),
		conn:           conn,
		mu:             mu,
		readFinal:      true,
		writeBuf:       make([]byte, writeBufferSize+maxFrameHeaderSize),
		writeFrameType: noFrame,
		writePos:       maxFrameHeaderSize,
	}
	c.SetPingHandler(nil)
	c.SetPongHandler(nil)
	return c
}

// Subprotocol returns the negotiated protocol for the connection.
func (c *Conn) Subprotocol() string {
	return c.subprotocol
}

// Close closes the underlying network connection without sending or waiting for a close frame.
func (c *Conn) Close() error {
	return c.conn.Close()
}

// LocalAddr returns the local network address.
func (c *Conn) LocalAddr() net.Addr {
	return c.conn.LocalAddr()
}

// RemoteAddr returns the remote network address.
func (c *Conn) RemoteAddr() net.Addr {
	return c.conn.RemoteAddr()
}

// Write methods

func (c *Conn) write(frameType int, deadline time.Time, bufs ...[]byte) error {
	<-c.mu
	defer func() { c.mu <- true }()

	if c.closeSent {
		return ErrCloseSent
	} else if frameType == CloseMessage {
		c.closeSent = true
	}

	c.conn.SetWriteDeadline(deadline)
	for _, buf := range bufs {
		if len(buf) > 0 {
			n, err := c.conn.Write(buf)
			if n != len(buf) {
				// Close on partial write.
				c.conn.Close()
			}
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// WriteControl writes a control message with the given deadline. The allowed
// message types are CloseMessage, PingMessage and PongMessage.
func (c *Conn) WriteControl(messageType int, data []byte, deadline time.Time) error {
	if !isControl(messageType) {
		return errBadWriteOpCode
	}
	if len(data) > maxControlFramePayloadSize {
		return errInvalidControlFrame
	}

	b0 := byte(messageType) | finalBit
	b1 := byte(len(data))
	if !c.isServer {
		b1 |= maskBit
	}

	buf := make([]byte, 0, maxFrameHeaderSize+maxControlFramePayloadSize)
	buf = append(buf, b0, b1)

	if c.isServer {
		buf = append(buf, data...)
	} else {
		key := newMaskKey()
		buf = append(buf, key[:]...)
		buf = append(buf, data...)
		maskBytes(key, 0, buf[6:])
	}

	d := time.Hour * 1000
	if !deadline.IsZero() {
		d = deadline.Sub(time.Now())
		if d < 0 {
			return errWriteTimeout
		}
	}

	timer := time.NewTimer(d)
	select {
	case <-c.mu:
		timer.Stop()
	case <-timer.C:
		return errWriteTimeout
	}
	defer func() { c.mu <- true }()

	if c.closeSent {
		return ErrCloseSent
	} else if messageType == CloseMessage {
		c.closeSent = true
	}

	c.conn.SetWriteDeadline(deadline)
	n, err := c.conn.Write(buf)
	if n != 0 && n != len(buf) {
		c.conn.Close()
	}
	return err
}

// NextWriter returns a writer for the next message to send.  The writer's
// Close method flushes the complete message to the network.
//
// There can be at most one open writer on a connection. NextWriter closes the
// previous writer if the application has not already done so.
//
// The NextWriter method and the writers returned from the method cannot be
// accessed by more than one goroutine at a time.
func (c *Conn) NextWriter(messageType int) (io.WriteCloser, error) {
	if c.writeErr != nil {
		return nil, c.writeErr
	}

	if c.writeFrameType != noFrame {
		if err := c.flushFrame(true, nil); err != nil {
			return nil, err
		}
	}

	if !isControl(messageType) && !isData(messageType) {
		return nil, errBadWriteOpCode
	}

	c.writeFrameType = messageType
	return messageWriter{c, c.writeSeq}, nil
}

func (c *Conn) flushFrame(final bool, extra []byte) error {
	length := c.writePos - maxFrameHeaderSize + len(extra)

	// Check for invalid control frames.
	if isControl(c.writeFrameType) &&
		(!final || length > maxControlFramePayloadSize) {
		c.writeSeq++
		c.writeFrameType = noFrame
		c.writePos = maxFrameHeaderSize
		return errInvalidControlFrame
	}

	b0 := byte(c.writeFrameType)
	if final {
		b0 |= finalBit
	}
	b1 := byte(0)
	if !c.isServer {
		b1 |= maskBit
	}

	// Assume that the frame starts at beginning of c.writeBuf.
	framePos := 0
	if c.isServer {
		// Adjust up if mask not included in the header.
		framePos = 4
	}

	switch {
	case length >= 65536:
		c.writeBuf[framePos] = b0
		c.writeBuf[framePos+1] = b1 | 127
		binary.BigEndian.PutUint64(c.writeBuf[framePos+2:], uint64(length))
	case length > 125:
		framePos += 6
		c.writeBuf[framePos] = b0
		c.writeBuf[framePos+1] = b1 | 126
		binary.BigEndian.PutUint16(c.writeBuf[framePos+2:], uint16(length))
	default:
		framePos += 8
		c.writeBuf[framePos] = b0
		c.writeBuf[framePos+1] = b1 | byte(length)
	}

	if !c.isServer {
		key := newMaskKey()
		copy(c.writeBuf[maxFrameHeaderSize-4:], key[:])
		maskBytes(key, 0, c.writeBuf[maxFrameHeaderSize:c.writePos])
		if len(extra) > 0 {
			c.writeErr = errors.New("websocket: internal error, extra used in client mode")
			return c.writeErr
		}
	}

	// Write the buffers to the connection.
	c.writeErr = c.write(c.writeFrameType, c.writeDeadline, c.writeBuf[framePos:c.writePos], extra)

	// Setup for next frame.
	c.writePos = maxFrameHeaderSize
	c.writeFrameType = continuationFrame
	if final {
		c.writeSeq++
		c.writeFrameType = noFrame
	}
	return c.writeErr
}

type messageWriter struct {
	c   *Conn
	seq int
}

func (w messageWriter) err() error {
	c := w.c
	if c.writeSeq != w.seq {
		return errWriteClosed
	}
	if c.writeErr != nil {
		return c.writeErr
	}
	return nil
}

func (w messageWriter) ncopy(max int) (int, error) {
	n := len(w.c.writeBuf) - w.c.writePos
	if n <= 0 {
		if err := w.c.flushFrame(false, nil); err != nil {
			return 0, err
		}
		n = len(w.c.writeBuf) - w.c.writePos
	}
	if n > max {
		n = max
	}
	return n, nil
}

func (w messageWriter) write(final bool, p []byte) (int, error) {
	if err := w.err(); err != nil {
		return 0, err
	}

	if len(p) > 2*len(w.c.writeBuf) && w.c.isServer {
		// Don't buffer large messages.
		err := w.c.flushFrame(final, p)
		if err != nil {
			return 0, err
		}
		return len(p), nil
	}

	nn := len(p)
	for len(p) > 0 {
		n, err := w.ncopy(len(p))
		if err != nil {
			return 0, err
		}
		copy(w.c.writeBuf[w.c.writePos:], p[:n])
		w.c.writePos += n
		p = p[n:]
	}
	return nn, nil
}

func (w messageWriter) Write(p []byte) (int, error) {
	return w.write(false, p)
}

func (w messageWriter) WriteString(p string) (int, error) {
	if err := w.err(); err != nil {
		return 0, err
	}

	nn := len(p)
	for len(p) > 0 {
		n, err := w.ncopy(len(p))
		if err != nil {
			return 0, err
		}
		copy(w.c.writeBuf[w.c.writePos:], p[:n])
		w.c.writePos += n
		p = p[n:]
	}
	return nn, nil
}

func (w messageWriter) ReadFrom(r io.Reader) (nn int64, err error) {
	if err := w.err(); err != nil {
		return 0, err
	}
	for {
		if w.c.writePos == len(w.c.writeBuf) {
			err = w.c.flushFrame(false, nil)
			if err != nil {
				break
			}
		}
		var n int
		n, err = r.Read(w.c.writeBuf[w.c.writePos:])
		w.c.writePos += n
		nn += int64(n)
		if err != nil {
			if err == io.EOF {
				err = nil
			}
			break
		}
	}
	return nn, err
}

func (w messageWriter) Close() error {
	if err := w.err(); err != nil {
		return err
	}
	return w.c.flushFrame(true, nil)
}

// WriteMessage is a helper method for getting a writer using NextWriter,
// writing the message and closing the writer.
func (c *Conn) WriteMessage(messageType int, data []byte) error {
	wr, err := c.NextWriter(messageType)
	if err != nil {
		return err
	}
	w := wr.(messageWriter)
	if _, err := w.write(true, data); err != nil {
		return err
	}
	if c.writeSeq == w.seq {
		if err := c.flushFrame(true, nil); err != nil {
			return err
		}
	}
	return nil
}

// SetWriteDeadline sets the write deadline on the underlying network
// connection. After a write has timed out, the websocket state is corrupt and
// all future writes will return an error. A zero value for t means writes will
// not time out.
func (c *Conn) SetWriteDeadline(t time.Time) error {
	c.writeDeadline = t
	return nil
}

// Read methods

// readFull is like io.ReadFull except that io.EOF is never returned.
func (c *Conn) readFull(p []byte) (err error) {
	var n int
	for n < len(p) && err == nil {
		var nn int
		nn, err = c.br.Read(p[n:])
		n += nn
	}
	if n == len(p) {
		err = nil
	} else if err == io.EOF {
		err = errUnexpectedEOF
	}
	return
}

func (c *Conn) advanceFrame() (int, error) {

	// 1. Skip remainder of previous frame.

	if c.readRemaining > 0 {
		if _, err := io.CopyN(ioutil.Discard, c.br, c.readRemaining); err != nil {
			return noFrame, err
		}
	}

	// 2. Read and parse first two bytes of frame header.

	var b [8]byte
	if err := c.readFull(b[:2]); err != nil {
		return noFrame, err
	}

	final := b[0]&finalBit != 0
	frameType := int(b[0] & 0xf)
	reserved := int((b[0] >> 4) & 0x7)
	mask := b[1]&maskBit != 0
	c.readRemaining = int64(b[1] & 0x7f)

	if reserved != 0 {
		return noFrame, c.handleProtocolError("unexpected reserved bits " + strconv.Itoa(reserved))
	}

	switch frameType {
	case CloseMessage, PingMessage, PongMessage:
		if c.readRemaining > maxControlFramePayloadSize {
			return noFrame, c.handleProtocolError("control frame length > 125")
		}
		if !final {
			return noFrame, c.handleProtocolError("control frame not final")
		}
	case TextMessage, BinaryMessage:
		if !c.readFinal {
			return noFrame, c.handleProtocolError("message start before final message frame")
		}
		c.readFinal = final
	case continuationFrame:
		if c.readFinal {
			return noFrame, c.handleProtocolError("continuation after final message frame")
		}
		c.readFinal = final
	default:
		return noFrame, c.handleProtocolError("unknown opcode " + strconv.Itoa(frameType))
	}

	// 3. Read and parse frame length.

	switch c.readRemaining {
	case 126:
		if err := c.readFull(b[:2]); err != nil {
			return noFrame, err
		}
		c.readRemaining = int64(binary.BigEndian.Uint16(b[:2]))
	case 127:
		if err := c.readFull(b[:8]); err != nil {
			return noFrame, err
		}
		c.readRemaining = int64(binary.BigEndian.Uint64(b[:8]))
	}

	// 4. Handle frame masking.

	if mask != c.isServer {
		return noFrame, c.handleProtocolError("incorrect mask flag")
	}

	if mask {
		c.readMaskPos = 0
		if err := c.readFull(c.readMaskKey[:]); err != nil {
			return noFrame, err
		}
	}

	// 5. For text and binary messages, enforce read limit and return.

	if frameType == continuationFrame || frameType == TextMessage || frameType == BinaryMessage {

		c.readLength += c.readRemaining
		if c.readLimit > 0 && c.readLength > c.readLimit {
			c.WriteControl(CloseMessage, FormatCloseMessage(CloseMessageTooBig, ""), time.Now().Add(writeWait))
			return noFrame, ErrReadLimit
		}

		return frameType, nil
	}

	// 6. Read control frame payload.

	var payload []byte
	if c.readRemaining > 0 {
		payload = make([]byte, c.readRemaining)
		c.readRemaining = 0
		if err := c.readFull(payload); err != nil {
			return noFrame, err
		}
		if c.isServer {
			maskBytes(c.readMaskKey, 0, payload)
		}
	}

	// 7. Process control frame payload.

	switch frameType {
	case PongMessage:
		if err := c.handlePong(string(payload)); err != nil {
			return noFrame, err
		}
	case PingMessage:
		if err := c.handlePing(string(payload)); err != nil {
			return noFrame, err
		}
	case CloseMessage:
		c.WriteControl(CloseMessage, []byte{}, time.Now().Add(writeWait))
		closeCode := CloseNoStatusReceived
		closeText := ""
		if len(payload) >= 2 {
			closeCode = int(binary.BigEndian.Uint16(payload))
			closeText = string(payload[2:])
		}
		switch closeCode {
		case CloseNormalClosure, CloseGoingAway:
			return noFrame, io.EOF
		default:
			return noFrame, &closeError{code: closeCode, text: closeText}
		}
	}

	return frameType, nil
}

func (c *Conn) handleProtocolError(message string) error {
	c.WriteControl(CloseMessage, FormatCloseMessage(CloseProtocolError, message), time.Now().Add(writeWait))
	return errors.New("websocket: " + message)
}

// NextReader returns the next data message received from the peer. The
// returned messageType is either TextMessage or BinaryMessage.
//
// There can be at most one open reader on a connection. NextReader discards
// the previous message if the application has not already consumed it.
//
// The NextReader method and the readers returned from the method cannot be
// accessed by more than one goroutine at a time.
func (c *Conn) NextReader() (messageType int, r io.Reader, err error) {

	c.readSeq++
	c.readLength = 0

	for c.readErr == nil {
		frameType, err := c.advanceFrame()
		if err != nil {
			c.readErr = hideTempErr(err)
			break
		}
		if frameType == TextMessage || frameType == BinaryMessage {
			return frameType, messageReader{c, c.readSeq}, nil
		}
	}
	return noFrame, nil, c.readErr
}

type messageReader struct {
	c   *Conn
	seq int
}

func (r messageReader) Read(b []byte) (int, error) {

	if r.seq != r.c.readSeq {
		return 0, io.EOF
	}

	for r.c.readErr == nil {

		if r.c.readRemaining > 0 {
			if int64(len(b)) > r.c.readRemaining {
				b = b[:r.c.readRemaining]
			}
			n, err := r.c.br.Read(b)
			r.c.readErr = hideTempErr(err)
			if r.c.isServer {
				r.c.readMaskPos = maskBytes(r.c.readMaskKey, r.c.readMaskPos, b[:n])
			}
			r.c.readRemaining -= int64(n)
			return n, r.c.readErr
		}

		if r.c.readFinal {
			r.c.readSeq++
			return 0, io.EOF
		}

		frameType, err := r.c.advanceFrame()
		switch {
		case err != nil:
			r.c.readErr = hideTempErr(err)
		case frameType == TextMessage || frameType == BinaryMessage:
			r.c.readErr = errors.New("websocket: internal error, unexpected text or binary in Reader")
		}
	}

	err := r.c.readErr
	if err == io.EOF && r.seq == r.c.readSeq {
		err = errUnexpectedEOF
	}
	return 0, err
}

// ReadMessage is a helper method for getting a reader using NextReader and
// reading from that reader to a buffer.
func (c *Conn) ReadMessage() (messageType int, p []byte, err error) {
	var r io.Reader
	messageType, r, err = c.NextReader()
	if err != nil {
		return messageType, nil, err
	}
	p, err = ioutil.ReadAll(r)
	return messageType, p, err
}

// SetReadDeadline sets the read deadline on the underlying network connection.
// After a read has timed out, the websocket connection state is corrupt and
// all future reads will return an error. A zero value for t means reads will
// not time out.
func (c *Conn) SetReadDeadline(t time.Time) error {
	return c.conn.SetReadDeadline(t)
}

// SetReadLimit sets the maximum size for a message read from the peer. If a
// message exceeds the limit, the connection sends a close frame to the peer
// and returns ErrReadLimit to the application.
func (c *Conn) SetReadLimit(limit int64) {
	c.readLimit = limit
}

// SetPingHandler sets the handler for ping messages received from the peer.
// The default ping handler sends a pong to the peer.
func (c *Conn) SetPingHandler(h func(string) error) {
	if h == nil {
		h = func(message string) error {
			c.WriteControl(PongMessage, []byte(message), time.Now().Add(writeWait))
			return nil
		}
	}
	c.handlePing = h
}

// SetPongHandler sets the handler for pong messages received from the peer.
// The default pong handler does nothing.
func (c *Conn) SetPongHandler(h func(string) error) {
	if h == nil {
		h = func(string) error { return nil }
	}
	c.handlePong = h
}

// UnderlyingConn returns the internal net.Conn. This can be used to further
// modifications to connection specific flags.
func (c *Conn) UnderlyingConn() net.Conn {
	return c.conn
}

// FormatCloseMessage formats closeCode and text as a WebSocket close message.
func FormatCloseMessage(closeCode int, text string) []byte {
	buf := make([]byte, 2+len(text))
	binary.BigEndian.PutUint16(buf, uint16(closeCode))
	copy(buf[2:], text)
	return buf
}
