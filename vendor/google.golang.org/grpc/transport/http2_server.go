/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package transport

import (
	"bytes"
	"errors"
	"io"
	"math"
	"net"
	"strconv"
	"sync"

	"golang.org/x/net/context"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
)

// ErrIllegalHeaderWrite indicates that setting header is illegal because of
// the stream's state.
var ErrIllegalHeaderWrite = errors.New("transport: the stream is done or WriteHeader was already called")

// http2Server implements the ServerTransport interface with HTTP2.
type http2Server struct {
	conn        net.Conn
	maxStreamID uint32               // max stream ID ever seen
	authInfo    credentials.AuthInfo // auth info about the connection
	// writableChan synchronizes write access to the transport.
	// A writer acquires the write lock by receiving a value on writableChan
	// and releases it by sending on writableChan.
	writableChan chan int
	// shutdownChan is closed when Close is called.
	// Blocking operations should select on shutdownChan to avoid
	// blocking forever after Close.
	shutdownChan chan struct{}
	framer       *framer
	hBuf         *bytes.Buffer  // the buffer for HPACK encoding
	hEnc         *hpack.Encoder // HPACK encoder

	// The max number of concurrent streams.
	maxStreams uint32
	// controlBuf delivers all the control related tasks (e.g., window
	// updates, reset streams, and various settings) to the controller.
	controlBuf *recvBuffer
	fc         *inFlow
	// sendQuotaPool provides flow control to outbound message.
	sendQuotaPool *quotaPool

	mu            sync.Mutex // guard the following
	state         transportState
	activeStreams map[uint32]*Stream
	// the per-stream outbound flow control window size set by the peer.
	streamSendQuota uint32
}

// newHTTP2Server constructs a ServerTransport based on HTTP2. ConnectionError is
// returned if something goes wrong.
func newHTTP2Server(conn net.Conn, maxStreams uint32, authInfo credentials.AuthInfo) (_ ServerTransport, err error) {
	framer := newFramer(conn)
	// Send initial settings as connection preface to client.
	var settings []http2.Setting
	// TODO(zhaoq): Have a better way to signal "no limit" because 0 is
	// permitted in the HTTP2 spec.
	if maxStreams == 0 {
		maxStreams = math.MaxUint32
	} else {
		settings = append(settings, http2.Setting{
			ID:  http2.SettingMaxConcurrentStreams,
			Val: maxStreams,
		})
	}
	if initialWindowSize != defaultWindowSize {
		settings = append(settings, http2.Setting{
			ID:  http2.SettingInitialWindowSize,
			Val: uint32(initialWindowSize)})
	}
	if err := framer.writeSettings(true, settings...); err != nil {
		return nil, connectionErrorf(true, err, "transport: %v", err)
	}
	// Adjust the connection flow control window if needed.
	if delta := uint32(initialConnWindowSize - defaultWindowSize); delta > 0 {
		if err := framer.writeWindowUpdate(true, 0, delta); err != nil {
			return nil, connectionErrorf(true, err, "transport: %v", err)
		}
	}
	var buf bytes.Buffer
	t := &http2Server{
		conn:            conn,
		authInfo:        authInfo,
		framer:          framer,
		hBuf:            &buf,
		hEnc:            hpack.NewEncoder(&buf),
		maxStreams:      maxStreams,
		controlBuf:      newRecvBuffer(),
		fc:              &inFlow{limit: initialConnWindowSize},
		sendQuotaPool:   newQuotaPool(defaultWindowSize),
		state:           reachable,
		writableChan:    make(chan int, 1),
		shutdownChan:    make(chan struct{}),
		activeStreams:   make(map[uint32]*Stream),
		streamSendQuota: defaultWindowSize,
	}
	go t.controller()
	t.writableChan <- 0
	return t, nil
}

// operateHeader takes action on the decoded headers.
func (t *http2Server) operateHeaders(frame *http2.MetaHeadersFrame, handle func(*Stream)) (close bool) {
	buf := newRecvBuffer()
	s := &Stream{
		id:  frame.Header().StreamID,
		st:  t,
		buf: buf,
		fc:  &inFlow{limit: initialWindowSize},
	}

	var state decodeState
	for _, hf := range frame.Fields {
		state.processHeaderField(hf)
	}
	if err := state.err; err != nil {
		if se, ok := err.(StreamError); ok {
			t.controlBuf.put(&resetStream{s.id, statusCodeConvTab[se.Code]})
		}
		return
	}

	if frame.StreamEnded() {
		// s is just created by the caller. No lock needed.
		s.state = streamReadDone
	}
	s.recvCompress = state.encoding
	if state.timeoutSet {
		s.ctx, s.cancel = context.WithTimeout(context.TODO(), state.timeout)
	} else {
		s.ctx, s.cancel = context.WithCancel(context.TODO())
	}
	pr := &peer.Peer{
		Addr: t.conn.RemoteAddr(),
	}
	// Attach Auth info if there is any.
	if t.authInfo != nil {
		pr.AuthInfo = t.authInfo
	}
	s.ctx = peer.NewContext(s.ctx, pr)
	// Cache the current stream to the context so that the server application
	// can find out. Required when the server wants to send some metadata
	// back to the client (unary call only).
	s.ctx = newContextWithStream(s.ctx, s)
	// Attach the received metadata to the context.
	if len(state.mdata) > 0 {
		s.ctx = metadata.NewContext(s.ctx, state.mdata)
	}

	s.dec = &recvBufferReader{
		ctx:  s.ctx,
		recv: s.buf,
	}
	s.recvCompress = state.encoding
	s.method = state.method
	t.mu.Lock()
	if t.state != reachable {
		t.mu.Unlock()
		return
	}
	if uint32(len(t.activeStreams)) >= t.maxStreams {
		t.mu.Unlock()
		t.controlBuf.put(&resetStream{s.id, http2.ErrCodeRefusedStream})
		return
	}
	if s.id%2 != 1 || s.id <= t.maxStreamID {
		t.mu.Unlock()
		// illegal gRPC stream id.
		grpclog.Println("transport: http2Server.HandleStreams received an illegal stream id: ", s.id)
		return true
	}
	t.maxStreamID = s.id
	s.sendQuotaPool = newQuotaPool(int(t.streamSendQuota))
	t.activeStreams[s.id] = s
	t.mu.Unlock()
	s.windowHandler = func(n int) {
		t.updateWindow(s, uint32(n))
	}
	handle(s)
	return
}

// HandleStreams receives incoming streams using the given handler. This is
// typically run in a separate goroutine.
func (t *http2Server) HandleStreams(handle func(*Stream)) {
	// Check the validity of client preface.
	preface := make([]byte, len(clientPreface))
	if _, err := io.ReadFull(t.conn, preface); err != nil {
		grpclog.Printf("transport: http2Server.HandleStreams failed to receive the preface from client: %v", err)
		t.Close()
		return
	}
	if !bytes.Equal(preface, clientPreface) {
		grpclog.Printf("transport: http2Server.HandleStreams received bogus greeting from client: %q", preface)
		t.Close()
		return
	}

	frame, err := t.framer.readFrame()
	if err == io.EOF || err == io.ErrUnexpectedEOF {
		t.Close()
		return
	}
	if err != nil {
		grpclog.Printf("transport: http2Server.HandleStreams failed to read frame: %v", err)
		t.Close()
		return
	}
	sf, ok := frame.(*http2.SettingsFrame)
	if !ok {
		grpclog.Printf("transport: http2Server.HandleStreams saw invalid preface type %T from client", frame)
		t.Close()
		return
	}
	t.handleSettings(sf)

	for {
		frame, err := t.framer.readFrame()
		if err != nil {
			if se, ok := err.(http2.StreamError); ok {
				t.mu.Lock()
				s := t.activeStreams[se.StreamID]
				t.mu.Unlock()
				if s != nil {
					t.closeStream(s)
				}
				t.controlBuf.put(&resetStream{se.StreamID, se.Code})
				continue
			}
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				t.Close()
				return
			}
			grpclog.Printf("transport: http2Server.HandleStreams failed to read frame: %v", err)
			t.Close()
			return
		}
		switch frame := frame.(type) {
		case *http2.MetaHeadersFrame:
			if t.operateHeaders(frame, handle) {
				t.Close()
				break
			}
		case *http2.DataFrame:
			t.handleData(frame)
		case *http2.RSTStreamFrame:
			t.handleRSTStream(frame)
		case *http2.SettingsFrame:
			t.handleSettings(frame)
		case *http2.PingFrame:
			t.handlePing(frame)
		case *http2.WindowUpdateFrame:
			t.handleWindowUpdate(frame)
		case *http2.GoAwayFrame:
			// TODO: Handle GoAway from the client appropriately.
		default:
			grpclog.Printf("transport: http2Server.HandleStreams found unhandled frame type %v.", frame)
		}
	}
}

func (t *http2Server) getStream(f http2.Frame) (*Stream, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.activeStreams == nil {
		// The transport is closing.
		return nil, false
	}
	s, ok := t.activeStreams[f.Header().StreamID]
	if !ok {
		// The stream is already done.
		return nil, false
	}
	return s, true
}

// updateWindow adjusts the inbound quota for the stream and the transport.
// Window updates will deliver to the controller for sending when
// the cumulative quota exceeds the corresponding threshold.
func (t *http2Server) updateWindow(s *Stream, n uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.state == streamDone {
		return
	}
	if w := t.fc.onRead(n); w > 0 {
		t.controlBuf.put(&windowUpdate{0, w})
	}
	if w := s.fc.onRead(n); w > 0 {
		t.controlBuf.put(&windowUpdate{s.id, w})
	}
}

func (t *http2Server) handleData(f *http2.DataFrame) {
	size := len(f.Data())
	if err := t.fc.onData(uint32(size)); err != nil {
		grpclog.Printf("transport: http2Server %v", err)
		t.Close()
		return
	}
	// Select the right stream to dispatch.
	s, ok := t.getStream(f)
	if !ok {
		if w := t.fc.onRead(uint32(size)); w > 0 {
			t.controlBuf.put(&windowUpdate{0, w})
		}
		return
	}
	if size > 0 {
		s.mu.Lock()
		if s.state == streamDone {
			s.mu.Unlock()
			// The stream has been closed. Release the corresponding quota.
			if w := t.fc.onRead(uint32(size)); w > 0 {
				t.controlBuf.put(&windowUpdate{0, w})
			}
			return
		}
		if err := s.fc.onData(uint32(size)); err != nil {
			s.mu.Unlock()
			t.closeStream(s)
			t.controlBuf.put(&resetStream{s.id, http2.ErrCodeFlowControl})
			return
		}
		s.mu.Unlock()
		// TODO(bradfitz, zhaoq): A copy is required here because there is no
		// guarantee f.Data() is consumed before the arrival of next frame.
		// Can this copy be eliminated?
		data := make([]byte, size)
		copy(data, f.Data())
		s.write(recvMsg{data: data})
	}
	if f.Header().Flags.Has(http2.FlagDataEndStream) {
		// Received the end of stream from the client.
		s.mu.Lock()
		if s.state != streamDone {
			s.state = streamReadDone
		}
		s.mu.Unlock()
		s.write(recvMsg{err: io.EOF})
	}
}

func (t *http2Server) handleRSTStream(f *http2.RSTStreamFrame) {
	s, ok := t.getStream(f)
	if !ok {
		return
	}
	t.closeStream(s)
}

func (t *http2Server) handleSettings(f *http2.SettingsFrame) {
	if f.IsAck() {
		return
	}
	var ss []http2.Setting
	f.ForeachSetting(func(s http2.Setting) error {
		ss = append(ss, s)
		return nil
	})
	// The settings will be applied once the ack is sent.
	t.controlBuf.put(&settings{ack: true, ss: ss})
}

func (t *http2Server) handlePing(f *http2.PingFrame) {
	if f.IsAck() { // Do nothing.
		return
	}
	pingAck := &ping{ack: true}
	copy(pingAck.data[:], f.Data[:])
	t.controlBuf.put(pingAck)
}

func (t *http2Server) handleWindowUpdate(f *http2.WindowUpdateFrame) {
	id := f.Header().StreamID
	incr := f.Increment
	if id == 0 {
		t.sendQuotaPool.add(int(incr))
		return
	}
	if s, ok := t.getStream(f); ok {
		s.sendQuotaPool.add(int(incr))
	}
}

func (t *http2Server) writeHeaders(s *Stream, b *bytes.Buffer, endStream bool) error {
	first := true
	endHeaders := false
	var err error
	// Sends the headers in a single batch.
	for !endHeaders {
		size := t.hBuf.Len()
		if size > http2MaxFrameLen {
			size = http2MaxFrameLen
		} else {
			endHeaders = true
		}
		if first {
			p := http2.HeadersFrameParam{
				StreamID:      s.id,
				BlockFragment: b.Next(size),
				EndStream:     endStream,
				EndHeaders:    endHeaders,
			}
			err = t.framer.writeHeaders(endHeaders, p)
			first = false
		} else {
			err = t.framer.writeContinuation(endHeaders, s.id, endHeaders, b.Next(size))
		}
		if err != nil {
			t.Close()
			return connectionErrorf(true, err, "transport: %v", err)
		}
	}
	return nil
}

// WriteHeader sends the header metedata md back to the client.
func (t *http2Server) WriteHeader(s *Stream, md metadata.MD) error {
	s.mu.Lock()
	if s.headerOk || s.state == streamDone {
		s.mu.Unlock()
		return ErrIllegalHeaderWrite
	}
	s.headerOk = true
	if md.Len() > 0 {
		if s.header.Len() > 0 {
			s.header = metadata.Join(s.header, md)
		} else {
			s.header = md
		}
	}
	md = s.header
	s.mu.Unlock()
	if _, err := wait(s.ctx, nil, nil, t.shutdownChan, t.writableChan); err != nil {
		return err
	}
	t.hBuf.Reset()
	t.hEnc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
	t.hEnc.WriteField(hpack.HeaderField{Name: "content-type", Value: "application/grpc"})
	if s.sendCompress != "" {
		t.hEnc.WriteField(hpack.HeaderField{Name: "grpc-encoding", Value: s.sendCompress})
	}
	for k, v := range md {
		if isReservedHeader(k) {
			// Clients don't tolerate reading restricted headers after some non restricted ones were sent.
			continue
		}
		for _, entry := range v {
			t.hEnc.WriteField(hpack.HeaderField{Name: k, Value: entry})
		}
	}
	if err := t.writeHeaders(s, t.hBuf, false); err != nil {
		return err
	}
	t.writableChan <- 0
	return nil
}

// WriteStatus sends stream status to the client and terminates the stream.
// There is no further I/O operations being able to perform on this stream.
// TODO(zhaoq): Now it indicates the end of entire stream. Revisit if early
// OK is adopted.
func (t *http2Server) WriteStatus(s *Stream, statusCode codes.Code, statusDesc string) error {
	var headersSent, hasHeader bool
	s.mu.Lock()
	if s.state == streamDone {
		s.mu.Unlock()
		return nil
	}
	if s.headerOk {
		headersSent = true
	}
	if s.header.Len() > 0 {
		hasHeader = true
	}
	s.mu.Unlock()

	if !headersSent && hasHeader {
		t.WriteHeader(s, nil)
		headersSent = true
	}

	if _, err := wait(s.ctx, nil, nil, t.shutdownChan, t.writableChan); err != nil {
		return err
	}
	t.hBuf.Reset()
	if !headersSent {
		t.hEnc.WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
		t.hEnc.WriteField(hpack.HeaderField{Name: "content-type", Value: "application/grpc"})
	}
	t.hEnc.WriteField(
		hpack.HeaderField{
			Name:  "grpc-status",
			Value: strconv.Itoa(int(statusCode)),
		})
	t.hEnc.WriteField(hpack.HeaderField{Name: "grpc-message", Value: encodeGrpcMessage(statusDesc)})
	// Attach the trailer metadata.
	for k, v := range s.trailer {
		// Clients don't tolerate reading restricted headers after some non restricted ones were sent.
		if isReservedHeader(k) {
			continue
		}
		for _, entry := range v {
			t.hEnc.WriteField(hpack.HeaderField{Name: k, Value: entry})
		}
	}
	if err := t.writeHeaders(s, t.hBuf, true); err != nil {
		t.Close()
		return err
	}
	t.closeStream(s)
	t.writableChan <- 0
	return nil
}

// Write converts the data into HTTP2 data frame and sends it out. Non-nil error
// is returns if it fails (e.g., framing error, transport error).
func (t *http2Server) Write(s *Stream, data []byte, opts *Options) error {
	// TODO(zhaoq): Support multi-writers for a single stream.
	var writeHeaderFrame bool
	s.mu.Lock()
	if s.state == streamDone {
		s.mu.Unlock()
		return streamErrorf(codes.Unknown, "the stream has been done")
	}
	if !s.headerOk {
		writeHeaderFrame = true
	}
	s.mu.Unlock()
	if writeHeaderFrame {
		t.WriteHeader(s, nil)
	}
	r := bytes.NewBuffer(data)
	for {
		if r.Len() == 0 {
			return nil
		}
		size := http2MaxFrameLen
		s.sendQuotaPool.add(0)
		// Wait until the stream has some quota to send the data.
		sq, err := wait(s.ctx, nil, nil, t.shutdownChan, s.sendQuotaPool.acquire())
		if err != nil {
			return err
		}
		t.sendQuotaPool.add(0)
		// Wait until the transport has some quota to send the data.
		tq, err := wait(s.ctx, nil, nil, t.shutdownChan, t.sendQuotaPool.acquire())
		if err != nil {
			if _, ok := err.(StreamError); ok {
				t.sendQuotaPool.cancel()
			}
			return err
		}
		if sq < size {
			size = sq
		}
		if tq < size {
			size = tq
		}
		p := r.Next(size)
		ps := len(p)
		if ps < sq {
			// Overbooked stream quota. Return it back.
			s.sendQuotaPool.add(sq - ps)
		}
		if ps < tq {
			// Overbooked transport quota. Return it back.
			t.sendQuotaPool.add(tq - ps)
		}
		t.framer.adjustNumWriters(1)
		// Got some quota. Try to acquire writing privilege on the
		// transport.
		if _, err := wait(s.ctx, nil, nil, t.shutdownChan, t.writableChan); err != nil {
			if _, ok := err.(StreamError); ok {
				// Return the connection quota back.
				t.sendQuotaPool.add(ps)
			}
			if t.framer.adjustNumWriters(-1) == 0 {
				// This writer is the last one in this batch and has the
				// responsibility to flush the buffered frames. It queues
				// a flush request to controlBuf instead of flushing directly
				// in order to avoid the race with other writing or flushing.
				t.controlBuf.put(&flushIO{})
			}
			return err
		}
		select {
		case <-s.ctx.Done():
			t.sendQuotaPool.add(ps)
			if t.framer.adjustNumWriters(-1) == 0 {
				t.controlBuf.put(&flushIO{})
			}
			t.writableChan <- 0
			return ContextErr(s.ctx.Err())
		default:
		}
		var forceFlush bool
		if r.Len() == 0 && t.framer.adjustNumWriters(0) == 1 && !opts.Last {
			forceFlush = true
		}
		if err := t.framer.writeData(forceFlush, s.id, false, p); err != nil {
			t.Close()
			return connectionErrorf(true, err, "transport: %v", err)
		}
		if t.framer.adjustNumWriters(-1) == 0 {
			t.framer.flushWrite()
		}
		t.writableChan <- 0
	}

}

func (t *http2Server) applySettings(ss []http2.Setting) {
	for _, s := range ss {
		if s.ID == http2.SettingInitialWindowSize {
			t.mu.Lock()
			defer t.mu.Unlock()
			for _, stream := range t.activeStreams {
				stream.sendQuotaPool.reset(int(s.Val - t.streamSendQuota))
			}
			t.streamSendQuota = s.Val
		}

	}
}

// controller running in a separate goroutine takes charge of sending control
// frames (e.g., window update, reset stream, setting, etc.) to the server.
func (t *http2Server) controller() {
	for {
		select {
		case i := <-t.controlBuf.get():
			t.controlBuf.load()
			select {
			case <-t.writableChan:
				switch i := i.(type) {
				case *windowUpdate:
					t.framer.writeWindowUpdate(true, i.streamID, i.increment)
				case *settings:
					if i.ack {
						t.framer.writeSettingsAck(true)
						t.applySettings(i.ss)
					} else {
						t.framer.writeSettings(true, i.ss...)
					}
				case *resetStream:
					t.framer.writeRSTStream(true, i.streamID, i.code)
				case *goAway:
					t.mu.Lock()
					if t.state == closing {
						t.mu.Unlock()
						// The transport is closing.
						return
					}
					sid := t.maxStreamID
					t.state = draining
					t.mu.Unlock()
					t.framer.writeGoAway(true, sid, http2.ErrCodeNo, nil)
				case *flushIO:
					t.framer.flushWrite()
				case *ping:
					t.framer.writePing(true, i.ack, i.data)
				default:
					grpclog.Printf("transport: http2Server.controller got unexpected item type %v\n", i)
				}
				t.writableChan <- 0
				continue
			case <-t.shutdownChan:
				return
			}
		case <-t.shutdownChan:
			return
		}
	}
}

// Close starts shutting down the http2Server transport.
// TODO(zhaoq): Now the destruction is not blocked on any pending streams. This
// could cause some resource issue. Revisit this later.
func (t *http2Server) Close() (err error) {
	t.mu.Lock()
	if t.state == closing {
		t.mu.Unlock()
		return errors.New("transport: Close() was already called")
	}
	t.state = closing
	streams := t.activeStreams
	t.activeStreams = nil
	t.mu.Unlock()
	close(t.shutdownChan)
	err = t.conn.Close()
	// Cancel all active streams.
	for _, s := range streams {
		s.cancel()
	}
	return
}

// closeStream clears the footprint of a stream when the stream is not needed
// any more.
func (t *http2Server) closeStream(s *Stream) {
	t.mu.Lock()
	delete(t.activeStreams, s.id)
	if t.state == draining && len(t.activeStreams) == 0 {
		defer t.Close()
	}
	t.mu.Unlock()
	// In case stream sending and receiving are invoked in separate
	// goroutines (e.g., bi-directional streaming), cancel needs to be
	// called to interrupt the potential blocking on other goroutines.
	s.cancel()
	s.mu.Lock()
	if q := s.fc.resetPendingData(); q > 0 {
		if w := t.fc.onRead(q); w > 0 {
			t.controlBuf.put(&windowUpdate{0, w})
		}
	}
	if s.state == streamDone {
		s.mu.Unlock()
		return
	}
	s.state = streamDone
	s.mu.Unlock()
}

func (t *http2Server) RemoteAddr() net.Addr {
	return t.conn.RemoteAddr()
}

func (t *http2Server) Drain() {
	t.controlBuf.put(&goAway{})
}
