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
	"io"
	"math"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
)

// http2Client implements the ClientTransport interface with HTTP2.
type http2Client struct {
	ctx        context.Context
	target     string // server name/addr
	userAgent  string
	md         interface{}
	conn       net.Conn // underlying communication channel
	remoteAddr net.Addr
	localAddr  net.Addr
	authInfo   credentials.AuthInfo // auth info about the connection
	nextID     uint32               // the next stream ID to be used

	// writableChan synchronizes write access to the transport.
	// A writer acquires the write lock by sending a value on writableChan
	// and releases it by receiving from writableChan.
	writableChan chan int
	// shutdownChan is closed when Close is called.
	// Blocking operations should select on shutdownChan to avoid
	// blocking forever after Close.
	// TODO(zhaoq): Maybe have a channel context?
	shutdownChan chan struct{}
	// errorChan is closed to notify the I/O error to the caller.
	errorChan chan struct{}
	// goAway is closed to notify the upper layer (i.e., addrConn.transportMonitor)
	// that the server sent GoAway on this transport.
	goAway chan struct{}
	// awakenKeepalive is used to wake up keepalive when after it has gone dormant.
	awakenKeepalive chan struct{}

	framer *framer
	hBuf   *bytes.Buffer  // the buffer for HPACK encoding
	hEnc   *hpack.Encoder // HPACK encoder

	// controlBuf delivers all the control related tasks (e.g., window
	// updates, reset streams, and various settings) to the controller.
	controlBuf *recvBuffer
	fc         *inFlow
	// sendQuotaPool provides flow control to outbound message.
	sendQuotaPool *quotaPool
	// streamsQuota limits the max number of concurrent streams.
	streamsQuota *quotaPool

	// The scheme used: https if TLS is on, http otherwise.
	scheme string

	creds []credentials.PerRPCCredentials

	// Boolean to keep track of reading activity on transport.
	// 1 is true and 0 is false.
	activity uint32 // Accessed atomically.
	kp       keepalive.ClientParameters

	statsHandler stats.Handler

	mu            sync.Mutex     // guard the following variables
	state         transportState // the state of underlying connection
	activeStreams map[uint32]*Stream
	// The max number of concurrent streams
	maxStreams int
	// the per-stream outbound flow control window size set by the peer.
	streamSendQuota uint32
	// goAwayID records the Last-Stream-ID in the GoAway frame from the server.
	goAwayID uint32
	// prevGoAway ID records the Last-Stream-ID in the previous GOAway frame.
	prevGoAwayID uint32
	// goAwayReason records the http2.ErrCode and debug data received with the
	// GoAway frame.
	goAwayReason GoAwayReason
}

func dial(ctx context.Context, fn func(context.Context, string) (net.Conn, error), addr string) (net.Conn, error) {
	if fn != nil {
		return fn(ctx, addr)
	}
	return dialContext(ctx, "tcp", addr)
}

func isTemporary(err error) bool {
	switch err {
	case io.EOF:
		// Connection closures may be resolved upon retry, and are thus
		// treated as temporary.
		return true
	case context.DeadlineExceeded:
		// In Go 1.7, context.DeadlineExceeded implements Timeout(), and this
		// special case is not needed. Until then, we need to keep this
		// clause.
		return true
	}

	switch err := err.(type) {
	case interface {
		Temporary() bool
	}:
		return err.Temporary()
	case interface {
		Timeout() bool
	}:
		// Timeouts may be resolved upon retry, and are thus treated as
		// temporary.
		return err.Timeout()
	}
	return false
}

// newHTTP2Client constructs a connected ClientTransport to addr based on HTTP2
// and starts to receive messages on it. Non-nil error returns if construction
// fails.
func newHTTP2Client(ctx context.Context, addr TargetInfo, opts ConnectOptions) (_ ClientTransport, err error) {
	scheme := "http"
	conn, err := dial(ctx, opts.Dialer, addr.Addr)
	if err != nil {
		if opts.FailOnNonTempDialError {
			return nil, connectionErrorf(isTemporary(err), err, "transport: %v", err)
		}
		return nil, connectionErrorf(true, err, "transport: %v", err)
	}
	// Any further errors will close the underlying connection
	defer func(conn net.Conn) {
		if err != nil {
			conn.Close()
		}
	}(conn)
	var authInfo credentials.AuthInfo
	if creds := opts.TransportCredentials; creds != nil {
		scheme = "https"
		conn, authInfo, err = creds.ClientHandshake(ctx, addr.Addr, conn)
		if err != nil {
			// Credentials handshake errors are typically considered permanent
			// to avoid retrying on e.g. bad certificates.
			temp := isTemporary(err)
			return nil, connectionErrorf(temp, err, "transport: %v", err)
		}
	}
	kp := opts.KeepaliveParams
	// Validate keepalive parameters.
	if kp.Time == 0 {
		kp.Time = defaultClientKeepaliveTime
	}
	if kp.Timeout == 0 {
		kp.Timeout = defaultClientKeepaliveTimeout
	}
	var buf bytes.Buffer
	t := &http2Client{
		ctx:        ctx,
		target:     addr.Addr,
		userAgent:  opts.UserAgent,
		md:         addr.Metadata,
		conn:       conn,
		remoteAddr: conn.RemoteAddr(),
		localAddr:  conn.LocalAddr(),
		authInfo:   authInfo,
		// The client initiated stream id is odd starting from 1.
		nextID:          1,
		writableChan:    make(chan int, 1),
		shutdownChan:    make(chan struct{}),
		errorChan:       make(chan struct{}),
		goAway:          make(chan struct{}),
		awakenKeepalive: make(chan struct{}, 1),
		framer:          newFramer(conn),
		hBuf:            &buf,
		hEnc:            hpack.NewEncoder(&buf),
		controlBuf:      newRecvBuffer(),
		fc:              &inFlow{limit: initialConnWindowSize},
		sendQuotaPool:   newQuotaPool(defaultWindowSize),
		scheme:          scheme,
		state:           reachable,
		activeStreams:   make(map[uint32]*Stream),
		creds:           opts.PerRPCCredentials,
		maxStreams:      defaultMaxStreamsClient,
		streamsQuota:    newQuotaPool(defaultMaxStreamsClient),
		streamSendQuota: defaultWindowSize,
		kp:              kp,
		statsHandler:    opts.StatsHandler,
	}
	// Make sure awakenKeepalive can't be written upon.
	// keepalive routine will make it writable, if need be.
	t.awakenKeepalive <- struct{}{}
	if t.statsHandler != nil {
		t.ctx = t.statsHandler.TagConn(t.ctx, &stats.ConnTagInfo{
			RemoteAddr: t.remoteAddr,
			LocalAddr:  t.localAddr,
		})
		connBegin := &stats.ConnBegin{
			Client: true,
		}
		t.statsHandler.HandleConn(t.ctx, connBegin)
	}
	// Start the reader goroutine for incoming message. Each transport has
	// a dedicated goroutine which reads HTTP2 frame from network. Then it
	// dispatches the frame to the corresponding stream entity.
	go t.reader()
	// Send connection preface to server.
	n, err := t.conn.Write(clientPreface)
	if err != nil {
		t.Close()
		return nil, connectionErrorf(true, err, "transport: %v", err)
	}
	if n != len(clientPreface) {
		t.Close()
		return nil, connectionErrorf(true, err, "transport: preface mismatch, wrote %d bytes; want %d", n, len(clientPreface))
	}
	if initialWindowSize != defaultWindowSize {
		err = t.framer.writeSettings(true, http2.Setting{
			ID:  http2.SettingInitialWindowSize,
			Val: uint32(initialWindowSize),
		})
	} else {
		err = t.framer.writeSettings(true)
	}
	if err != nil {
		t.Close()
		return nil, connectionErrorf(true, err, "transport: %v", err)
	}
	// Adjust the connection flow control window if needed.
	if delta := uint32(initialConnWindowSize - defaultWindowSize); delta > 0 {
		if err := t.framer.writeWindowUpdate(true, 0, delta); err != nil {
			t.Close()
			return nil, connectionErrorf(true, err, "transport: %v", err)
		}
	}
	go t.controller()
	if t.kp.Time != infinity {
		go t.keepalive()
	}
	t.writableChan <- 0
	return t, nil
}

func (t *http2Client) newStream(ctx context.Context, callHdr *CallHdr) *Stream {
	// TODO(zhaoq): Handle uint32 overflow of Stream.id.
	s := &Stream{
		id:            t.nextID,
		done:          make(chan struct{}),
		goAway:        make(chan struct{}),
		method:        callHdr.Method,
		sendCompress:  callHdr.SendCompress,
		buf:           newRecvBuffer(),
		fc:            &inFlow{limit: initialWindowSize},
		sendQuotaPool: newQuotaPool(int(t.streamSendQuota)),
		headerChan:    make(chan struct{}),
	}
	t.nextID += 2
	s.windowHandler = func(n int) {
		t.updateWindow(s, uint32(n))
	}
	// The client side stream context should have exactly the same life cycle with the user provided context.
	// That means, s.ctx should be read-only. And s.ctx is done iff ctx is done.
	// So we use the original context here instead of creating a copy.
	s.ctx = ctx
	s.dec = &recvBufferReader{
		ctx:    s.ctx,
		goAway: s.goAway,
		recv:   s.buf,
	}
	return s
}

// NewStream creates a stream and registers it into the transport as "active"
// streams.
func (t *http2Client) NewStream(ctx context.Context, callHdr *CallHdr) (_ *Stream, err error) {
	pr := &peer.Peer{
		Addr: t.remoteAddr,
	}
	// Attach Auth info if there is any.
	if t.authInfo != nil {
		pr.AuthInfo = t.authInfo
	}
	userCtx := ctx
	ctx = peer.NewContext(ctx, pr)
	authData := make(map[string]string)
	for _, c := range t.creds {
		// Construct URI required to get auth request metadata.
		var port string
		if pos := strings.LastIndex(t.target, ":"); pos != -1 {
			// Omit port if it is the default one.
			if t.target[pos+1:] != "443" {
				port = ":" + t.target[pos+1:]
			}
		}
		pos := strings.LastIndex(callHdr.Method, "/")
		if pos == -1 {
			return nil, streamErrorf(codes.InvalidArgument, "transport: malformed method name: %q", callHdr.Method)
		}
		audience := "https://" + callHdr.Host + port + callHdr.Method[:pos]
		data, err := c.GetRequestMetadata(ctx, audience)
		if err != nil {
			return nil, streamErrorf(codes.InvalidArgument, "transport: %v", err)
		}
		for k, v := range data {
			authData[k] = v
		}
	}
	t.mu.Lock()
	if t.activeStreams == nil {
		t.mu.Unlock()
		return nil, ErrConnClosing
	}
	if t.state == draining {
		t.mu.Unlock()
		return nil, ErrStreamDrain
	}
	if t.state != reachable {
		t.mu.Unlock()
		return nil, ErrConnClosing
	}
	t.mu.Unlock()
	sq, err := wait(ctx, nil, nil, t.shutdownChan, t.streamsQuota.acquire())
	if err != nil {
		return nil, err
	}
	// Returns the quota balance back.
	if sq > 1 {
		t.streamsQuota.add(sq - 1)
	}
	if _, err := wait(ctx, nil, nil, t.shutdownChan, t.writableChan); err != nil {
		// Return the quota back now because there is no stream returned to the caller.
		if _, ok := err.(StreamError); ok {
			t.streamsQuota.add(1)
		}
		return nil, err
	}
	t.mu.Lock()
	if t.state == draining {
		t.mu.Unlock()
		t.streamsQuota.add(1)
		// Need to make t writable again so that the rpc in flight can still proceed.
		t.writableChan <- 0
		return nil, ErrStreamDrain
	}
	if t.state != reachable {
		t.mu.Unlock()
		return nil, ErrConnClosing
	}
	s := t.newStream(ctx, callHdr)
	s.clientStatsCtx = userCtx
	t.activeStreams[s.id] = s
	// If the number of active streams change from 0 to 1, then check if keepalive
	// has gone dormant. If so, wake it up.
	if len(t.activeStreams) == 1 {
		select {
		case t.awakenKeepalive <- struct{}{}:
			t.framer.writePing(false, false, [8]byte{})
		default:
		}
	}

	t.mu.Unlock()

	// HPACK encodes various headers. Note that once WriteField(...) is
	// called, the corresponding headers/continuation frame has to be sent
	// because hpack.Encoder is stateful.
	t.hBuf.Reset()
	t.hEnc.WriteField(hpack.HeaderField{Name: ":method", Value: "POST"})
	t.hEnc.WriteField(hpack.HeaderField{Name: ":scheme", Value: t.scheme})
	t.hEnc.WriteField(hpack.HeaderField{Name: ":path", Value: callHdr.Method})
	t.hEnc.WriteField(hpack.HeaderField{Name: ":authority", Value: callHdr.Host})
	t.hEnc.WriteField(hpack.HeaderField{Name: "content-type", Value: "application/grpc"})
	t.hEnc.WriteField(hpack.HeaderField{Name: "user-agent", Value: t.userAgent})
	t.hEnc.WriteField(hpack.HeaderField{Name: "te", Value: "trailers"})

	if callHdr.SendCompress != "" {
		t.hEnc.WriteField(hpack.HeaderField{Name: "grpc-encoding", Value: callHdr.SendCompress})
	}
	if dl, ok := ctx.Deadline(); ok {
		// Send out timeout regardless its value. The server can detect timeout context by itself.
		timeout := dl.Sub(time.Now())
		t.hEnc.WriteField(hpack.HeaderField{Name: "grpc-timeout", Value: encodeTimeout(timeout)})
	}

	for k, v := range authData {
		// Capital header names are illegal in HTTP/2.
		k = strings.ToLower(k)
		t.hEnc.WriteField(hpack.HeaderField{Name: k, Value: v})
	}
	var (
		hasMD      bool
		endHeaders bool
	)
	if md, ok := metadata.FromOutgoingContext(ctx); ok {
		hasMD = true
		for k, vv := range md {
			// HTTP doesn't allow you to set pseudoheaders after non pseudoheaders were set.
			if isReservedHeader(k) {
				continue
			}
			for _, v := range vv {
				t.hEnc.WriteField(hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
			}
		}
	}
	if md, ok := t.md.(*metadata.MD); ok {
		for k, vv := range *md {
			if isReservedHeader(k) {
				continue
			}
			for _, v := range vv {
				t.hEnc.WriteField(hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
			}
		}
	}
	first := true
	bufLen := t.hBuf.Len()
	// Sends the headers in a single batch even when they span multiple frames.
	for !endHeaders {
		size := t.hBuf.Len()
		if size > http2MaxFrameLen {
			size = http2MaxFrameLen
		} else {
			endHeaders = true
		}
		var flush bool
		if endHeaders && (hasMD || callHdr.Flush) {
			flush = true
		}
		if first {
			// Sends a HeadersFrame to server to start a new stream.
			p := http2.HeadersFrameParam{
				StreamID:      s.id,
				BlockFragment: t.hBuf.Next(size),
				EndStream:     false,
				EndHeaders:    endHeaders,
			}
			// Do a force flush for the buffered frames iff it is the last headers frame
			// and there is header metadata to be sent. Otherwise, there is flushing until
			// the corresponding data frame is written.
			err = t.framer.writeHeaders(flush, p)
			first = false
		} else {
			// Sends Continuation frames for the leftover headers.
			err = t.framer.writeContinuation(flush, s.id, endHeaders, t.hBuf.Next(size))
		}
		if err != nil {
			t.notifyError(err)
			return nil, connectionErrorf(true, err, "transport: %v", err)
		}
	}
	s.bytesSent = true

	if t.statsHandler != nil {
		outHeader := &stats.OutHeader{
			Client:      true,
			WireLength:  bufLen,
			FullMethod:  callHdr.Method,
			RemoteAddr:  t.remoteAddr,
			LocalAddr:   t.localAddr,
			Compression: callHdr.SendCompress,
		}
		t.statsHandler.HandleRPC(s.clientStatsCtx, outHeader)
	}
	t.writableChan <- 0
	return s, nil
}

// CloseStream clears the footprint of a stream when the stream is not needed any more.
// This must not be executed in reader's goroutine.
func (t *http2Client) CloseStream(s *Stream, err error) {
	t.mu.Lock()
	if t.activeStreams == nil {
		t.mu.Unlock()
		return
	}
	delete(t.activeStreams, s.id)
	if t.state == draining && len(t.activeStreams) == 0 {
		// The transport is draining and s is the last live stream on t.
		t.mu.Unlock()
		t.Close()
		return
	}
	t.mu.Unlock()
	// rstStream is true in case the stream is being closed at the client-side
	// and the server needs to be intimated about it by sending a RST_STREAM
	// frame.
	// To make sure this frame is written to the wire before the headers of the
	// next stream waiting for streamsQuota, we add to streamsQuota pool only
	// after having acquired the writableChan to send RST_STREAM out (look at
	// the controller() routine).
	var rstStream bool
	var rstError http2.ErrCode
	defer func() {
		// In case, the client doesn't have to send RST_STREAM to server
		// we can safely add back to streamsQuota pool now.
		if !rstStream {
			t.streamsQuota.add(1)
			return
		}
		t.controlBuf.put(&resetStream{s.id, rstError})
	}()
	s.mu.Lock()
	rstStream = s.rstStream
	rstError = s.rstError
	if q := s.fc.resetPendingData(); q > 0 {
		if n := t.fc.onRead(q); n > 0 {
			t.controlBuf.put(&windowUpdate{0, n})
		}
	}
	if s.state == streamDone {
		s.mu.Unlock()
		return
	}
	if !s.headerDone {
		close(s.headerChan)
		s.headerDone = true
	}
	s.state = streamDone
	s.mu.Unlock()
	if _, ok := err.(StreamError); ok {
		rstStream = true
		rstError = http2.ErrCodeCancel
	}
}

// Close kicks off the shutdown process of the transport. This should be called
// only once on a transport. Once it is called, the transport should not be
// accessed any more.
func (t *http2Client) Close() (err error) {
	t.mu.Lock()
	if t.state == closing {
		t.mu.Unlock()
		return
	}
	if t.state == reachable || t.state == draining {
		close(t.errorChan)
	}
	t.state = closing
	t.mu.Unlock()
	close(t.shutdownChan)
	err = t.conn.Close()
	t.mu.Lock()
	streams := t.activeStreams
	t.activeStreams = nil
	t.mu.Unlock()
	// Notify all active streams.
	for _, s := range streams {
		s.mu.Lock()
		if !s.headerDone {
			close(s.headerChan)
			s.headerDone = true
		}
		s.mu.Unlock()
		s.write(recvMsg{err: ErrConnClosing})
	}
	if t.statsHandler != nil {
		connEnd := &stats.ConnEnd{
			Client: true,
		}
		t.statsHandler.HandleConn(t.ctx, connEnd)
	}
	return
}

func (t *http2Client) GracefulClose() error {
	t.mu.Lock()
	switch t.state {
	case unreachable:
		// The server may close the connection concurrently. t is not available for
		// any streams. Close it now.
		t.mu.Unlock()
		t.Close()
		return nil
	case closing:
		t.mu.Unlock()
		return nil
	}
	// Notify the streams which were initiated after the server sent GOAWAY.
	select {
	case <-t.goAway:
		n := t.prevGoAwayID
		if n == 0 && t.nextID > 1 {
			n = t.nextID - 2
		}
		m := t.goAwayID + 2
		if m == 2 {
			m = 1
		}
		for i := m; i <= n; i += 2 {
			if s, ok := t.activeStreams[i]; ok {
				close(s.goAway)
			}
		}
	default:
	}
	if t.state == draining {
		t.mu.Unlock()
		return nil
	}
	t.state = draining
	active := len(t.activeStreams)
	t.mu.Unlock()
	if active == 0 {
		return t.Close()
	}
	return nil
}

// Write formats the data into HTTP2 data frame(s) and sends it out. The caller
// should proceed only if Write returns nil.
// TODO(zhaoq): opts.Delay is ignored in this implementation. Support it later
// if it improves the performance.
func (t *http2Client) Write(s *Stream, data []byte, opts *Options) error {
	r := bytes.NewBuffer(data)
	for {
		var p []byte
		if r.Len() > 0 {
			size := http2MaxFrameLen
			// Wait until the stream has some quota to send the data.
			sq, err := wait(s.ctx, s.done, s.goAway, t.shutdownChan, s.sendQuotaPool.acquire())
			if err != nil {
				return err
			}
			// Wait until the transport has some quota to send the data.
			tq, err := wait(s.ctx, s.done, s.goAway, t.shutdownChan, t.sendQuotaPool.acquire())
			if err != nil {
				return err
			}
			if sq < size {
				size = sq
			}
			if tq < size {
				size = tq
			}
			p = r.Next(size)
			ps := len(p)
			if ps < sq {
				// Overbooked stream quota. Return it back.
				s.sendQuotaPool.add(sq - ps)
			}
			if ps < tq {
				// Overbooked transport quota. Return it back.
				t.sendQuotaPool.add(tq - ps)
			}
		}
		var (
			endStream  bool
			forceFlush bool
		)
		if opts.Last && r.Len() == 0 {
			endStream = true
		}
		// Indicate there is a writer who is about to write a data frame.
		t.framer.adjustNumWriters(1)
		// Got some quota. Try to acquire writing privilege on the transport.
		if _, err := wait(s.ctx, s.done, s.goAway, t.shutdownChan, t.writableChan); err != nil {
			if _, ok := err.(StreamError); ok || err == io.EOF {
				// Return the connection quota back.
				t.sendQuotaPool.add(len(p))
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
			t.sendQuotaPool.add(len(p))
			if t.framer.adjustNumWriters(-1) == 0 {
				t.controlBuf.put(&flushIO{})
			}
			t.writableChan <- 0
			return ContextErr(s.ctx.Err())
		default:
		}
		if r.Len() == 0 && t.framer.adjustNumWriters(0) == 1 {
			// Do a force flush iff this is last frame for the entire gRPC message
			// and the caller is the only writer at this moment.
			forceFlush = true
		}
		// If WriteData fails, all the pending streams will be handled
		// by http2Client.Close(). No explicit CloseStream() needs to be
		// invoked.
		if err := t.framer.writeData(forceFlush, s.id, endStream, p); err != nil {
			t.notifyError(err)
			return connectionErrorf(true, err, "transport: %v", err)
		}
		if t.framer.adjustNumWriters(-1) == 0 {
			t.framer.flushWrite()
		}
		t.writableChan <- 0
		if r.Len() == 0 {
			break
		}
	}
	if !opts.Last {
		return nil
	}
	s.mu.Lock()
	if s.state != streamDone {
		s.state = streamWriteDone
	}
	s.mu.Unlock()
	return nil
}

func (t *http2Client) getStream(f http2.Frame) (*Stream, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	s, ok := t.activeStreams[f.Header().StreamID]
	return s, ok
}

// updateWindow adjusts the inbound quota for the stream and the transport.
// Window updates will deliver to the controller for sending when
// the cumulative quota exceeds the corresponding threshold.
func (t *http2Client) updateWindow(s *Stream, n uint32) {
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

func (t *http2Client) handleData(f *http2.DataFrame) {
	size := f.Header().Length
	if err := t.fc.onData(uint32(size)); err != nil {
		t.notifyError(connectionErrorf(true, err, "%v", err))
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
		if f.Header().Flags.Has(http2.FlagDataPadded) {
			if w := t.fc.onRead(uint32(size) - uint32(len(f.Data()))); w > 0 {
				t.controlBuf.put(&windowUpdate{0, w})
			}
		}
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
			s.rstStream = true
			s.rstError = http2.ErrCodeFlowControl
			s.finish(status.New(codes.Internal, err.Error()))
			s.mu.Unlock()
			s.write(recvMsg{err: io.EOF})
			return
		}
		if f.Header().Flags.Has(http2.FlagDataPadded) {
			if w := s.fc.onRead(uint32(size) - uint32(len(f.Data()))); w > 0 {
				t.controlBuf.put(&windowUpdate{s.id, w})
			}
		}
		s.mu.Unlock()
		// TODO(bradfitz, zhaoq): A copy is required here because there is no
		// guarantee f.Data() is consumed before the arrival of next frame.
		// Can this copy be eliminated?
		if len(f.Data()) > 0 {
			data := make([]byte, len(f.Data()))
			copy(data, f.Data())
			s.write(recvMsg{data: data})
		}
	}
	// The server has closed the stream without sending trailers.  Record that
	// the read direction is closed, and set the status appropriately.
	if f.FrameHeader.Flags.Has(http2.FlagDataEndStream) {
		s.mu.Lock()
		if s.state == streamDone {
			s.mu.Unlock()
			return
		}
		s.finish(status.New(codes.Internal, "server closed the stream without sending trailers"))
		s.mu.Unlock()
		s.write(recvMsg{err: io.EOF})
	}
}

func (t *http2Client) handleRSTStream(f *http2.RSTStreamFrame) {
	s, ok := t.getStream(f)
	if !ok {
		return
	}
	s.mu.Lock()
	if s.state == streamDone {
		s.mu.Unlock()
		return
	}
	if !s.headerDone {
		close(s.headerChan)
		s.headerDone = true
	}
	statusCode, ok := http2ErrConvTab[http2.ErrCode(f.ErrCode)]
	if !ok {
		grpclog.Println("transport: http2Client.handleRSTStream found no mapped gRPC status for the received http2 error ", f.ErrCode)
		statusCode = codes.Unknown
	}
	s.finish(status.Newf(statusCode, "stream terminated by RST_STREAM with error code: %d", f.ErrCode))
	s.mu.Unlock()
	s.write(recvMsg{err: io.EOF})
}

func (t *http2Client) handleSettings(f *http2.SettingsFrame) {
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

func (t *http2Client) handlePing(f *http2.PingFrame) {
	if f.IsAck() { // Do nothing.
		return
	}
	pingAck := &ping{ack: true}
	copy(pingAck.data[:], f.Data[:])
	t.controlBuf.put(pingAck)
}

func (t *http2Client) handleGoAway(f *http2.GoAwayFrame) {
	if f.ErrCode == http2.ErrCodeEnhanceYourCalm {
		grpclog.Printf("Client received GoAway with http2.ErrCodeEnhanceYourCalm.")
	}
	t.mu.Lock()
	if t.state == reachable || t.state == draining {
		if f.LastStreamID > 0 && f.LastStreamID%2 != 1 {
			t.mu.Unlock()
			t.notifyError(connectionErrorf(true, nil, "received illegal http2 GOAWAY frame: stream ID %d is even", f.LastStreamID))
			return
		}
		select {
		case <-t.goAway:
			id := t.goAwayID
			// t.goAway has been closed (i.e.,multiple GoAways).
			if id < f.LastStreamID {
				t.mu.Unlock()
				t.notifyError(connectionErrorf(true, nil, "received illegal http2 GOAWAY frame: previously recv GOAWAY frame with LastStramID %d, currently recv %d", id, f.LastStreamID))
				return
			}
			t.prevGoAwayID = id
			t.goAwayID = f.LastStreamID
			t.mu.Unlock()
			return
		default:
			t.setGoAwayReason(f)
		}
		t.goAwayID = f.LastStreamID
		close(t.goAway)
	}
	t.mu.Unlock()
}

// setGoAwayReason sets the value of t.goAwayReason based
// on the GoAway frame received.
// It expects a lock on transport's mutext to be held by
// the caller.
func (t *http2Client) setGoAwayReason(f *http2.GoAwayFrame) {
	t.goAwayReason = NoReason
	switch f.ErrCode {
	case http2.ErrCodeEnhanceYourCalm:
		if string(f.DebugData()) == "too_many_pings" {
			t.goAwayReason = TooManyPings
		}
	}
}

func (t *http2Client) GetGoAwayReason() GoAwayReason {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.goAwayReason
}

func (t *http2Client) handleWindowUpdate(f *http2.WindowUpdateFrame) {
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

// operateHeaders takes action on the decoded headers.
func (t *http2Client) operateHeaders(frame *http2.MetaHeadersFrame) {
	s, ok := t.getStream(frame)
	if !ok {
		return
	}
	s.bytesReceived = true
	var state decodeState
	for _, hf := range frame.Fields {
		if err := state.processHeaderField(hf); err != nil {
			s.mu.Lock()
			if !s.headerDone {
				close(s.headerChan)
				s.headerDone = true
			}
			s.mu.Unlock()
			s.write(recvMsg{err: err})
			// Something wrong. Stops reading even when there is remaining.
			return
		}
	}

	endStream := frame.StreamEnded()
	var isHeader bool
	defer func() {
		if t.statsHandler != nil {
			if isHeader {
				inHeader := &stats.InHeader{
					Client:     true,
					WireLength: int(frame.Header().Length),
				}
				t.statsHandler.HandleRPC(s.clientStatsCtx, inHeader)
			} else {
				inTrailer := &stats.InTrailer{
					Client:     true,
					WireLength: int(frame.Header().Length),
				}
				t.statsHandler.HandleRPC(s.clientStatsCtx, inTrailer)
			}
		}
	}()

	s.mu.Lock()
	if !endStream {
		s.recvCompress = state.encoding
	}
	if !s.headerDone {
		if !endStream && len(state.mdata) > 0 {
			s.header = state.mdata
		}
		close(s.headerChan)
		s.headerDone = true
		isHeader = true
	}
	if !endStream || s.state == streamDone {
		s.mu.Unlock()
		return
	}

	if len(state.mdata) > 0 {
		s.trailer = state.mdata
	}
	s.finish(state.status())
	s.mu.Unlock()
	s.write(recvMsg{err: io.EOF})
}

func handleMalformedHTTP2(s *Stream, err error) {
	s.mu.Lock()
	if !s.headerDone {
		close(s.headerChan)
		s.headerDone = true
	}
	s.mu.Unlock()
	s.write(recvMsg{err: err})
}

// reader runs as a separate goroutine in charge of reading data from network
// connection.
//
// TODO(zhaoq): currently one reader per transport. Investigate whether this is
// optimal.
// TODO(zhaoq): Check the validity of the incoming frame sequence.
func (t *http2Client) reader() {
	// Check the validity of server preface.
	frame, err := t.framer.readFrame()
	if err != nil {
		t.notifyError(err)
		return
	}
	atomic.CompareAndSwapUint32(&t.activity, 0, 1)
	sf, ok := frame.(*http2.SettingsFrame)
	if !ok {
		t.notifyError(err)
		return
	}
	t.handleSettings(sf)

	// loop to keep reading incoming messages on this transport.
	for {
		frame, err := t.framer.readFrame()
		atomic.CompareAndSwapUint32(&t.activity, 0, 1)
		if err != nil {
			// Abort an active stream if the http2.Framer returns a
			// http2.StreamError. This can happen only if the server's response
			// is malformed http2.
			if se, ok := err.(http2.StreamError); ok {
				t.mu.Lock()
				s := t.activeStreams[se.StreamID]
				t.mu.Unlock()
				if s != nil {
					// use error detail to provide better err message
					handleMalformedHTTP2(s, streamErrorf(http2ErrConvTab[se.Code], "%v", t.framer.errorDetail()))
				}
				continue
			} else {
				// Transport error.
				t.notifyError(err)
				return
			}
		}
		switch frame := frame.(type) {
		case *http2.MetaHeadersFrame:
			t.operateHeaders(frame)
		case *http2.DataFrame:
			t.handleData(frame)
		case *http2.RSTStreamFrame:
			t.handleRSTStream(frame)
		case *http2.SettingsFrame:
			t.handleSettings(frame)
		case *http2.PingFrame:
			t.handlePing(frame)
		case *http2.GoAwayFrame:
			t.handleGoAway(frame)
		case *http2.WindowUpdateFrame:
			t.handleWindowUpdate(frame)
		default:
			grpclog.Printf("transport: http2Client.reader got unhandled frame type %v.", frame)
		}
	}
}

func (t *http2Client) applySettings(ss []http2.Setting) {
	for _, s := range ss {
		switch s.ID {
		case http2.SettingMaxConcurrentStreams:
			// TODO(zhaoq): This is a hack to avoid significant refactoring of the
			// code to deal with the unrealistic int32 overflow. Probably will try
			// to find a better way to handle this later.
			if s.Val > math.MaxInt32 {
				s.Val = math.MaxInt32
			}
			t.mu.Lock()
			ms := t.maxStreams
			t.maxStreams = int(s.Val)
			t.mu.Unlock()
			t.streamsQuota.add(int(s.Val) - ms)
		case http2.SettingInitialWindowSize:
			t.mu.Lock()
			for _, stream := range t.activeStreams {
				// Adjust the sending quota for each stream.
				stream.sendQuotaPool.add(int(s.Val - t.streamSendQuota))
			}
			t.streamSendQuota = s.Val
			t.mu.Unlock()
		}
	}
}

// controller running in a separate goroutine takes charge of sending control
// frames (e.g., window update, reset stream, setting, etc.) to the server.
func (t *http2Client) controller() {
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
					// If the server needs to be to intimated about stream closing,
					// then we need to make sure the RST_STREAM frame is written to
					// the wire before the headers of the next stream waiting on
					// streamQuota. We ensure this by adding to the streamsQuota pool
					// only after having acquired the writableChan to send RST_STREAM.
					t.streamsQuota.add(1)
					t.framer.writeRSTStream(true, i.streamID, i.code)
				case *flushIO:
					t.framer.flushWrite()
				case *ping:
					t.framer.writePing(true, i.ack, i.data)
				default:
					grpclog.Printf("transport: http2Client.controller got unexpected item type %v\n", i)
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

// keepalive running in a separate goroutune makes sure the connection is alive by sending pings.
func (t *http2Client) keepalive() {
	p := &ping{data: [8]byte{}}
	timer := time.NewTimer(t.kp.Time)
	for {
		select {
		case <-timer.C:
			if atomic.CompareAndSwapUint32(&t.activity, 1, 0) {
				timer.Reset(t.kp.Time)
				continue
			}
			// Check if keepalive should go dormant.
			t.mu.Lock()
			if len(t.activeStreams) < 1 && !t.kp.PermitWithoutStream {
				// Make awakenKeepalive writable.
				<-t.awakenKeepalive
				t.mu.Unlock()
				select {
				case <-t.awakenKeepalive:
					// If the control gets here a ping has been sent
					// need to reset the timer with keepalive.Timeout.
				case <-t.shutdownChan:
					return
				}
			} else {
				t.mu.Unlock()
				// Send ping.
				t.controlBuf.put(p)
			}

			// By the time control gets here a ping has been sent one way or the other.
			timer.Reset(t.kp.Timeout)
			select {
			case <-timer.C:
				if atomic.CompareAndSwapUint32(&t.activity, 1, 0) {
					timer.Reset(t.kp.Time)
					continue
				}
				t.Close()
				return
			case <-t.shutdownChan:
				if !timer.Stop() {
					<-timer.C
				}
				return
			}
		case <-t.shutdownChan:
			if !timer.Stop() {
				<-timer.C
			}
			return
		}
	}
}

func (t *http2Client) Error() <-chan struct{} {
	return t.errorChan
}

func (t *http2Client) GoAway() <-chan struct{} {
	return t.goAway
}

func (t *http2Client) notifyError(err error) {
	t.mu.Lock()
	// make sure t.errorChan is closed only once.
	if t.state == draining {
		t.mu.Unlock()
		t.Close()
		return
	}
	if t.state == reachable {
		t.state = unreachable
		close(t.errorChan)
		grpclog.Printf("transport: http2Client.notifyError got notified that the client transport was broken %v.", err)
	}
	t.mu.Unlock()
}
