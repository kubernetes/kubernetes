/*
 *
 * Copyright 2014 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package transport

import (
	"bytes"
	"fmt"
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
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
)

// http2Client implements the ClientTransport interface with HTTP2.
type http2Client struct {
	ctx        context.Context
	cancel     context.CancelFunc
	userAgent  string
	md         interface{}
	conn       net.Conn // underlying communication channel
	remoteAddr net.Addr
	localAddr  net.Addr
	authInfo   credentials.AuthInfo // auth info about the connection
	nextID     uint32               // the next stream ID to be used

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
	controlBuf *controlBuffer
	fc         *inFlow
	// sendQuotaPool provides flow control to outbound message.
	sendQuotaPool *quotaPool
	// localSendQuota limits the amount of data that can be scheduled
	// for writing before it is actually written out.
	localSendQuota *quotaPool
	// streamsQuota limits the max number of concurrent streams.
	streamsQuota *quotaPool

	// The scheme used: https if TLS is on, http otherwise.
	scheme string

	isSecure bool

	creds []credentials.PerRPCCredentials

	// Boolean to keep track of reading activity on transport.
	// 1 is true and 0 is false.
	activity uint32 // Accessed atomically.
	kp       keepalive.ClientParameters

	statsHandler stats.Handler

	initialWindowSize int32

	bdpEst          *bdpEstimator
	outQuotaVersion uint32

	// onSuccess is a callback that client transport calls upon
	// receiving server preface to signal that a succefull HTTP2
	// connection was established.
	onSuccess func()

	mu            sync.Mutex     // guard the following variables
	state         transportState // the state of underlying connection
	activeStreams map[uint32]*Stream
	// The max number of concurrent streams
	maxStreams int
	// the per-stream outbound flow control window size set by the peer.
	streamSendQuota uint32
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
	return true
}

// newHTTP2Client constructs a connected ClientTransport to addr based on HTTP2
// and starts to receive messages on it. Non-nil error returns if construction
// fails.
func newHTTP2Client(connectCtx, ctx context.Context, addr TargetInfo, opts ConnectOptions, onSuccess func()) (_ ClientTransport, err error) {
	scheme := "http"
	ctx, cancel := context.WithCancel(ctx)
	defer func() {
		if err != nil {
			cancel()
		}
	}()

	conn, err := dial(connectCtx, opts.Dialer, addr.Addr)
	if err != nil {
		if opts.FailOnNonTempDialError {
			return nil, connectionErrorf(isTemporary(err), err, "transport: error while dialing: %v", err)
		}
		return nil, connectionErrorf(true, err, "transport: Error while dialing %v", err)
	}
	// Any further errors will close the underlying connection
	defer func(conn net.Conn) {
		if err != nil {
			conn.Close()
		}
	}(conn)
	var (
		isSecure bool
		authInfo credentials.AuthInfo
	)
	if creds := opts.TransportCredentials; creds != nil {
		scheme = "https"
		conn, authInfo, err = creds.ClientHandshake(connectCtx, addr.Authority, conn)
		if err != nil {
			return nil, connectionErrorf(isTemporary(err), err, "transport: authentication handshake failed: %v", err)
		}
		isSecure = true
	}
	kp := opts.KeepaliveParams
	// Validate keepalive parameters.
	if kp.Time == 0 {
		kp.Time = defaultClientKeepaliveTime
	}
	if kp.Timeout == 0 {
		kp.Timeout = defaultClientKeepaliveTimeout
	}
	dynamicWindow := true
	icwz := int32(initialWindowSize)
	if opts.InitialConnWindowSize >= defaultWindowSize {
		icwz = opts.InitialConnWindowSize
		dynamicWindow = false
	}
	var buf bytes.Buffer
	writeBufSize := defaultWriteBufSize
	if opts.WriteBufferSize > 0 {
		writeBufSize = opts.WriteBufferSize
	}
	readBufSize := defaultReadBufSize
	if opts.ReadBufferSize > 0 {
		readBufSize = opts.ReadBufferSize
	}
	t := &http2Client{
		ctx:        ctx,
		cancel:     cancel,
		userAgent:  opts.UserAgent,
		md:         addr.Metadata,
		conn:       conn,
		remoteAddr: conn.RemoteAddr(),
		localAddr:  conn.LocalAddr(),
		authInfo:   authInfo,
		// The client initiated stream id is odd starting from 1.
		nextID:            1,
		goAway:            make(chan struct{}),
		awakenKeepalive:   make(chan struct{}, 1),
		hBuf:              &buf,
		hEnc:              hpack.NewEncoder(&buf),
		framer:            newFramer(conn, writeBufSize, readBufSize),
		controlBuf:        newControlBuffer(),
		fc:                &inFlow{limit: uint32(icwz)},
		sendQuotaPool:     newQuotaPool(defaultWindowSize),
		localSendQuota:    newQuotaPool(defaultLocalSendQuota),
		scheme:            scheme,
		state:             reachable,
		activeStreams:     make(map[uint32]*Stream),
		isSecure:          isSecure,
		creds:             opts.PerRPCCredentials,
		maxStreams:        defaultMaxStreamsClient,
		streamsQuota:      newQuotaPool(defaultMaxStreamsClient),
		streamSendQuota:   defaultWindowSize,
		kp:                kp,
		statsHandler:      opts.StatsHandler,
		initialWindowSize: initialWindowSize,
		onSuccess:         onSuccess,
	}
	if opts.InitialWindowSize >= defaultWindowSize {
		t.initialWindowSize = opts.InitialWindowSize
		dynamicWindow = false
	}
	if dynamicWindow {
		t.bdpEst = &bdpEstimator{
			bdp:               initialWindowSize,
			updateFlowControl: t.updateFlowControl,
		}
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
		return nil, connectionErrorf(true, err, "transport: failed to write client preface: %v", err)
	}
	if n != len(clientPreface) {
		t.Close()
		return nil, connectionErrorf(true, err, "transport: preface mismatch, wrote %d bytes; want %d", n, len(clientPreface))
	}
	if t.initialWindowSize != defaultWindowSize {
		err = t.framer.fr.WriteSettings(http2.Setting{
			ID:  http2.SettingInitialWindowSize,
			Val: uint32(t.initialWindowSize),
		})
	} else {
		err = t.framer.fr.WriteSettings()
	}
	if err != nil {
		t.Close()
		return nil, connectionErrorf(true, err, "transport: failed to write initial settings frame: %v", err)
	}
	// Adjust the connection flow control window if needed.
	if delta := uint32(icwz - defaultWindowSize); delta > 0 {
		if err := t.framer.fr.WriteWindowUpdate(0, delta); err != nil {
			t.Close()
			return nil, connectionErrorf(true, err, "transport: failed to write window update: %v", err)
		}
	}
	t.framer.writer.Flush()
	go func() {
		loopyWriter(t.ctx, t.controlBuf, t.itemHandler)
		t.conn.Close()
	}()
	if t.kp.Time != infinity {
		go t.keepalive()
	}
	return t, nil
}

func (t *http2Client) newStream(ctx context.Context, callHdr *CallHdr) *Stream {
	// TODO(zhaoq): Handle uint32 overflow of Stream.id.
	s := &Stream{
		id:             t.nextID,
		done:           make(chan struct{}),
		goAway:         make(chan struct{}),
		method:         callHdr.Method,
		sendCompress:   callHdr.SendCompress,
		buf:            newRecvBuffer(),
		fc:             &inFlow{limit: uint32(t.initialWindowSize)},
		sendQuotaPool:  newQuotaPool(int(t.streamSendQuota)),
		headerChan:     make(chan struct{}),
		contentSubtype: callHdr.ContentSubtype,
	}
	t.nextID += 2
	s.requestRead = func(n int) {
		t.adjustWindow(s, uint32(n))
	}
	// The client side stream context should have exactly the same life cycle with the user provided context.
	// That means, s.ctx should be read-only. And s.ctx is done iff ctx is done.
	// So we use the original context here instead of creating a copy.
	s.ctx = ctx
	s.trReader = &transportReader{
		reader: &recvBufferReader{
			ctx:    s.ctx,
			goAway: s.goAway,
			recv:   s.buf,
		},
		windowHandler: func(n int) {
			t.updateWindow(s, uint32(n))
		},
	}
	s.waiters = waiters{
		ctx:    s.ctx,
		tctx:   t.ctx,
		done:   s.done,
		goAway: s.goAway,
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
	ctx = peer.NewContext(ctx, pr)
	var (
		authData = make(map[string]string)
		audience string
	)
	// Create an audience string only if needed.
	if len(t.creds) > 0 || callHdr.Creds != nil {
		// Construct URI required to get auth request metadata.
		// Omit port if it is the default one.
		host := strings.TrimSuffix(callHdr.Host, ":443")
		pos := strings.LastIndex(callHdr.Method, "/")
		if pos == -1 {
			pos = len(callHdr.Method)
		}
		audience = "https://" + host + callHdr.Method[:pos]
	}
	for _, c := range t.creds {
		data, err := c.GetRequestMetadata(ctx, audience)
		if err != nil {
			if _, ok := status.FromError(err); ok {
				return nil, err
			}

			return nil, streamErrorf(codes.Unauthenticated, "transport: %v", err)
		}
		for k, v := range data {
			// Capital header names are illegal in HTTP/2.
			k = strings.ToLower(k)
			authData[k] = v
		}
	}
	callAuthData := map[string]string{}
	// Check if credentials.PerRPCCredentials were provided via call options.
	// Note: if these credentials are provided both via dial options and call
	// options, then both sets of credentials will be applied.
	if callCreds := callHdr.Creds; callCreds != nil {
		if !t.isSecure && callCreds.RequireTransportSecurity() {
			return nil, streamErrorf(codes.Unauthenticated, "transport: cannot send secure credentials on an insecure connection")
		}
		data, err := callCreds.GetRequestMetadata(ctx, audience)
		if err != nil {
			return nil, streamErrorf(codes.Internal, "transport: %v", err)
		}
		for k, v := range data {
			// Capital header names are illegal in HTTP/2
			k = strings.ToLower(k)
			callAuthData[k] = v
		}
	}
	t.mu.Lock()
	if t.activeStreams == nil {
		t.mu.Unlock()
		return nil, ErrConnClosing
	}
	if t.state == draining {
		t.mu.Unlock()
		return nil, errStreamDrain
	}
	if t.state != reachable {
		t.mu.Unlock()
		return nil, ErrConnClosing
	}
	t.mu.Unlock()
	// Get a quota of 1 from streamsQuota.
	if _, _, err := t.streamsQuota.get(1, waiters{ctx: ctx, tctx: t.ctx}); err != nil {
		return nil, err
	}
	// TODO(mmukhi): Benchmark if the performance gets better if count the metadata and other header fields
	// first and create a slice of that exact size.
	// Make the slice of certain predictable size to reduce allocations made by append.
	hfLen := 7 // :method, :scheme, :path, :authority, content-type, user-agent, te
	hfLen += len(authData) + len(callAuthData)
	headerFields := make([]hpack.HeaderField, 0, hfLen)
	headerFields = append(headerFields, hpack.HeaderField{Name: ":method", Value: "POST"})
	headerFields = append(headerFields, hpack.HeaderField{Name: ":scheme", Value: t.scheme})
	headerFields = append(headerFields, hpack.HeaderField{Name: ":path", Value: callHdr.Method})
	headerFields = append(headerFields, hpack.HeaderField{Name: ":authority", Value: callHdr.Host})
	headerFields = append(headerFields, hpack.HeaderField{Name: "content-type", Value: contentType(callHdr.ContentSubtype)})
	headerFields = append(headerFields, hpack.HeaderField{Name: "user-agent", Value: t.userAgent})
	headerFields = append(headerFields, hpack.HeaderField{Name: "te", Value: "trailers"})

	if callHdr.SendCompress != "" {
		headerFields = append(headerFields, hpack.HeaderField{Name: "grpc-encoding", Value: callHdr.SendCompress})
	}
	if dl, ok := ctx.Deadline(); ok {
		// Send out timeout regardless its value. The server can detect timeout context by itself.
		// TODO(mmukhi): Perhaps this field should be updated when actually writing out to the wire.
		timeout := dl.Sub(time.Now())
		headerFields = append(headerFields, hpack.HeaderField{Name: "grpc-timeout", Value: encodeTimeout(timeout)})
	}
	for k, v := range authData {
		headerFields = append(headerFields, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
	}
	for k, v := range callAuthData {
		headerFields = append(headerFields, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
	}
	if b := stats.OutgoingTags(ctx); b != nil {
		headerFields = append(headerFields, hpack.HeaderField{Name: "grpc-tags-bin", Value: encodeBinHeader(b)})
	}
	if b := stats.OutgoingTrace(ctx); b != nil {
		headerFields = append(headerFields, hpack.HeaderField{Name: "grpc-trace-bin", Value: encodeBinHeader(b)})
	}

	if md, added, ok := metadata.FromOutgoingContextRaw(ctx); ok {
		var k string
		for _, vv := range added {
			for i, v := range vv {
				if i%2 == 0 {
					k = v
					continue
				}
				// HTTP doesn't allow you to set pseudoheaders after non pseudoheaders were set.
				if isReservedHeader(k) {
					continue
				}
				headerFields = append(headerFields, hpack.HeaderField{Name: strings.ToLower(k), Value: encodeMetadataHeader(k, v)})
			}
		}
		for k, vv := range md {
			// HTTP doesn't allow you to set pseudoheaders after non pseudoheaders were set.
			if isReservedHeader(k) {
				continue
			}
			for _, v := range vv {
				headerFields = append(headerFields, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
			}
		}
	}
	if md, ok := t.md.(*metadata.MD); ok {
		for k, vv := range *md {
			if isReservedHeader(k) {
				continue
			}
			for _, v := range vv {
				headerFields = append(headerFields, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
			}
		}
	}
	t.mu.Lock()
	if t.state == draining {
		t.mu.Unlock()
		t.streamsQuota.add(1)
		return nil, errStreamDrain
	}
	if t.state != reachable {
		t.mu.Unlock()
		return nil, ErrConnClosing
	}
	s := t.newStream(ctx, callHdr)
	t.activeStreams[s.id] = s
	// If the number of active streams change from 0 to 1, then check if keepalive
	// has gone dormant. If so, wake it up.
	if len(t.activeStreams) == 1 {
		select {
		case t.awakenKeepalive <- struct{}{}:
			t.controlBuf.put(&ping{data: [8]byte{}})
			// Fill the awakenKeepalive channel again as this channel must be
			// kept non-writable except at the point that the keepalive()
			// goroutine is waiting either to be awaken or shutdown.
			t.awakenKeepalive <- struct{}{}
		default:
		}
	}
	t.controlBuf.put(&headerFrame{
		streamID:  s.id,
		hf:        headerFields,
		endStream: false,
	})
	t.mu.Unlock()

	if t.statsHandler != nil {
		outHeader := &stats.OutHeader{
			Client:      true,
			FullMethod:  callHdr.Method,
			RemoteAddr:  t.remoteAddr,
			LocalAddr:   t.localAddr,
			Compression: callHdr.SendCompress,
		}
		t.statsHandler.HandleRPC(s.ctx, outHeader)
	}
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
	if err != nil {
		// notify in-flight streams, before the deletion
		s.write(recvMsg{err: err})
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
	if err != nil && !rstStream {
		rstStream = true
		rstError = http2.ErrCodeCancel
	}
}

// Close kicks off the shutdown process of the transport. This should be called
// only once on a transport. Once it is called, the transport should not be
// accessed any more.
func (t *http2Client) Close() error {
	t.mu.Lock()
	if t.state == closing {
		t.mu.Unlock()
		return nil
	}
	t.state = closing
	t.mu.Unlock()
	t.cancel()
	err := t.conn.Close()
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
	return err
}

// GracefulClose sets the state to draining, which prevents new streams from
// being created and causes the transport to be closed when the last active
// stream is closed.  If there are no active streams, the transport is closed
// immediately.  This does nothing if the transport is already draining or
// closing.
func (t *http2Client) GracefulClose() error {
	t.mu.Lock()
	switch t.state {
	case closing, draining:
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
func (t *http2Client) Write(s *Stream, hdr []byte, data []byte, opts *Options) error {
	select {
	case <-s.ctx.Done():
		return ContextErr(s.ctx.Err())
	case <-s.done:
		return io.EOF
	case <-t.ctx.Done():
		return ErrConnClosing
	default:
	}

	if hdr == nil && data == nil && opts.Last {
		// stream.CloseSend uses this to send an empty frame with endStream=True
		t.controlBuf.put(&dataFrame{streamID: s.id, endStream: true, f: func() {}})
		return nil
	}
	// Add data to header frame so that we can equally distribute data across frames.
	emptyLen := http2MaxFrameLen - len(hdr)
	if emptyLen > len(data) {
		emptyLen = len(data)
	}
	hdr = append(hdr, data[:emptyLen]...)
	data = data[emptyLen:]
	var (
		streamQuota    int
		streamQuotaVer uint32
		err            error
	)
	for idx, r := range [][]byte{hdr, data} {
		for len(r) > 0 {
			size := http2MaxFrameLen
			if size > len(r) {
				size = len(r)
			}
			if streamQuota == 0 { // Used up all the locally cached stream quota.
				// Get all the stream quota there is.
				streamQuota, streamQuotaVer, err = s.sendQuotaPool.get(math.MaxInt32, s.waiters)
				if err != nil {
					return err
				}
			}
			if size > streamQuota {
				size = streamQuota
			}

			// Get size worth quota from transport.
			tq, _, err := t.sendQuotaPool.get(size, s.waiters)
			if err != nil {
				return err
			}
			if tq < size {
				size = tq
			}
			ltq, _, err := t.localSendQuota.get(size, s.waiters)
			if err != nil {
				// Add the acquired quota back to transport.
				t.sendQuotaPool.add(tq)
				return err
			}
			// even if ltq is smaller than size we don't adjust size since
			// ltq is only a soft limit.
			streamQuota -= size
			p := r[:size]
			var endStream bool
			// See if this is the last frame to be written.
			if opts.Last {
				if len(r)-size == 0 { // No more data in r after this iteration.
					if idx == 0 { // We're writing data header.
						if len(data) == 0 { // There's no data to follow.
							endStream = true
						}
					} else { // We're writing data.
						endStream = true
					}
				}
			}
			success := func() {
				ltq := ltq
				t.controlBuf.put(&dataFrame{streamID: s.id, endStream: endStream, d: p, f: func() { t.localSendQuota.add(ltq) }})
				r = r[size:]
			}
			failure := func() { // The stream quota version must have changed.
				// Our streamQuota cache is invalidated now, so give it back.
				s.sendQuotaPool.lockedAdd(streamQuota + size)
			}
			if !s.sendQuotaPool.compareAndExecute(streamQuotaVer, success, failure) {
				// Couldn't send this chunk out.
				t.sendQuotaPool.add(size)
				t.localSendQuota.add(ltq)
				streamQuota = 0
			}
		}
	}
	if streamQuota > 0 { // Add the left over quota back to stream.
		s.sendQuotaPool.add(streamQuota)
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

// adjustWindow sends out extra window update over the initial window size
// of stream if the application is requesting data larger in size than
// the window.
func (t *http2Client) adjustWindow(s *Stream, n uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.state == streamDone {
		return
	}
	if w := s.fc.maybeAdjust(n); w > 0 {
		// Piggyback connection's window update along.
		if cw := t.fc.resetPendingUpdate(); cw > 0 {
			t.controlBuf.put(&windowUpdate{0, cw})
		}
		t.controlBuf.put(&windowUpdate{s.id, w})
	}
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
	if w := s.fc.onRead(n); w > 0 {
		if cw := t.fc.resetPendingUpdate(); cw > 0 {
			t.controlBuf.put(&windowUpdate{0, cw})
		}
		t.controlBuf.put(&windowUpdate{s.id, w})
	}
}

// updateFlowControl updates the incoming flow control windows
// for the transport and the stream based on the current bdp
// estimation.
func (t *http2Client) updateFlowControl(n uint32) {
	t.mu.Lock()
	for _, s := range t.activeStreams {
		s.fc.newLimit(n)
	}
	t.initialWindowSize = int32(n)
	t.mu.Unlock()
	t.controlBuf.put(&windowUpdate{0, t.fc.newLimit(n)})
	t.controlBuf.put(&settings{
		ss: []http2.Setting{
			{
				ID:  http2.SettingInitialWindowSize,
				Val: uint32(n),
			},
		},
	})
}

func (t *http2Client) handleData(f *http2.DataFrame) {
	size := f.Header().Length
	var sendBDPPing bool
	if t.bdpEst != nil {
		sendBDPPing = t.bdpEst.add(uint32(size))
	}
	// Decouple connection's flow control from application's read.
	// An update on connection's flow control should not depend on
	// whether user application has read the data or not. Such a
	// restriction is already imposed on the stream's flow control,
	// and therefore the sender will be blocked anyways.
	// Decoupling the connection flow control will prevent other
	// active(fast) streams from starving in presence of slow or
	// inactive streams.
	//
	// Furthermore, if a bdpPing is being sent out we can piggyback
	// connection's window update for the bytes we just received.
	if sendBDPPing {
		if size != 0 { // Could've been an empty data frame.
			t.controlBuf.put(&windowUpdate{0, uint32(size)})
		}
		t.controlBuf.put(bdpPing)
	} else {
		if err := t.fc.onData(uint32(size)); err != nil {
			t.Close()
			return
		}
		if w := t.fc.onRead(uint32(size)); w > 0 {
			t.controlBuf.put(&windowUpdate{0, w})
		}
	}
	// Select the right stream to dispatch.
	s, ok := t.getStream(f)
	if !ok {
		return
	}
	if size > 0 {
		s.mu.Lock()
		if s.state == streamDone {
			s.mu.Unlock()
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

	code := http2.ErrCode(f.ErrCode)
	if code == http2.ErrCodeRefusedStream {
		// The stream was unprocessed by the server.
		s.unprocessed = true
	}
	statusCode, ok := http2ErrConvTab[code]
	if !ok {
		warningf("transport: http2Client.handleRSTStream found no mapped gRPC status for the received http2 error %v", f.ErrCode)
		statusCode = codes.Unknown
	}
	s.finish(status.Newf(statusCode, "stream terminated by RST_STREAM with error code: %v", f.ErrCode))
	s.mu.Unlock()
	s.write(recvMsg{err: io.EOF})
}

func (t *http2Client) handleSettings(f *http2.SettingsFrame, isFirst bool) {
	if f.IsAck() {
		return
	}
	var rs []http2.Setting
	var ps []http2.Setting
	isMaxConcurrentStreamsMissing := true
	f.ForeachSetting(func(s http2.Setting) error {
		if s.ID == http2.SettingMaxConcurrentStreams {
			isMaxConcurrentStreamsMissing = false
		}
		if t.isRestrictive(s) {
			rs = append(rs, s)
		} else {
			ps = append(ps, s)
		}
		return nil
	})
	if isFirst && isMaxConcurrentStreamsMissing {
		// This means server is imposing no limits on
		// maximum number of concurrent streams initiated by client.
		// So we must remove our self-imposed limit.
		ps = append(ps, http2.Setting{
			ID:  http2.SettingMaxConcurrentStreams,
			Val: math.MaxUint32,
		})
	}
	t.applySettings(rs)
	t.controlBuf.put(&settingsAck{})
	t.applySettings(ps)
}

func (t *http2Client) isRestrictive(s http2.Setting) bool {
	switch s.ID {
	case http2.SettingMaxConcurrentStreams:
		return int(s.Val) < t.maxStreams
	case http2.SettingInitialWindowSize:
		// Note: we don't acquire a lock here to read streamSendQuota
		// because the same goroutine updates it later.
		return s.Val < t.streamSendQuota
	}
	return false
}

func (t *http2Client) handlePing(f *http2.PingFrame) {
	if f.IsAck() {
		// Maybe it's a BDP ping.
		if t.bdpEst != nil {
			t.bdpEst.calculate(f.Data)
		}
		return
	}
	pingAck := &ping{ack: true}
	copy(pingAck.data[:], f.Data[:])
	t.controlBuf.put(pingAck)
}

func (t *http2Client) handleGoAway(f *http2.GoAwayFrame) {
	t.mu.Lock()
	if t.state != reachable && t.state != draining {
		t.mu.Unlock()
		return
	}
	if f.ErrCode == http2.ErrCodeEnhanceYourCalm {
		infof("Client received GoAway with http2.ErrCodeEnhanceYourCalm.")
	}
	id := f.LastStreamID
	if id > 0 && id%2 != 1 {
		t.mu.Unlock()
		t.Close()
		return
	}
	// A client can receive multiple GoAways from the server (see
	// https://github.com/grpc/grpc-go/issues/1387).  The idea is that the first
	// GoAway will be sent with an ID of MaxInt32 and the second GoAway will be
	// sent after an RTT delay with the ID of the last stream the server will
	// process.
	//
	// Therefore, when we get the first GoAway we don't necessarily close any
	// streams. While in case of second GoAway we close all streams created after
	// the GoAwayId. This way streams that were in-flight while the GoAway from
	// server was being sent don't get killed.
	select {
	case <-t.goAway: // t.goAway has been closed (i.e.,multiple GoAways).
		// If there are multiple GoAways the first one should always have an ID greater than the following ones.
		if id > t.prevGoAwayID {
			t.mu.Unlock()
			t.Close()
			return
		}
	default:
		t.setGoAwayReason(f)
		close(t.goAway)
		t.state = draining
	}
	// All streams with IDs greater than the GoAwayId
	// and smaller than the previous GoAway ID should be killed.
	upperLimit := t.prevGoAwayID
	if upperLimit == 0 { // This is the first GoAway Frame.
		upperLimit = math.MaxUint32 // Kill all streams after the GoAway ID.
	}
	for streamID, stream := range t.activeStreams {
		if streamID > id && streamID <= upperLimit {
			// The stream was unprocessed by the server.
			stream.mu.Lock()
			stream.unprocessed = true
			stream.finish(statusGoAway)
			stream.mu.Unlock()
			close(stream.goAway)
		}
	}
	t.prevGoAwayID = id
	active := len(t.activeStreams)
	t.mu.Unlock()
	if active == 0 {
		t.Close()
	}
}

// setGoAwayReason sets the value of t.goAwayReason based
// on the GoAway frame received.
// It expects a lock on transport's mutext to be held by
// the caller.
func (t *http2Client) setGoAwayReason(f *http2.GoAwayFrame) {
	t.goAwayReason = GoAwayNoReason
	switch f.ErrCode {
	case http2.ErrCodeEnhanceYourCalm:
		if string(f.DebugData()) == "too_many_pings" {
			t.goAwayReason = GoAwayTooManyPings
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
	s.mu.Lock()
	s.bytesReceived = true
	s.mu.Unlock()
	var state decodeState
	if err := state.decodeResponseHeader(frame); err != nil {
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

	endStream := frame.StreamEnded()
	var isHeader bool
	defer func() {
		if t.statsHandler != nil {
			if isHeader {
				inHeader := &stats.InHeader{
					Client:     true,
					WireLength: int(frame.Header().Length),
				}
				t.statsHandler.HandleRPC(s.ctx, inHeader)
			} else {
				inTrailer := &stats.InTrailer{
					Client:     true,
					WireLength: int(frame.Header().Length),
				}
				t.statsHandler.HandleRPC(s.ctx, inTrailer)
			}
		}
	}()

	s.mu.Lock()
	if !s.headerDone {
		if !endStream {
			// Headers frame is not actually a trailers-only frame.
			isHeader = true
			s.recvCompress = state.encoding
			if len(state.mdata) > 0 {
				s.header = state.mdata
			}
		}
		close(s.headerChan)
		s.headerDone = true
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
	frame, err := t.framer.fr.ReadFrame()
	if err != nil {
		t.Close()
		return
	}
	atomic.CompareAndSwapUint32(&t.activity, 0, 1)
	sf, ok := frame.(*http2.SettingsFrame)
	if !ok {
		t.Close()
		return
	}
	t.onSuccess()
	t.handleSettings(sf, true)

	// loop to keep reading incoming messages on this transport.
	for {
		frame, err := t.framer.fr.ReadFrame()
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
					handleMalformedHTTP2(s, streamErrorf(http2ErrConvTab[se.Code], "%v", t.framer.fr.ErrorDetail()))
				}
				continue
			} else {
				// Transport error.
				t.Close()
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
			t.handleSettings(frame, false)
		case *http2.PingFrame:
			t.handlePing(frame)
		case *http2.GoAwayFrame:
			t.handleGoAway(frame)
		case *http2.WindowUpdateFrame:
			t.handleWindowUpdate(frame)
		default:
			errorf("transport: http2Client.reader got unhandled frame type %v.", frame)
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
			ms := t.maxStreams
			t.maxStreams = int(s.Val)
			t.streamsQuota.add(int(s.Val) - ms)
		case http2.SettingInitialWindowSize:
			t.mu.Lock()
			for _, stream := range t.activeStreams {
				// Adjust the sending quota for each stream.
				stream.sendQuotaPool.addAndUpdate(int(s.Val) - int(t.streamSendQuota))
			}
			t.streamSendQuota = s.Val
			t.mu.Unlock()
		}
	}
}

// TODO(mmukhi): A lot of this code(and code in other places in the tranpsort layer)
// is duplicated between the client and the server.
// The transport layer needs to be refactored to take care of this.
func (t *http2Client) itemHandler(i item) (err error) {
	defer func() {
		if err != nil {
			errorf(" error in itemHandler: %v", err)
		}
	}()
	switch i := i.(type) {
	case *dataFrame:
		if err := t.framer.fr.WriteData(i.streamID, i.endStream, i.d); err != nil {
			return err
		}
		i.f()
		return nil
	case *headerFrame:
		t.hBuf.Reset()
		for _, f := range i.hf {
			t.hEnc.WriteField(f)
		}
		endHeaders := false
		first := true
		for !endHeaders {
			size := t.hBuf.Len()
			if size > http2MaxFrameLen {
				size = http2MaxFrameLen
			} else {
				endHeaders = true
			}
			if first {
				first = false
				err = t.framer.fr.WriteHeaders(http2.HeadersFrameParam{
					StreamID:      i.streamID,
					BlockFragment: t.hBuf.Next(size),
					EndStream:     i.endStream,
					EndHeaders:    endHeaders,
				})
			} else {
				err = t.framer.fr.WriteContinuation(
					i.streamID,
					endHeaders,
					t.hBuf.Next(size),
				)
			}
			if err != nil {
				return err
			}
		}
		return nil
	case *windowUpdate:
		return t.framer.fr.WriteWindowUpdate(i.streamID, i.increment)
	case *settings:
		return t.framer.fr.WriteSettings(i.ss...)
	case *settingsAck:
		return t.framer.fr.WriteSettingsAck()
	case *resetStream:
		// If the server needs to be to intimated about stream closing,
		// then we need to make sure the RST_STREAM frame is written to
		// the wire before the headers of the next stream waiting on
		// streamQuota. We ensure this by adding to the streamsQuota pool
		// only after having acquired the writableChan to send RST_STREAM.
		err := t.framer.fr.WriteRSTStream(i.streamID, i.code)
		t.streamsQuota.add(1)
		return err
	case *flushIO:
		return t.framer.writer.Flush()
	case *ping:
		if !i.ack {
			t.bdpEst.timesnap(i.data)
		}
		return t.framer.fr.WritePing(i.ack, i.data)
	default:
		errorf("transport: http2Client.controller got unexpected item type %v", i)
		return fmt.Errorf("transport: http2Client.controller got unexpected item type %v", i)
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
				case <-t.ctx.Done():
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
			case <-t.ctx.Done():
				if !timer.Stop() {
					<-timer.C
				}
				return
			}
		case <-t.ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return
		}
	}
}

func (t *http2Client) Error() <-chan struct{} {
	return t.ctx.Done()
}

func (t *http2Client) GoAway() <-chan struct{} {
	return t.goAway
}
