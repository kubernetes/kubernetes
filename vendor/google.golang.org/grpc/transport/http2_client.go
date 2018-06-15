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

	"google.golang.org/grpc/channelz"
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
	ctxDone    <-chan struct{} // Cache the ctx.Done() chan.
	userAgent  string
	md         interface{}
	conn       net.Conn // underlying communication channel
	loopy      *loopyWriter
	remoteAddr net.Addr
	localAddr  net.Addr
	authInfo   credentials.AuthInfo // auth info about the connection

	readerDone chan struct{} // sync point to enable testing.
	writerDone chan struct{} // sync point to enable testing.
	// goAway is closed to notify the upper layer (i.e., addrConn.transportMonitor)
	// that the server sent GoAway on this transport.
	goAway chan struct{}
	// awakenKeepalive is used to wake up keepalive when after it has gone dormant.
	awakenKeepalive chan struct{}

	framer *framer
	// controlBuf delivers all the control related tasks (e.g., window
	// updates, reset streams, and various settings) to the controller.
	controlBuf *controlBuffer
	fc         *trInFlow
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

	bdpEst *bdpEstimator
	// onSuccess is a callback that client transport calls upon
	// receiving server preface to signal that a succefull HTTP2
	// connection was established.
	onSuccess func()

	maxConcurrentStreams  uint32
	streamQuota           int64
	streamsQuotaAvailable chan struct{}
	waitingStreams        uint32
	nextID                uint32

	mu            sync.Mutex // guard the following variables
	state         transportState
	activeStreams map[uint32]*Stream
	// prevGoAway ID records the Last-Stream-ID in the previous GOAway frame.
	prevGoAwayID uint32
	// goAwayReason records the http2.ErrCode and debug data received with the
	// GoAway frame.
	goAwayReason GoAwayReason

	// Fields below are for channelz metric collection.
	channelzID int64 // channelz unique identification number
	czmu       sync.RWMutex
	kpCount    int64
	// The number of streams that have started, including already finished ones.
	streamsStarted int64
	// The number of streams that have ended successfully by receiving EoS bit set
	// frame from server.
	streamsSucceeded  int64
	streamsFailed     int64
	lastStreamCreated time.Time
	msgSent           int64
	msgRecv           int64
	lastMsgSent       time.Time
	lastMsgRecv       time.Time
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
	writeBufSize := defaultWriteBufSize
	if opts.WriteBufferSize > 0 {
		writeBufSize = opts.WriteBufferSize
	}
	readBufSize := defaultReadBufSize
	if opts.ReadBufferSize > 0 {
		readBufSize = opts.ReadBufferSize
	}
	t := &http2Client{
		ctx:                   ctx,
		ctxDone:               ctx.Done(), // Cache Done chan.
		cancel:                cancel,
		userAgent:             opts.UserAgent,
		md:                    addr.Metadata,
		conn:                  conn,
		remoteAddr:            conn.RemoteAddr(),
		localAddr:             conn.LocalAddr(),
		authInfo:              authInfo,
		readerDone:            make(chan struct{}),
		writerDone:            make(chan struct{}),
		goAway:                make(chan struct{}),
		awakenKeepalive:       make(chan struct{}, 1),
		framer:                newFramer(conn, writeBufSize, readBufSize),
		fc:                    &trInFlow{limit: uint32(icwz)},
		scheme:                scheme,
		activeStreams:         make(map[uint32]*Stream),
		isSecure:              isSecure,
		creds:                 opts.PerRPCCredentials,
		kp:                    kp,
		statsHandler:          opts.StatsHandler,
		initialWindowSize:     initialWindowSize,
		onSuccess:             onSuccess,
		nextID:                1,
		maxConcurrentStreams:  defaultMaxStreamsClient,
		streamQuota:           defaultMaxStreamsClient,
		streamsQuotaAvailable: make(chan struct{}, 1),
	}
	t.controlBuf = newControlBuffer(t.ctxDone)
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
	if channelz.IsOn() {
		t.channelzID = channelz.RegisterNormalSocket(t, opts.ChannelzParentID, "")
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
		t.loopy = newLoopyWriter(clientSide, t.framer, t.controlBuf, t.bdpEst)
		t.loopy.run()
		t.conn.Close()
		close(t.writerDone)
	}()
	if t.kp.Time != infinity {
		go t.keepalive()
	}
	return t, nil
}

func (t *http2Client) newStream(ctx context.Context, callHdr *CallHdr) *Stream {
	// TODO(zhaoq): Handle uint32 overflow of Stream.id.
	s := &Stream{
		done:           make(chan struct{}),
		method:         callHdr.Method,
		sendCompress:   callHdr.SendCompress,
		buf:            newRecvBuffer(),
		headerChan:     make(chan struct{}),
		contentSubtype: callHdr.ContentSubtype,
	}
	s.wq = newWriteQuota(defaultWriteQuota, s.done)
	s.requestRead = func(n int) {
		t.adjustWindow(s, uint32(n))
	}
	// The client side stream context should have exactly the same life cycle with the user provided context.
	// That means, s.ctx should be read-only. And s.ctx is done iff ctx is done.
	// So we use the original context here instead of creating a copy.
	s.ctx = ctx
	s.trReader = &transportReader{
		reader: &recvBufferReader{
			ctx:     s.ctx,
			ctxDone: s.ctx.Done(),
			recv:    s.buf,
		},
		windowHandler: func(n int) {
			t.updateWindow(s, uint32(n))
		},
	}
	return s
}

func (t *http2Client) getPeer() *peer.Peer {
	pr := &peer.Peer{
		Addr: t.remoteAddr,
	}
	// Attach Auth info if there is any.
	if t.authInfo != nil {
		pr.AuthInfo = t.authInfo
	}
	return pr
}

func (t *http2Client) createHeaderFields(ctx context.Context, callHdr *CallHdr) ([]hpack.HeaderField, error) {
	aud := t.createAudience(callHdr)
	authData, err := t.getTrAuthData(ctx, aud)
	if err != nil {
		return nil, err
	}
	callAuthData, err := t.getCallAuthData(ctx, aud, callHdr)
	if err != nil {
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
	return headerFields, nil
}

func (t *http2Client) createAudience(callHdr *CallHdr) string {
	// Create an audience string only if needed.
	if len(t.creds) == 0 && callHdr.Creds == nil {
		return ""
	}
	// Construct URI required to get auth request metadata.
	// Omit port if it is the default one.
	host := strings.TrimSuffix(callHdr.Host, ":443")
	pos := strings.LastIndex(callHdr.Method, "/")
	if pos == -1 {
		pos = len(callHdr.Method)
	}
	return "https://" + host + callHdr.Method[:pos]
}

func (t *http2Client) getTrAuthData(ctx context.Context, audience string) (map[string]string, error) {
	authData := map[string]string{}
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
	return authData, nil
}

func (t *http2Client) getCallAuthData(ctx context.Context, audience string, callHdr *CallHdr) (map[string]string, error) {
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
	return callAuthData, nil
}

// NewStream creates a stream and registers it into the transport as "active"
// streams.
func (t *http2Client) NewStream(ctx context.Context, callHdr *CallHdr) (_ *Stream, err error) {
	ctx = peer.NewContext(ctx, t.getPeer())
	headerFields, err := t.createHeaderFields(ctx, callHdr)
	if err != nil {
		return nil, err
	}
	s := t.newStream(ctx, callHdr)
	cleanup := func(err error) {
		if s.swapState(streamDone) == streamDone {
			// If it was already done, return.
			return
		}
		// The stream was unprocessed by the server.
		atomic.StoreUint32(&s.unprocessed, 1)
		s.write(recvMsg{err: err})
		close(s.done)
		// If headerChan isn't closed, then close it.
		if atomic.SwapUint32(&s.headerDone, 1) == 0 {
			close(s.headerChan)
		}

	}
	hdr := &headerFrame{
		hf:        headerFields,
		endStream: false,
		initStream: func(id uint32) (bool, error) {
			t.mu.Lock()
			if state := t.state; state != reachable {
				t.mu.Unlock()
				// Do a quick cleanup.
				err := error(errStreamDrain)
				if state == closing {
					err = ErrConnClosing
				}
				cleanup(err)
				return false, err
			}
			t.activeStreams[id] = s
			if channelz.IsOn() {
				t.czmu.Lock()
				t.streamsStarted++
				t.lastStreamCreated = time.Now()
				t.czmu.Unlock()
			}
			var sendPing bool
			// If the number of active streams change from 0 to 1, then check if keepalive
			// has gone dormant. If so, wake it up.
			if len(t.activeStreams) == 1 {
				select {
				case t.awakenKeepalive <- struct{}{}:
					sendPing = true
					// Fill the awakenKeepalive channel again as this channel must be
					// kept non-writable except at the point that the keepalive()
					// goroutine is waiting either to be awaken or shutdown.
					t.awakenKeepalive <- struct{}{}
				default:
				}
			}
			t.mu.Unlock()
			return sendPing, nil
		},
		onOrphaned: cleanup,
		wq:         s.wq,
	}
	firstTry := true
	var ch chan struct{}
	checkForStreamQuota := func(it interface{}) bool {
		if t.streamQuota <= 0 { // Can go negative if server decreases it.
			if firstTry {
				t.waitingStreams++
			}
			ch = t.streamsQuotaAvailable
			return false
		}
		if !firstTry {
			t.waitingStreams--
		}
		t.streamQuota--
		h := it.(*headerFrame)
		h.streamID = t.nextID
		t.nextID += 2
		s.id = h.streamID
		s.fc = &inFlow{limit: uint32(t.initialWindowSize)}
		if t.streamQuota > 0 && t.waitingStreams > 0 {
			select {
			case t.streamsQuotaAvailable <- struct{}{}:
			default:
			}
		}
		return true
	}
	for {
		success, err := t.controlBuf.executeAndPut(checkForStreamQuota, hdr)
		if err != nil {
			return nil, err
		}
		if success {
			break
		}
		firstTry = false
		select {
		case <-ch:
		case <-s.ctx.Done():
			return nil, ContextErr(s.ctx.Err())
		case <-t.goAway:
			return nil, errStreamDrain
		case <-t.ctx.Done():
			return nil, ErrConnClosing
		}
	}
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
	var (
		rst     bool
		rstCode http2.ErrCode
	)
	if err != nil {
		rst = true
		rstCode = http2.ErrCodeCancel
	}
	t.closeStream(s, err, rst, rstCode, nil, nil, false)
}

func (t *http2Client) closeStream(s *Stream, err error, rst bool, rstCode http2.ErrCode, st *status.Status, mdata map[string][]string, eosReceived bool) {
	// Set stream status to done.
	if s.swapState(streamDone) == streamDone {
		// If it was already done, return.
		return
	}
	// status and trailers can be updated here without any synchronization because the stream goroutine will
	// only read it after it sees an io.EOF error from read or write and we'll write those errors
	// only after updating this.
	s.status = st
	if len(mdata) > 0 {
		s.trailer = mdata
	}
	if err != nil {
		// This will unblock reads eventually.
		s.write(recvMsg{err: err})
	}
	// This will unblock write.
	close(s.done)
	// If headerChan isn't closed, then close it.
	if atomic.SwapUint32(&s.headerDone, 1) == 0 {
		close(s.headerChan)
	}
	cleanup := &cleanupStream{
		streamID: s.id,
		onWrite: func() {
			t.mu.Lock()
			if t.activeStreams != nil {
				delete(t.activeStreams, s.id)
			}
			t.mu.Unlock()
			if channelz.IsOn() {
				t.czmu.Lock()
				if eosReceived {
					t.streamsSucceeded++
				} else {
					t.streamsFailed++
				}
				t.czmu.Unlock()
			}
		},
		rst:     rst,
		rstCode: rstCode,
	}
	addBackStreamQuota := func(interface{}) bool {
		t.streamQuota++
		if t.streamQuota > 0 && t.waitingStreams > 0 {
			select {
			case t.streamsQuotaAvailable <- struct{}{}:
			default:
			}
		}
		return true
	}
	t.controlBuf.executeAndPut(addBackStreamQuota, cleanup)
}

// Close kicks off the shutdown process of the transport. This should be called
// only once on a transport. Once it is called, the transport should not be
// accessed any more.
func (t *http2Client) Close() error {
	t.mu.Lock()
	// Make sure we only Close once.
	if t.state == closing {
		t.mu.Unlock()
		return nil
	}
	t.state = closing
	streams := t.activeStreams
	t.activeStreams = nil
	t.mu.Unlock()
	t.controlBuf.finish()
	t.cancel()
	err := t.conn.Close()
	if channelz.IsOn() {
		channelz.RemoveEntry(t.channelzID)
	}
	// Notify all active streams.
	for _, s := range streams {
		t.closeStream(s, ErrConnClosing, false, http2.ErrCodeNo, nil, nil, false)
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
	// Make sure we move to draining only from active.
	if t.state == draining || t.state == closing {
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
	if opts.Last {
		// If it's the last message, update stream state.
		if !s.compareAndSwapState(streamActive, streamWriteDone) {
			return errStreamDone
		}
	} else if s.getState() != streamActive {
		return errStreamDone
	}
	df := &dataFrame{
		streamID:  s.id,
		endStream: opts.Last,
	}
	if hdr != nil || data != nil { // If it's not an empty data frame.
		// Add some data to grpc message header so that we can equally
		// distribute bytes across frames.
		emptyLen := http2MaxFrameLen - len(hdr)
		if emptyLen > len(data) {
			emptyLen = len(data)
		}
		hdr = append(hdr, data[:emptyLen]...)
		data = data[emptyLen:]
		df.h, df.d = hdr, data
		// TODO(mmukhi): The above logic in this if can be moved to loopyWriter's data handler.
		if err := s.wq.get(int32(len(hdr) + len(data))); err != nil {
			return err
		}
	}
	return t.controlBuf.put(df)
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
	if w := s.fc.maybeAdjust(n); w > 0 {
		t.controlBuf.put(&outgoingWindowUpdate{streamID: s.id, increment: w})
	}
}

// updateWindow adjusts the inbound quota for the stream.
// Window updates will be sent out when the cumulative quota
// exceeds the corresponding threshold.
func (t *http2Client) updateWindow(s *Stream, n uint32) {
	if w := s.fc.onRead(n); w > 0 {
		t.controlBuf.put(&outgoingWindowUpdate{streamID: s.id, increment: w})
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
	t.mu.Unlock()
	updateIWS := func(interface{}) bool {
		t.initialWindowSize = int32(n)
		return true
	}
	t.controlBuf.executeAndPut(updateIWS, &outgoingWindowUpdate{streamID: 0, increment: t.fc.newLimit(n)})
	t.controlBuf.put(&outgoingSettings{
		ss: []http2.Setting{
			{
				ID:  http2.SettingInitialWindowSize,
				Val: n,
			},
		},
	})
}

func (t *http2Client) handleData(f *http2.DataFrame) {
	size := f.Header().Length
	var sendBDPPing bool
	if t.bdpEst != nil {
		sendBDPPing = t.bdpEst.add(size)
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
	if w := t.fc.onData(size); w > 0 {
		t.controlBuf.put(&outgoingWindowUpdate{
			streamID:  0,
			increment: w,
		})
	}
	if sendBDPPing {
		// Avoid excessive ping detection (e.g. in an L7 proxy)
		// by sending a window update prior to the BDP ping.

		if w := t.fc.reset(); w > 0 {
			t.controlBuf.put(&outgoingWindowUpdate{
				streamID:  0,
				increment: w,
			})
		}

		t.controlBuf.put(bdpPing)
	}
	// Select the right stream to dispatch.
	s, ok := t.getStream(f)
	if !ok {
		return
	}
	if size > 0 {
		if err := s.fc.onData(size); err != nil {
			t.closeStream(s, io.EOF, true, http2.ErrCodeFlowControl, status.New(codes.Internal, err.Error()), nil, false)
			return
		}
		if f.Header().Flags.Has(http2.FlagDataPadded) {
			if w := s.fc.onRead(size - uint32(len(f.Data()))); w > 0 {
				t.controlBuf.put(&outgoingWindowUpdate{s.id, w})
			}
		}
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
		t.closeStream(s, io.EOF, false, http2.ErrCodeNo, status.New(codes.Internal, "server closed the stream without sending trailers"), nil, true)
	}
}

func (t *http2Client) handleRSTStream(f *http2.RSTStreamFrame) {
	s, ok := t.getStream(f)
	if !ok {
		return
	}
	if f.ErrCode == http2.ErrCodeRefusedStream {
		// The stream was unprocessed by the server.
		atomic.StoreUint32(&s.unprocessed, 1)
	}
	statusCode, ok := http2ErrConvTab[f.ErrCode]
	if !ok {
		warningf("transport: http2Client.handleRSTStream found no mapped gRPC status for the received http2 error %v", f.ErrCode)
		statusCode = codes.Unknown
	}
	t.closeStream(s, io.EOF, false, http2.ErrCodeNo, status.Newf(statusCode, "stream terminated by RST_STREAM with error code: %v", f.ErrCode), nil, false)
}

func (t *http2Client) handleSettings(f *http2.SettingsFrame, isFirst bool) {
	if f.IsAck() {
		return
	}
	var maxStreams *uint32
	var ss []http2.Setting
	f.ForeachSetting(func(s http2.Setting) error {
		if s.ID == http2.SettingMaxConcurrentStreams {
			maxStreams = new(uint32)
			*maxStreams = s.Val
			return nil
		}
		ss = append(ss, s)
		return nil
	})
	if isFirst && maxStreams == nil {
		maxStreams = new(uint32)
		*maxStreams = math.MaxUint32
	}
	sf := &incomingSettings{
		ss: ss,
	}
	if maxStreams == nil {
		t.controlBuf.put(sf)
		return
	}
	updateStreamQuota := func(interface{}) bool {
		delta := int64(*maxStreams) - int64(t.maxConcurrentStreams)
		t.maxConcurrentStreams = *maxStreams
		t.streamQuota += delta
		if delta > 0 && t.waitingStreams > 0 {
			close(t.streamsQuotaAvailable) // wake all of them up.
			t.streamsQuotaAvailable = make(chan struct{}, 1)
		}
		return true
	}
	t.controlBuf.executeAndPut(updateStreamQuota, sf)
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
	if t.state == closing {
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
		t.controlBuf.put(&incomingGoAway{})
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
			atomic.StoreUint32(&stream.unprocessed, 1)
			t.closeStream(stream, errStreamDrain, false, http2.ErrCodeNo, statusGoAway, nil, false)
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
	t.controlBuf.put(&incomingWindowUpdate{
		streamID:  f.Header().StreamID,
		increment: f.Increment,
	})
}

// operateHeaders takes action on the decoded headers.
func (t *http2Client) operateHeaders(frame *http2.MetaHeadersFrame) {
	s, ok := t.getStream(frame)
	if !ok {
		return
	}
	atomic.StoreUint32(&s.bytesReceived, 1)
	var state decodeState
	if err := state.decodeResponseHeader(frame); err != nil {
		t.closeStream(s, err, true, http2.ErrCodeProtocol, nil, nil, false)
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
	// If headers haven't been received yet.
	if atomic.SwapUint32(&s.headerDone, 1) == 0 {
		if !endStream {
			// Headers frame is not actually a trailers-only frame.
			isHeader = true
			// These values can be set without any synchronization because
			// stream goroutine will read it only after seeing a closed
			// headerChan which we'll close after setting this.
			s.recvCompress = state.encoding
			if len(state.mdata) > 0 {
				s.header = state.mdata
			}
		}
		close(s.headerChan)
	}
	if !endStream {
		return
	}
	t.closeStream(s, io.EOF, false, http2.ErrCodeNo, state.status(), state.mdata, true)
}

// reader runs as a separate goroutine in charge of reading data from network
// connection.
//
// TODO(zhaoq): currently one reader per transport. Investigate whether this is
// optimal.
// TODO(zhaoq): Check the validity of the incoming frame sequence.
func (t *http2Client) reader() {
	defer close(t.readerDone)
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
					t.closeStream(s, streamErrorf(http2ErrConvTab[se.Code], "%v", t.framer.fr.ErrorDetail()), true, http2.ErrCodeProtocol, nil, nil, false)
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
				if channelz.IsOn() {
					t.czmu.Lock()
					t.kpCount++
					t.czmu.Unlock()
				}
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

func (t *http2Client) ChannelzMetric() *channelz.SocketInternalMetric {
	t.czmu.RLock()
	s := channelz.SocketInternalMetric{
		StreamsStarted:                  t.streamsStarted,
		StreamsSucceeded:                t.streamsSucceeded,
		StreamsFailed:                   t.streamsFailed,
		MessagesSent:                    t.msgSent,
		MessagesReceived:                t.msgRecv,
		KeepAlivesSent:                  t.kpCount,
		LastLocalStreamCreatedTimestamp: t.lastStreamCreated,
		LastMessageSentTimestamp:        t.lastMsgSent,
		LastMessageReceivedTimestamp:    t.lastMsgRecv,
		LocalFlowControlWindow:          int64(t.fc.getSize()),
		//socket options
		LocalAddr:  t.localAddr,
		RemoteAddr: t.remoteAddr,
		// Security
		// RemoteName :
	}
	t.czmu.RUnlock()
	s.RemoteFlowControlWindow = t.getOutFlowWindow()
	return &s
}

func (t *http2Client) IncrMsgSent() {
	t.czmu.Lock()
	t.msgSent++
	t.lastMsgSent = time.Now()
	t.czmu.Unlock()
}

func (t *http2Client) IncrMsgRecv() {
	t.czmu.Lock()
	t.msgRecv++
	t.lastMsgRecv = time.Now()
	t.czmu.Unlock()
}

func (t *http2Client) getOutFlowWindow() int64 {
	resp := make(chan uint32, 1)
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	t.controlBuf.put(&outFlowControlSizeRequest{resp})
	select {
	case sz := <-resp:
		return int64(sz)
	case <-t.ctxDone:
		return -1
	case <-timer.C:
		return -2
	}
}
