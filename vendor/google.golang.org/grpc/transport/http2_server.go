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
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/internal/grpcrand"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/tap"
)

// ErrIllegalHeaderWrite indicates that setting header is illegal because of
// the stream's state.
var ErrIllegalHeaderWrite = errors.New("transport: the stream is done or WriteHeader was already called")

// http2Server implements the ServerTransport interface with HTTP2.
type http2Server struct {
	ctx         context.Context
	ctxDone     <-chan struct{} // Cache the context.Done() chan
	cancel      context.CancelFunc
	conn        net.Conn
	loopy       *loopyWriter
	readerDone  chan struct{} // sync point to enable testing.
	writerDone  chan struct{} // sync point to enable testing.
	remoteAddr  net.Addr
	localAddr   net.Addr
	maxStreamID uint32               // max stream ID ever seen
	authInfo    credentials.AuthInfo // auth info about the connection
	inTapHandle tap.ServerInHandle
	framer      *framer
	// The max number of concurrent streams.
	maxStreams uint32
	// controlBuf delivers all the control related tasks (e.g., window
	// updates, reset streams, and various settings) to the controller.
	controlBuf *controlBuffer
	fc         *trInFlow
	stats      stats.Handler
	// Flag to keep track of reading activity on transport.
	// 1 is true and 0 is false.
	activity uint32 // Accessed atomically.
	// Keepalive and max-age parameters for the server.
	kp keepalive.ServerParameters

	// Keepalive enforcement policy.
	kep keepalive.EnforcementPolicy
	// The time instance last ping was received.
	lastPingAt time.Time
	// Number of times the client has violated keepalive ping policy so far.
	pingStrikes uint8
	// Flag to signify that number of ping strikes should be reset to 0.
	// This is set whenever data or header frames are sent.
	// 1 means yes.
	resetPingStrikes  uint32 // Accessed atomically.
	initialWindowSize int32
	bdpEst            *bdpEstimator

	mu sync.Mutex // guard the following

	// drainChan is initialized when drain(...) is called the first time.
	// After which the server writes out the first GoAway(with ID 2^31-1) frame.
	// Then an independent goroutine will be launched to later send the second GoAway.
	// During this time we don't want to write another first GoAway(with ID 2^31 -1) frame.
	// Thus call to drain(...) will be a no-op if drainChan is already initialized since draining is
	// already underway.
	drainChan     chan struct{}
	state         transportState
	activeStreams map[uint32]*Stream
	// idle is the time instant when the connection went idle.
	// This is either the beginning of the connection or when the number of
	// RPCs go down to 0.
	// When the connection is busy, this value is set to 0.
	idle time.Time

	// Fields below are for channelz metric collection.
	channelzID int64 // channelz unique identification number
	czmu       sync.RWMutex
	kpCount    int64
	// The number of streams that have started, including already finished ones.
	streamsStarted int64
	// The number of streams that have ended successfully by sending frame with
	// EoS bit set.
	streamsSucceeded  int64
	streamsFailed     int64
	lastStreamCreated time.Time
	msgSent           int64
	msgRecv           int64
	lastMsgSent       time.Time
	lastMsgRecv       time.Time
}

// newHTTP2Server constructs a ServerTransport based on HTTP2. ConnectionError is
// returned if something goes wrong.
func newHTTP2Server(conn net.Conn, config *ServerConfig) (_ ServerTransport, err error) {
	writeBufSize := defaultWriteBufSize
	if config.WriteBufferSize > 0 {
		writeBufSize = config.WriteBufferSize
	}
	readBufSize := defaultReadBufSize
	if config.ReadBufferSize > 0 {
		readBufSize = config.ReadBufferSize
	}
	framer := newFramer(conn, writeBufSize, readBufSize)
	// Send initial settings as connection preface to client.
	var isettings []http2.Setting
	// TODO(zhaoq): Have a better way to signal "no limit" because 0 is
	// permitted in the HTTP2 spec.
	maxStreams := config.MaxStreams
	if maxStreams == 0 {
		maxStreams = math.MaxUint32
	} else {
		isettings = append(isettings, http2.Setting{
			ID:  http2.SettingMaxConcurrentStreams,
			Val: maxStreams,
		})
	}
	dynamicWindow := true
	iwz := int32(initialWindowSize)
	if config.InitialWindowSize >= defaultWindowSize {
		iwz = config.InitialWindowSize
		dynamicWindow = false
	}
	icwz := int32(initialWindowSize)
	if config.InitialConnWindowSize >= defaultWindowSize {
		icwz = config.InitialConnWindowSize
		dynamicWindow = false
	}
	if iwz != defaultWindowSize {
		isettings = append(isettings, http2.Setting{
			ID:  http2.SettingInitialWindowSize,
			Val: uint32(iwz)})
	}
	if err := framer.fr.WriteSettings(isettings...); err != nil {
		return nil, connectionErrorf(false, err, "transport: %v", err)
	}
	// Adjust the connection flow control window if needed.
	if delta := uint32(icwz - defaultWindowSize); delta > 0 {
		if err := framer.fr.WriteWindowUpdate(0, delta); err != nil {
			return nil, connectionErrorf(false, err, "transport: %v", err)
		}
	}
	kp := config.KeepaliveParams
	if kp.MaxConnectionIdle == 0 {
		kp.MaxConnectionIdle = defaultMaxConnectionIdle
	}
	if kp.MaxConnectionAge == 0 {
		kp.MaxConnectionAge = defaultMaxConnectionAge
	}
	// Add a jitter to MaxConnectionAge.
	kp.MaxConnectionAge += getJitter(kp.MaxConnectionAge)
	if kp.MaxConnectionAgeGrace == 0 {
		kp.MaxConnectionAgeGrace = defaultMaxConnectionAgeGrace
	}
	if kp.Time == 0 {
		kp.Time = defaultServerKeepaliveTime
	}
	if kp.Timeout == 0 {
		kp.Timeout = defaultServerKeepaliveTimeout
	}
	kep := config.KeepalivePolicy
	if kep.MinTime == 0 {
		kep.MinTime = defaultKeepalivePolicyMinTime
	}
	ctx, cancel := context.WithCancel(context.Background())
	t := &http2Server{
		ctx:               ctx,
		cancel:            cancel,
		ctxDone:           ctx.Done(),
		conn:              conn,
		remoteAddr:        conn.RemoteAddr(),
		localAddr:         conn.LocalAddr(),
		authInfo:          config.AuthInfo,
		framer:            framer,
		readerDone:        make(chan struct{}),
		writerDone:        make(chan struct{}),
		maxStreams:        maxStreams,
		inTapHandle:       config.InTapHandle,
		fc:                &trInFlow{limit: uint32(icwz)},
		state:             reachable,
		activeStreams:     make(map[uint32]*Stream),
		stats:             config.StatsHandler,
		kp:                kp,
		idle:              time.Now(),
		kep:               kep,
		initialWindowSize: iwz,
	}
	t.controlBuf = newControlBuffer(t.ctxDone)
	if dynamicWindow {
		t.bdpEst = &bdpEstimator{
			bdp:               initialWindowSize,
			updateFlowControl: t.updateFlowControl,
		}
	}
	if t.stats != nil {
		t.ctx = t.stats.TagConn(t.ctx, &stats.ConnTagInfo{
			RemoteAddr: t.remoteAddr,
			LocalAddr:  t.localAddr,
		})
		connBegin := &stats.ConnBegin{}
		t.stats.HandleConn(t.ctx, connBegin)
	}
	if channelz.IsOn() {
		t.channelzID = channelz.RegisterNormalSocket(t, config.ChannelzParentID, "")
	}
	t.framer.writer.Flush()

	defer func() {
		if err != nil {
			t.Close()
		}
	}()

	// Check the validity of client preface.
	preface := make([]byte, len(clientPreface))
	if _, err := io.ReadFull(t.conn, preface); err != nil {
		return nil, connectionErrorf(false, err, "transport: http2Server.HandleStreams failed to receive the preface from client: %v", err)
	}
	if !bytes.Equal(preface, clientPreface) {
		return nil, connectionErrorf(false, nil, "transport: http2Server.HandleStreams received bogus greeting from client: %q", preface)
	}

	frame, err := t.framer.fr.ReadFrame()
	if err == io.EOF || err == io.ErrUnexpectedEOF {
		return nil, err
	}
	if err != nil {
		return nil, connectionErrorf(false, err, "transport: http2Server.HandleStreams failed to read initial settings frame: %v", err)
	}
	atomic.StoreUint32(&t.activity, 1)
	sf, ok := frame.(*http2.SettingsFrame)
	if !ok {
		return nil, connectionErrorf(false, nil, "transport: http2Server.HandleStreams saw invalid preface type %T from client", frame)
	}
	t.handleSettings(sf)

	go func() {
		t.loopy = newLoopyWriter(serverSide, t.framer, t.controlBuf, t.bdpEst)
		t.loopy.ssGoAwayHandler = t.outgoingGoAwayHandler
		if err := t.loopy.run(); err != nil {
			errorf("transport: loopyWriter.run returning. Err: %v", err)
		}
		t.conn.Close()
		close(t.writerDone)
	}()
	go t.keepalive()
	return t, nil
}

// operateHeader takes action on the decoded headers.
func (t *http2Server) operateHeaders(frame *http2.MetaHeadersFrame, handle func(*Stream), traceCtx func(context.Context, string) context.Context) (close bool) {
	streamID := frame.Header().StreamID
	var state decodeState
	for _, hf := range frame.Fields {
		if err := state.processHeaderField(hf); err != nil {
			if se, ok := err.(StreamError); ok {
				t.controlBuf.put(&cleanupStream{
					streamID: streamID,
					rst:      true,
					rstCode:  statusCodeConvTab[se.Code],
					onWrite:  func() {},
				})
			}
			return
		}
	}

	buf := newRecvBuffer()
	s := &Stream{
		id:             streamID,
		st:             t,
		buf:            buf,
		fc:             &inFlow{limit: uint32(t.initialWindowSize)},
		recvCompress:   state.encoding,
		method:         state.method,
		contentSubtype: state.contentSubtype,
	}
	if frame.StreamEnded() {
		// s is just created by the caller. No lock needed.
		s.state = streamReadDone
	}
	if state.timeoutSet {
		s.ctx, s.cancel = context.WithTimeout(t.ctx, state.timeout)
	} else {
		s.ctx, s.cancel = context.WithCancel(t.ctx)
	}
	pr := &peer.Peer{
		Addr: t.remoteAddr,
	}
	// Attach Auth info if there is any.
	if t.authInfo != nil {
		pr.AuthInfo = t.authInfo
	}
	s.ctx = peer.NewContext(s.ctx, pr)
	// Attach the received metadata to the context.
	if len(state.mdata) > 0 {
		s.ctx = metadata.NewIncomingContext(s.ctx, state.mdata)
	}
	if state.statsTags != nil {
		s.ctx = stats.SetIncomingTags(s.ctx, state.statsTags)
	}
	if state.statsTrace != nil {
		s.ctx = stats.SetIncomingTrace(s.ctx, state.statsTrace)
	}
	if t.inTapHandle != nil {
		var err error
		info := &tap.Info{
			FullMethodName: state.method,
		}
		s.ctx, err = t.inTapHandle(s.ctx, info)
		if err != nil {
			warningf("transport: http2Server.operateHeaders got an error from InTapHandle: %v", err)
			t.controlBuf.put(&cleanupStream{
				streamID: s.id,
				rst:      true,
				rstCode:  http2.ErrCodeRefusedStream,
				onWrite:  func() {},
			})
			return
		}
	}
	t.mu.Lock()
	if t.state != reachable {
		t.mu.Unlock()
		return
	}
	if uint32(len(t.activeStreams)) >= t.maxStreams {
		t.mu.Unlock()
		t.controlBuf.put(&cleanupStream{
			streamID: streamID,
			rst:      true,
			rstCode:  http2.ErrCodeRefusedStream,
			onWrite:  func() {},
		})
		return
	}
	if streamID%2 != 1 || streamID <= t.maxStreamID {
		t.mu.Unlock()
		// illegal gRPC stream id.
		errorf("transport: http2Server.HandleStreams received an illegal stream id: %v", streamID)
		return true
	}
	t.maxStreamID = streamID
	t.activeStreams[streamID] = s
	if len(t.activeStreams) == 1 {
		t.idle = time.Time{}
	}
	t.mu.Unlock()
	if channelz.IsOn() {
		t.czmu.Lock()
		t.streamsStarted++
		t.lastStreamCreated = time.Now()
		t.czmu.Unlock()
	}
	s.requestRead = func(n int) {
		t.adjustWindow(s, uint32(n))
	}
	s.ctx = traceCtx(s.ctx, s.method)
	if t.stats != nil {
		s.ctx = t.stats.TagRPC(s.ctx, &stats.RPCTagInfo{FullMethodName: s.method})
		inHeader := &stats.InHeader{
			FullMethod:  s.method,
			RemoteAddr:  t.remoteAddr,
			LocalAddr:   t.localAddr,
			Compression: s.recvCompress,
			WireLength:  int(frame.Header().Length),
		}
		t.stats.HandleRPC(s.ctx, inHeader)
	}
	s.ctxDone = s.ctx.Done()
	s.wq = newWriteQuota(defaultWriteQuota, s.ctxDone)
	s.trReader = &transportReader{
		reader: &recvBufferReader{
			ctx:     s.ctx,
			ctxDone: s.ctxDone,
			recv:    s.buf,
		},
		windowHandler: func(n int) {
			t.updateWindow(s, uint32(n))
		},
	}
	// Register the stream with loopy.
	t.controlBuf.put(&registerStream{
		streamID: s.id,
		wq:       s.wq,
	})
	handle(s)
	return
}

// HandleStreams receives incoming streams using the given handler. This is
// typically run in a separate goroutine.
// traceCtx attaches trace to ctx and returns the new context.
func (t *http2Server) HandleStreams(handle func(*Stream), traceCtx func(context.Context, string) context.Context) {
	defer close(t.readerDone)
	for {
		frame, err := t.framer.fr.ReadFrame()
		atomic.StoreUint32(&t.activity, 1)
		if err != nil {
			if se, ok := err.(http2.StreamError); ok {
				warningf("transport: http2Server.HandleStreams encountered http2.StreamError: %v", se)
				t.mu.Lock()
				s := t.activeStreams[se.StreamID]
				t.mu.Unlock()
				if s != nil {
					t.closeStream(s, true, se.Code, nil, false)
				} else {
					t.controlBuf.put(&cleanupStream{
						streamID: se.StreamID,
						rst:      true,
						rstCode:  se.Code,
						onWrite:  func() {},
					})
				}
				continue
			}
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				t.Close()
				return
			}
			warningf("transport: http2Server.HandleStreams failed to read frame: %v", err)
			t.Close()
			return
		}
		switch frame := frame.(type) {
		case *http2.MetaHeadersFrame:
			if t.operateHeaders(frame, handle, traceCtx) {
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
			errorf("transport: http2Server.HandleStreams found unhandled frame type %v.", frame)
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

// adjustWindow sends out extra window update over the initial window size
// of stream if the application is requesting data larger in size than
// the window.
func (t *http2Server) adjustWindow(s *Stream, n uint32) {
	if w := s.fc.maybeAdjust(n); w > 0 {
		t.controlBuf.put(&outgoingWindowUpdate{streamID: s.id, increment: w})
	}

}

// updateWindow adjusts the inbound quota for the stream and the transport.
// Window updates will deliver to the controller for sending when
// the cumulative quota exceeds the corresponding threshold.
func (t *http2Server) updateWindow(s *Stream, n uint32) {
	if w := s.fc.onRead(n); w > 0 {
		t.controlBuf.put(&outgoingWindowUpdate{streamID: s.id,
			increment: w,
		})
	}
}

// updateFlowControl updates the incoming flow control windows
// for the transport and the stream based on the current bdp
// estimation.
func (t *http2Server) updateFlowControl(n uint32) {
	t.mu.Lock()
	for _, s := range t.activeStreams {
		s.fc.newLimit(n)
	}
	t.initialWindowSize = int32(n)
	t.mu.Unlock()
	t.controlBuf.put(&outgoingWindowUpdate{
		streamID:  0,
		increment: t.fc.newLimit(n),
	})
	t.controlBuf.put(&outgoingSettings{
		ss: []http2.Setting{
			{
				ID:  http2.SettingInitialWindowSize,
				Val: n,
			},
		},
	})

}

func (t *http2Server) handleData(f *http2.DataFrame) {
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
			t.closeStream(s, true, http2.ErrCodeFlowControl, nil, false)
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
	if f.Header().Flags.Has(http2.FlagDataEndStream) {
		// Received the end of stream from the client.
		s.compareAndSwapState(streamActive, streamReadDone)
		s.write(recvMsg{err: io.EOF})
	}
}

func (t *http2Server) handleRSTStream(f *http2.RSTStreamFrame) {
	s, ok := t.getStream(f)
	if !ok {
		return
	}
	t.closeStream(s, false, 0, nil, false)
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
	t.controlBuf.put(&incomingSettings{
		ss: ss,
	})
}

const (
	maxPingStrikes     = 2
	defaultPingTimeout = 2 * time.Hour
)

func (t *http2Server) handlePing(f *http2.PingFrame) {
	if f.IsAck() {
		if f.Data == goAwayPing.data && t.drainChan != nil {
			close(t.drainChan)
			return
		}
		// Maybe it's a BDP ping.
		if t.bdpEst != nil {
			t.bdpEst.calculate(f.Data)
		}
		return
	}
	pingAck := &ping{ack: true}
	copy(pingAck.data[:], f.Data[:])
	t.controlBuf.put(pingAck)

	now := time.Now()
	defer func() {
		t.lastPingAt = now
	}()
	// A reset ping strikes means that we don't need to check for policy
	// violation for this ping and the pingStrikes counter should be set
	// to 0.
	if atomic.CompareAndSwapUint32(&t.resetPingStrikes, 1, 0) {
		t.pingStrikes = 0
		return
	}
	t.mu.Lock()
	ns := len(t.activeStreams)
	t.mu.Unlock()
	if ns < 1 && !t.kep.PermitWithoutStream {
		// Keepalive shouldn't be active thus, this new ping should
		// have come after at least defaultPingTimeout.
		if t.lastPingAt.Add(defaultPingTimeout).After(now) {
			t.pingStrikes++
		}
	} else {
		// Check if keepalive policy is respected.
		if t.lastPingAt.Add(t.kep.MinTime).After(now) {
			t.pingStrikes++
		}
	}

	if t.pingStrikes > maxPingStrikes {
		// Send goaway and close the connection.
		errorf("transport: Got too many pings from the client, closing the connection.")
		t.controlBuf.put(&goAway{code: http2.ErrCodeEnhanceYourCalm, debugData: []byte("too_many_pings"), closeConn: true})
	}
}

func (t *http2Server) handleWindowUpdate(f *http2.WindowUpdateFrame) {
	t.controlBuf.put(&incomingWindowUpdate{
		streamID:  f.Header().StreamID,
		increment: f.Increment,
	})
}

func appendHeaderFieldsFromMD(headerFields []hpack.HeaderField, md metadata.MD) []hpack.HeaderField {
	for k, vv := range md {
		if isReservedHeader(k) {
			// Clients don't tolerate reading restricted headers after some non restricted ones were sent.
			continue
		}
		for _, v := range vv {
			headerFields = append(headerFields, hpack.HeaderField{Name: k, Value: encodeMetadataHeader(k, v)})
		}
	}
	return headerFields
}

// WriteHeader sends the header metedata md back to the client.
func (t *http2Server) WriteHeader(s *Stream, md metadata.MD) error {
	if s.updateHeaderSent() || s.getState() == streamDone {
		return ErrIllegalHeaderWrite
	}
	s.hdrMu.Lock()
	if md.Len() > 0 {
		if s.header.Len() > 0 {
			s.header = metadata.Join(s.header, md)
		} else {
			s.header = md
		}
	}
	t.writeHeaderLocked(s)
	s.hdrMu.Unlock()
	return nil
}

func (t *http2Server) writeHeaderLocked(s *Stream) {
	// TODO(mmukhi): Benchmark if the performance gets better if count the metadata and other header fields
	// first and create a slice of that exact size.
	headerFields := make([]hpack.HeaderField, 0, 2) // at least :status, content-type will be there if none else.
	headerFields = append(headerFields, hpack.HeaderField{Name: ":status", Value: "200"})
	headerFields = append(headerFields, hpack.HeaderField{Name: "content-type", Value: contentType(s.contentSubtype)})
	if s.sendCompress != "" {
		headerFields = append(headerFields, hpack.HeaderField{Name: "grpc-encoding", Value: s.sendCompress})
	}
	headerFields = appendHeaderFieldsFromMD(headerFields, s.header)
	t.controlBuf.put(&headerFrame{
		streamID:  s.id,
		hf:        headerFields,
		endStream: false,
		onWrite: func() {
			atomic.StoreUint32(&t.resetPingStrikes, 1)
		},
	})
	if t.stats != nil {
		// Note: WireLength is not set in outHeader.
		// TODO(mmukhi): Revisit this later, if needed.
		outHeader := &stats.OutHeader{}
		t.stats.HandleRPC(s.Context(), outHeader)
	}
}

// WriteStatus sends stream status to the client and terminates the stream.
// There is no further I/O operations being able to perform on this stream.
// TODO(zhaoq): Now it indicates the end of entire stream. Revisit if early
// OK is adopted.
func (t *http2Server) WriteStatus(s *Stream, st *status.Status) error {
	if s.getState() == streamDone {
		return nil
	}
	s.hdrMu.Lock()
	// TODO(mmukhi): Benchmark if the performance gets better if count the metadata and other header fields
	// first and create a slice of that exact size.
	headerFields := make([]hpack.HeaderField, 0, 2) // grpc-status and grpc-message will be there if none else.
	if !s.updateHeaderSent() {                      // No headers have been sent.
		if len(s.header) > 0 { // Send a separate header frame.
			t.writeHeaderLocked(s)
		} else { // Send a trailer only response.
			headerFields = append(headerFields, hpack.HeaderField{Name: ":status", Value: "200"})
			headerFields = append(headerFields, hpack.HeaderField{Name: "content-type", Value: contentType(s.contentSubtype)})
		}
	}
	headerFields = append(headerFields, hpack.HeaderField{Name: "grpc-status", Value: strconv.Itoa(int(st.Code()))})
	headerFields = append(headerFields, hpack.HeaderField{Name: "grpc-message", Value: encodeGrpcMessage(st.Message())})

	if p := st.Proto(); p != nil && len(p.Details) > 0 {
		stBytes, err := proto.Marshal(p)
		if err != nil {
			// TODO: return error instead, when callers are able to handle it.
			grpclog.Errorf("transport: failed to marshal rpc status: %v, error: %v", p, err)
		} else {
			headerFields = append(headerFields, hpack.HeaderField{Name: "grpc-status-details-bin", Value: encodeBinHeader(stBytes)})
		}
	}

	// Attach the trailer metadata.
	headerFields = appendHeaderFieldsFromMD(headerFields, s.trailer)
	trailingHeader := &headerFrame{
		streamID:  s.id,
		hf:        headerFields,
		endStream: true,
		onWrite: func() {
			atomic.StoreUint32(&t.resetPingStrikes, 1)
		},
	}
	s.hdrMu.Unlock()
	t.closeStream(s, false, 0, trailingHeader, true)
	if t.stats != nil {
		t.stats.HandleRPC(s.Context(), &stats.OutTrailer{})
	}
	return nil
}

// Write converts the data into HTTP2 data frame and sends it out. Non-nil error
// is returns if it fails (e.g., framing error, transport error).
func (t *http2Server) Write(s *Stream, hdr []byte, data []byte, opts *Options) error {
	if !s.isHeaderSent() { // Headers haven't been written yet.
		if err := t.WriteHeader(s, nil); err != nil {
			// TODO(mmukhi, dfawley): Make sure this is the right code to return.
			return streamErrorf(codes.Internal, "transport: %v", err)
		}
	} else {
		// Writing headers checks for this condition.
		if s.getState() == streamDone {
			// TODO(mmukhi, dfawley): Should the server write also return io.EOF?
			s.cancel()
			select {
			case <-t.ctx.Done():
				return ErrConnClosing
			default:
			}
			return ContextErr(s.ctx.Err())
		}
	}
	// Add some data to header frame so that we can equally distribute bytes across frames.
	emptyLen := http2MaxFrameLen - len(hdr)
	if emptyLen > len(data) {
		emptyLen = len(data)
	}
	hdr = append(hdr, data[:emptyLen]...)
	data = data[emptyLen:]
	df := &dataFrame{
		streamID: s.id,
		h:        hdr,
		d:        data,
		onEachWrite: func() {
			atomic.StoreUint32(&t.resetPingStrikes, 1)
		},
	}
	if err := s.wq.get(int32(len(hdr) + len(data))); err != nil {
		select {
		case <-t.ctx.Done():
			return ErrConnClosing
		default:
		}
		return ContextErr(s.ctx.Err())
	}
	return t.controlBuf.put(df)
}

// keepalive running in a separate goroutine does the following:
// 1. Gracefully closes an idle connection after a duration of keepalive.MaxConnectionIdle.
// 2. Gracefully closes any connection after a duration of keepalive.MaxConnectionAge.
// 3. Forcibly closes a connection after an additive period of keepalive.MaxConnectionAgeGrace over keepalive.MaxConnectionAge.
// 4. Makes sure a connection is alive by sending pings with a frequency of keepalive.Time and closes a non-responsive connection
// after an additional duration of keepalive.Timeout.
func (t *http2Server) keepalive() {
	p := &ping{}
	var pingSent bool
	maxIdle := time.NewTimer(t.kp.MaxConnectionIdle)
	maxAge := time.NewTimer(t.kp.MaxConnectionAge)
	keepalive := time.NewTimer(t.kp.Time)
	// NOTE: All exit paths of this function should reset their
	// respective timers. A failure to do so will cause the
	// following clean-up to deadlock and eventually leak.
	defer func() {
		if !maxIdle.Stop() {
			<-maxIdle.C
		}
		if !maxAge.Stop() {
			<-maxAge.C
		}
		if !keepalive.Stop() {
			<-keepalive.C
		}
	}()
	for {
		select {
		case <-maxIdle.C:
			t.mu.Lock()
			idle := t.idle
			if idle.IsZero() { // The connection is non-idle.
				t.mu.Unlock()
				maxIdle.Reset(t.kp.MaxConnectionIdle)
				continue
			}
			val := t.kp.MaxConnectionIdle - time.Since(idle)
			t.mu.Unlock()
			if val <= 0 {
				// The connection has been idle for a duration of keepalive.MaxConnectionIdle or more.
				// Gracefully close the connection.
				t.drain(http2.ErrCodeNo, []byte{})
				// Resetting the timer so that the clean-up doesn't deadlock.
				maxIdle.Reset(infinity)
				return
			}
			maxIdle.Reset(val)
		case <-maxAge.C:
			t.drain(http2.ErrCodeNo, []byte{})
			maxAge.Reset(t.kp.MaxConnectionAgeGrace)
			select {
			case <-maxAge.C:
				// Close the connection after grace period.
				t.Close()
				// Resetting the timer so that the clean-up doesn't deadlock.
				maxAge.Reset(infinity)
			case <-t.ctx.Done():
			}
			return
		case <-keepalive.C:
			if atomic.CompareAndSwapUint32(&t.activity, 1, 0) {
				pingSent = false
				keepalive.Reset(t.kp.Time)
				continue
			}
			if pingSent {
				t.Close()
				// Resetting the timer so that the clean-up doesn't deadlock.
				keepalive.Reset(infinity)
				return
			}
			pingSent = true
			if channelz.IsOn() {
				t.czmu.Lock()
				t.kpCount++
				t.czmu.Unlock()
			}
			t.controlBuf.put(p)
			keepalive.Reset(t.kp.Timeout)
		case <-t.ctx.Done():
			return
		}
	}
}

// Close starts shutting down the http2Server transport.
// TODO(zhaoq): Now the destruction is not blocked on any pending streams. This
// could cause some resource issue. Revisit this later.
func (t *http2Server) Close() error {
	t.mu.Lock()
	if t.state == closing {
		t.mu.Unlock()
		return errors.New("transport: Close() was already called")
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
	// Cancel all active streams.
	for _, s := range streams {
		s.cancel()
	}
	if t.stats != nil {
		connEnd := &stats.ConnEnd{}
		t.stats.HandleConn(t.ctx, connEnd)
	}
	return err
}

// closeStream clears the footprint of a stream when the stream is not needed
// any more.
func (t *http2Server) closeStream(s *Stream, rst bool, rstCode http2.ErrCode, hdr *headerFrame, eosReceived bool) {
	if s.swapState(streamDone) == streamDone {
		// If the stream was already done, return.
		return
	}
	// In case stream sending and receiving are invoked in separate
	// goroutines (e.g., bi-directional streaming), cancel needs to be
	// called to interrupt the potential blocking on other goroutines.
	s.cancel()
	cleanup := &cleanupStream{
		streamID: s.id,
		rst:      rst,
		rstCode:  rstCode,
		onWrite: func() {
			t.mu.Lock()
			if t.activeStreams != nil {
				delete(t.activeStreams, s.id)
				if len(t.activeStreams) == 0 {
					t.idle = time.Now()
				}
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
	}
	if hdr != nil {
		hdr.cleanup = cleanup
		t.controlBuf.put(hdr)
	} else {
		t.controlBuf.put(cleanup)
	}
}

func (t *http2Server) RemoteAddr() net.Addr {
	return t.remoteAddr
}

func (t *http2Server) Drain() {
	t.drain(http2.ErrCodeNo, []byte{})
}

func (t *http2Server) drain(code http2.ErrCode, debugData []byte) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.drainChan != nil {
		return
	}
	t.drainChan = make(chan struct{})
	t.controlBuf.put(&goAway{code: code, debugData: debugData, headsUp: true})
}

var goAwayPing = &ping{data: [8]byte{1, 6, 1, 8, 0, 3, 3, 9}}

// Handles outgoing GoAway and returns true if loopy needs to put itself
// in draining mode.
func (t *http2Server) outgoingGoAwayHandler(g *goAway) (bool, error) {
	t.mu.Lock()
	if t.state == closing { // TODO(mmukhi): This seems unnecessary.
		t.mu.Unlock()
		// The transport is closing.
		return false, ErrConnClosing
	}
	sid := t.maxStreamID
	if !g.headsUp {
		// Stop accepting more streams now.
		t.state = draining
		if len(t.activeStreams) == 0 {
			g.closeConn = true
		}
		t.mu.Unlock()
		if err := t.framer.fr.WriteGoAway(sid, g.code, g.debugData); err != nil {
			return false, err
		}
		if g.closeConn {
			// Abruptly close the connection following the GoAway (via
			// loopywriter).  But flush out what's inside the buffer first.
			t.framer.writer.Flush()
			return false, fmt.Errorf("transport: Connection closing")
		}
		return true, nil
	}
	t.mu.Unlock()
	// For a graceful close, send out a GoAway with stream ID of MaxUInt32,
	// Follow that with a ping and wait for the ack to come back or a timer
	// to expire. During this time accept new streams since they might have
	// originated before the GoAway reaches the client.
	// After getting the ack or timer expiration send out another GoAway this
	// time with an ID of the max stream server intends to process.
	if err := t.framer.fr.WriteGoAway(math.MaxUint32, http2.ErrCodeNo, []byte{}); err != nil {
		return false, err
	}
	if err := t.framer.fr.WritePing(false, goAwayPing.data); err != nil {
		return false, err
	}
	go func() {
		timer := time.NewTimer(time.Minute)
		defer timer.Stop()
		select {
		case <-t.drainChan:
		case <-timer.C:
		case <-t.ctx.Done():
			return
		}
		t.controlBuf.put(&goAway{code: g.code, debugData: g.debugData})
	}()
	return false, nil
}

func (t *http2Server) ChannelzMetric() *channelz.SocketInternalMetric {
	t.czmu.RLock()
	s := channelz.SocketInternalMetric{
		StreamsStarted:                   t.streamsStarted,
		StreamsSucceeded:                 t.streamsSucceeded,
		StreamsFailed:                    t.streamsFailed,
		MessagesSent:                     t.msgSent,
		MessagesReceived:                 t.msgRecv,
		KeepAlivesSent:                   t.kpCount,
		LastRemoteStreamCreatedTimestamp: t.lastStreamCreated,
		LastMessageSentTimestamp:         t.lastMsgSent,
		LastMessageReceivedTimestamp:     t.lastMsgRecv,
		LocalFlowControlWindow:           int64(t.fc.getSize()),
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

func (t *http2Server) IncrMsgSent() {
	t.czmu.Lock()
	t.msgSent++
	t.lastMsgSent = time.Now()
	t.czmu.Unlock()
}

func (t *http2Server) IncrMsgRecv() {
	t.czmu.Lock()
	t.msgRecv++
	t.lastMsgRecv = time.Now()
	t.czmu.Unlock()
}

func (t *http2Server) getOutFlowWindow() int64 {
	resp := make(chan uint32)
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

func getJitter(v time.Duration) time.Duration {
	if v == infinity {
		return 0
	}
	// Generate a jitter between +/- 10% of the value.
	r := int64(v / 10)
	j := grpcrand.Int63n(2*r) - r
	return time.Duration(j)
}
