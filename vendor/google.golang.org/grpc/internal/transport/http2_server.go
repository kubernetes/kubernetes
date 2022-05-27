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
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
	"google.golang.org/grpc/internal/grpcutil"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/internal/grpcrand"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/tap"
)

var (
	// ErrIllegalHeaderWrite indicates that setting header is illegal because of
	// the stream's state.
	ErrIllegalHeaderWrite = errors.New("transport: the stream is done or WriteHeader was already called")
	// ErrHeaderListSizeLimitViolation indicates that the header list size is larger
	// than the limit set by peer.
	ErrHeaderListSizeLimitViolation = errors.New("transport: trying to send header list size larger than the limit set by peer")
)

// serverConnectionCounter counts the number of connections a server has seen
// (equal to the number of http2Servers created). Must be accessed atomically.
var serverConnectionCounter uint64

// http2Server implements the ServerTransport interface with HTTP2.
type http2Server struct {
	lastRead    int64 // Keep this field 64-bit aligned. Accessed atomically.
	ctx         context.Context
	done        chan struct{}
	conn        net.Conn
	loopy       *loopyWriter
	readerDone  chan struct{} // sync point to enable testing.
	writerDone  chan struct{} // sync point to enable testing.
	remoteAddr  net.Addr
	localAddr   net.Addr
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
	resetPingStrikes      uint32 // Accessed atomically.
	initialWindowSize     int32
	bdpEst                *bdpEstimator
	maxSendHeaderListSize *uint32

	mu sync.Mutex // guard the following

	// drainChan is initialized when Drain() is called the first time.
	// After which the server writes out the first GoAway(with ID 2^31-1) frame.
	// Then an independent goroutine will be launched to later send the second GoAway.
	// During this time we don't want to write another first GoAway(with ID 2^31 -1) frame.
	// Thus call to Drain() will be a no-op if drainChan is already initialized since draining is
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
	czData     *channelzData
	bufferPool *bufferPool

	connectionID uint64

	// maxStreamMu guards the maximum stream ID
	// This lock may not be taken if mu is already held.
	maxStreamMu sync.Mutex
	maxStreamID uint32 // max stream ID ever seen
}

// NewServerTransport creates a http2 transport with conn and configuration
// options from config.
//
// It returns a non-nil transport and a nil error on success. On failure, it
// returns a nil transport and a non-nil error. For a special case where the
// underlying conn gets closed before the client preface could be read, it
// returns a nil transport and a nil error.
func NewServerTransport(conn net.Conn, config *ServerConfig) (_ ServerTransport, err error) {
	var authInfo credentials.AuthInfo
	rawConn := conn
	if config.Credentials != nil {
		var err error
		conn, authInfo, err = config.Credentials.ServerHandshake(rawConn)
		if err != nil {
			// ErrConnDispatched means that the connection was dispatched away
			// from gRPC; those connections should be left open. io.EOF means
			// the connection was closed before handshaking completed, which can
			// happen naturally from probers. Return these errors directly.
			if err == credentials.ErrConnDispatched || err == io.EOF {
				return nil, err
			}
			return nil, connectionErrorf(false, err, "ServerHandshake(%q) failed: %v", rawConn.RemoteAddr(), err)
		}
	}
	writeBufSize := config.WriteBufferSize
	readBufSize := config.ReadBufferSize
	maxHeaderListSize := defaultServerMaxHeaderListSize
	if config.MaxHeaderListSize != nil {
		maxHeaderListSize = *config.MaxHeaderListSize
	}
	framer := newFramer(conn, writeBufSize, readBufSize, maxHeaderListSize)
	// Send initial settings as connection preface to client.
	isettings := []http2.Setting{{
		ID:  http2.SettingMaxFrameSize,
		Val: http2MaxFrameLen,
	}}
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
	if config.MaxHeaderListSize != nil {
		isettings = append(isettings, http2.Setting{
			ID:  http2.SettingMaxHeaderListSize,
			Val: *config.MaxHeaderListSize,
		})
	}
	if config.HeaderTableSize != nil {
		isettings = append(isettings, http2.Setting{
			ID:  http2.SettingHeaderTableSize,
			Val: *config.HeaderTableSize,
		})
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

	done := make(chan struct{})
	t := &http2Server{
		ctx:               setConnection(context.Background(), rawConn),
		done:              done,
		conn:              conn,
		remoteAddr:        conn.RemoteAddr(),
		localAddr:         conn.LocalAddr(),
		authInfo:          authInfo,
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
		czData:            new(channelzData),
		bufferPool:        newBufferPool(),
	}
	t.controlBuf = newControlBuffer(t.done)
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
		t.channelzID = channelz.RegisterNormalSocket(t, config.ChannelzParentID, fmt.Sprintf("%s -> %s", t.remoteAddr, t.localAddr))
	}

	t.connectionID = atomic.AddUint64(&serverConnectionCounter, 1)

	t.framer.writer.Flush()

	defer func() {
		if err != nil {
			t.Close()
		}
	}()

	// Check the validity of client preface.
	preface := make([]byte, len(clientPreface))
	if _, err := io.ReadFull(t.conn, preface); err != nil {
		// In deployments where a gRPC server runs behind a cloud load balancer
		// which performs regular TCP level health checks, the connection is
		// closed immediately by the latter.  Returning io.EOF here allows the
		// grpc server implementation to recognize this scenario and suppress
		// logging to reduce spam.
		if err == io.EOF {
			return nil, io.EOF
		}
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
	atomic.StoreInt64(&t.lastRead, time.Now().UnixNano())
	sf, ok := frame.(*http2.SettingsFrame)
	if !ok {
		return nil, connectionErrorf(false, nil, "transport: http2Server.HandleStreams saw invalid preface type %T from client", frame)
	}
	t.handleSettings(sf)

	go func() {
		t.loopy = newLoopyWriter(serverSide, t.framer, t.controlBuf, t.bdpEst)
		t.loopy.ssGoAwayHandler = t.outgoingGoAwayHandler
		if err := t.loopy.run(); err != nil {
			if logger.V(logLevel) {
				logger.Errorf("transport: loopyWriter.run returning. Err: %v", err)
			}
		}
		t.conn.Close()
		t.controlBuf.finish()
		close(t.writerDone)
	}()
	go t.keepalive()
	return t, nil
}

// operateHeader takes action on the decoded headers.
func (t *http2Server) operateHeaders(frame *http2.MetaHeadersFrame, handle func(*Stream), traceCtx func(context.Context, string) context.Context) (fatal bool) {
	// Acquire max stream ID lock for entire duration
	t.maxStreamMu.Lock()
	defer t.maxStreamMu.Unlock()

	streamID := frame.Header().StreamID

	// frame.Truncated is set to true when framer detects that the current header
	// list size hits MaxHeaderListSize limit.
	if frame.Truncated {
		t.controlBuf.put(&cleanupStream{
			streamID: streamID,
			rst:      true,
			rstCode:  http2.ErrCodeFrameSize,
			onWrite:  func() {},
		})
		return false
	}

	if streamID%2 != 1 || streamID <= t.maxStreamID {
		// illegal gRPC stream id.
		if logger.V(logLevel) {
			logger.Errorf("transport: http2Server.HandleStreams received an illegal stream id: %v", streamID)
		}
		return true
	}
	t.maxStreamID = streamID

	buf := newRecvBuffer()
	s := &Stream{
		id:  streamID,
		st:  t,
		buf: buf,
		fc:  &inFlow{limit: uint32(t.initialWindowSize)},
	}
	var (
		// If a gRPC Response-Headers has already been received, then it means
		// that the peer is speaking gRPC and we are in gRPC mode.
		isGRPC     = false
		mdata      = make(map[string][]string)
		httpMethod string
		// headerError is set if an error is encountered while parsing the headers
		headerError bool

		timeoutSet bool
		timeout    time.Duration
	)

	for _, hf := range frame.Fields {
		switch hf.Name {
		case "content-type":
			contentSubtype, validContentType := grpcutil.ContentSubtype(hf.Value)
			if !validContentType {
				break
			}
			mdata[hf.Name] = append(mdata[hf.Name], hf.Value)
			s.contentSubtype = contentSubtype
			isGRPC = true
		case "grpc-encoding":
			s.recvCompress = hf.Value
		case ":method":
			httpMethod = hf.Value
		case ":path":
			s.method = hf.Value
		case "grpc-timeout":
			timeoutSet = true
			var err error
			if timeout, err = decodeTimeout(hf.Value); err != nil {
				headerError = true
			}
		// "Transports must consider requests containing the Connection header
		// as malformed." - A41
		case "connection":
			if logger.V(logLevel) {
				logger.Errorf("transport: http2Server.operateHeaders parsed a :connection header which makes a request malformed as per the HTTP/2 spec")
			}
			headerError = true
		default:
			if isReservedHeader(hf.Name) && !isWhitelistedHeader(hf.Name) {
				break
			}
			v, err := decodeMetadataHeader(hf.Name, hf.Value)
			if err != nil {
				headerError = true
				logger.Warningf("Failed to decode metadata header (%q, %q): %v", hf.Name, hf.Value, err)
				break
			}
			mdata[hf.Name] = append(mdata[hf.Name], v)
		}
	}

	// "If multiple Host headers or multiple :authority headers are present, the
	// request must be rejected with an HTTP status code 400 as required by Host
	// validation in RFC 7230 §5.4, gRPC status code INTERNAL, or RST_STREAM
	// with HTTP/2 error code PROTOCOL_ERROR." - A41. Since this is a HTTP/2
	// error, this takes precedence over a client not speaking gRPC.
	if len(mdata[":authority"]) > 1 || len(mdata["host"]) > 1 {
		errMsg := fmt.Sprintf("num values of :authority: %v, num values of host: %v, both must only have 1 value as per HTTP/2 spec", len(mdata[":authority"]), len(mdata["host"]))
		if logger.V(logLevel) {
			logger.Errorf("transport: %v", errMsg)
		}
		t.controlBuf.put(&earlyAbortStream{
			httpStatus:     400,
			streamID:       streamID,
			contentSubtype: s.contentSubtype,
			status:         status.New(codes.Internal, errMsg),
		})
		return false
	}

	if !isGRPC || headerError {
		t.controlBuf.put(&cleanupStream{
			streamID: streamID,
			rst:      true,
			rstCode:  http2.ErrCodeProtocol,
			onWrite:  func() {},
		})
		return false
	}

	// "If :authority is missing, Host must be renamed to :authority." - A41
	if len(mdata[":authority"]) == 0 {
		// No-op if host isn't present, no eventual :authority header is a valid
		// RPC.
		if host, ok := mdata["host"]; ok {
			mdata[":authority"] = host
			delete(mdata, "host")
		}
	} else {
		// "If :authority is present, Host must be discarded" - A41
		delete(mdata, "host")
	}

	if frame.StreamEnded() {
		// s is just created by the caller. No lock needed.
		s.state = streamReadDone
	}
	if timeoutSet {
		s.ctx, s.cancel = context.WithTimeout(t.ctx, timeout)
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
	if len(mdata) > 0 {
		s.ctx = metadata.NewIncomingContext(s.ctx, mdata)
		if statsTags := mdata["grpc-tags-bin"]; len(statsTags) > 0 {
			s.ctx = stats.SetIncomingTags(s.ctx, []byte(statsTags[len(statsTags)-1]))
		}
		if statsTrace := mdata["grpc-trace-bin"]; len(statsTrace) > 0 {
			s.ctx = stats.SetIncomingTrace(s.ctx, []byte(statsTrace[len(statsTrace)-1]))
		}
	}
	t.mu.Lock()
	if t.state != reachable {
		t.mu.Unlock()
		s.cancel()
		return false
	}
	if uint32(len(t.activeStreams)) >= t.maxStreams {
		t.mu.Unlock()
		t.controlBuf.put(&cleanupStream{
			streamID: streamID,
			rst:      true,
			rstCode:  http2.ErrCodeRefusedStream,
			onWrite:  func() {},
		})
		s.cancel()
		return false
	}
	if httpMethod != http.MethodPost {
		t.mu.Unlock()
		if logger.V(logLevel) {
			logger.Infof("transport: http2Server.operateHeaders parsed a :method field: %v which should be POST", httpMethod)
		}
		t.controlBuf.put(&cleanupStream{
			streamID: streamID,
			rst:      true,
			rstCode:  http2.ErrCodeProtocol,
			onWrite:  func() {},
		})
		s.cancel()
		return false
	}
	if t.inTapHandle != nil {
		var err error
		if s.ctx, err = t.inTapHandle(s.ctx, &tap.Info{FullMethodName: s.method}); err != nil {
			t.mu.Unlock()
			if logger.V(logLevel) {
				logger.Infof("transport: http2Server.operateHeaders got an error from InTapHandle: %v", err)
			}
			stat, ok := status.FromError(err)
			if !ok {
				stat = status.New(codes.PermissionDenied, err.Error())
			}
			t.controlBuf.put(&earlyAbortStream{
				httpStatus:     200,
				streamID:       s.id,
				contentSubtype: s.contentSubtype,
				status:         stat,
			})
			return false
		}
	}
	t.activeStreams[streamID] = s
	if len(t.activeStreams) == 1 {
		t.idle = time.Time{}
	}
	t.mu.Unlock()
	if channelz.IsOn() {
		atomic.AddInt64(&t.czData.streamsStarted, 1)
		atomic.StoreInt64(&t.czData.lastStreamCreatedTime, time.Now().UnixNano())
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
			Header:      metadata.MD(mdata).Copy(),
		}
		t.stats.HandleRPC(s.ctx, inHeader)
	}
	s.ctxDone = s.ctx.Done()
	s.wq = newWriteQuota(defaultWriteQuota, s.ctxDone)
	s.trReader = &transportReader{
		reader: &recvBufferReader{
			ctx:        s.ctx,
			ctxDone:    s.ctxDone,
			recv:       s.buf,
			freeBuffer: t.bufferPool.put,
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
	return false
}

// HandleStreams receives incoming streams using the given handler. This is
// typically run in a separate goroutine.
// traceCtx attaches trace to ctx and returns the new context.
func (t *http2Server) HandleStreams(handle func(*Stream), traceCtx func(context.Context, string) context.Context) {
	defer close(t.readerDone)
	for {
		t.controlBuf.throttle()
		frame, err := t.framer.fr.ReadFrame()
		atomic.StoreInt64(&t.lastRead, time.Now().UnixNano())
		if err != nil {
			if se, ok := err.(http2.StreamError); ok {
				if logger.V(logLevel) {
					logger.Warningf("transport: http2Server.HandleStreams encountered http2.StreamError: %v", se)
				}
				t.mu.Lock()
				s := t.activeStreams[se.StreamID]
				t.mu.Unlock()
				if s != nil {
					t.closeStream(s, true, se.Code, false)
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
			if logger.V(logLevel) {
				logger.Warningf("transport: http2Server.HandleStreams failed to read frame: %v", err)
			}
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
			if logger.V(logLevel) {
				logger.Errorf("transport: http2Server.HandleStreams found unhandled frame type %v.", frame)
			}
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
	if s.getState() == streamReadDone {
		t.closeStream(s, true, http2.ErrCodeStreamClosed, false)
		return
	}
	if size > 0 {
		if err := s.fc.onData(size); err != nil {
			t.closeStream(s, true, http2.ErrCodeFlowControl, false)
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
			buffer := t.bufferPool.get()
			buffer.Reset()
			buffer.Write(f.Data())
			s.write(recvMsg{buffer: buffer})
		}
	}
	if f.StreamEnded() {
		// Received the end of stream from the client.
		s.compareAndSwapState(streamActive, streamReadDone)
		s.write(recvMsg{err: io.EOF})
	}
}

func (t *http2Server) handleRSTStream(f *http2.RSTStreamFrame) {
	// If the stream is not deleted from the transport's active streams map, then do a regular close stream.
	if s, ok := t.getStream(f); ok {
		t.closeStream(s, false, 0, false)
		return
	}
	// If the stream is already deleted from the active streams map, then put a cleanupStream item into controlbuf to delete the stream from loopy writer's established streams map.
	t.controlBuf.put(&cleanupStream{
		streamID: f.Header().StreamID,
		rst:      false,
		rstCode:  0,
		onWrite:  func() {},
	})
}

func (t *http2Server) handleSettings(f *http2.SettingsFrame) {
	if f.IsAck() {
		return
	}
	var ss []http2.Setting
	var updateFuncs []func()
	f.ForeachSetting(func(s http2.Setting) error {
		switch s.ID {
		case http2.SettingMaxHeaderListSize:
			updateFuncs = append(updateFuncs, func() {
				t.maxSendHeaderListSize = new(uint32)
				*t.maxSendHeaderListSize = s.Val
			})
		default:
			ss = append(ss, s)
		}
		return nil
	})
	t.controlBuf.executeAndPut(func(interface{}) bool {
		for _, f := range updateFuncs {
			f()
		}
		return true
	}, &incomingSettings{
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
		if logger.V(logLevel) {
			logger.Errorf("transport: Got too many pings from the client, closing the connection.")
		}
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

func (t *http2Server) checkForHeaderListSize(it interface{}) bool {
	if t.maxSendHeaderListSize == nil {
		return true
	}
	hdrFrame := it.(*headerFrame)
	var sz int64
	for _, f := range hdrFrame.hf {
		if sz += int64(f.Size()); sz > int64(*t.maxSendHeaderListSize) {
			if logger.V(logLevel) {
				logger.Errorf("header list size to send violates the maximum size (%d bytes) set by client", *t.maxSendHeaderListSize)
			}
			return false
		}
	}
	return true
}

// WriteHeader sends the header metadata md back to the client.
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
	if err := t.writeHeaderLocked(s); err != nil {
		s.hdrMu.Unlock()
		return err
	}
	s.hdrMu.Unlock()
	return nil
}

func (t *http2Server) setResetPingStrikes() {
	atomic.StoreUint32(&t.resetPingStrikes, 1)
}

func (t *http2Server) writeHeaderLocked(s *Stream) error {
	// TODO(mmukhi): Benchmark if the performance gets better if count the metadata and other header fields
	// first and create a slice of that exact size.
	headerFields := make([]hpack.HeaderField, 0, 2) // at least :status, content-type will be there if none else.
	headerFields = append(headerFields, hpack.HeaderField{Name: ":status", Value: "200"})
	headerFields = append(headerFields, hpack.HeaderField{Name: "content-type", Value: grpcutil.ContentType(s.contentSubtype)})
	if s.sendCompress != "" {
		headerFields = append(headerFields, hpack.HeaderField{Name: "grpc-encoding", Value: s.sendCompress})
	}
	headerFields = appendHeaderFieldsFromMD(headerFields, s.header)
	success, err := t.controlBuf.executeAndPut(t.checkForHeaderListSize, &headerFrame{
		streamID:  s.id,
		hf:        headerFields,
		endStream: false,
		onWrite:   t.setResetPingStrikes,
	})
	if !success {
		if err != nil {
			return err
		}
		t.closeStream(s, true, http2.ErrCodeInternal, false)
		return ErrHeaderListSizeLimitViolation
	}
	if t.stats != nil {
		// Note: Headers are compressed with hpack after this call returns.
		// No WireLength field is set here.
		outHeader := &stats.OutHeader{
			Header:      s.header.Copy(),
			Compression: s.sendCompress,
		}
		t.stats.HandleRPC(s.Context(), outHeader)
	}
	return nil
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
			if err := t.writeHeaderLocked(s); err != nil {
				s.hdrMu.Unlock()
				return err
			}
		} else { // Send a trailer only response.
			headerFields = append(headerFields, hpack.HeaderField{Name: ":status", Value: "200"})
			headerFields = append(headerFields, hpack.HeaderField{Name: "content-type", Value: grpcutil.ContentType(s.contentSubtype)})
		}
	}
	headerFields = append(headerFields, hpack.HeaderField{Name: "grpc-status", Value: strconv.Itoa(int(st.Code()))})
	headerFields = append(headerFields, hpack.HeaderField{Name: "grpc-message", Value: encodeGrpcMessage(st.Message())})

	if p := st.Proto(); p != nil && len(p.Details) > 0 {
		stBytes, err := proto.Marshal(p)
		if err != nil {
			// TODO: return error instead, when callers are able to handle it.
			logger.Errorf("transport: failed to marshal rpc status: %v, error: %v", p, err)
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
		onWrite:   t.setResetPingStrikes,
	}
	s.hdrMu.Unlock()
	success, err := t.controlBuf.execute(t.checkForHeaderListSize, trailingHeader)
	if !success {
		if err != nil {
			return err
		}
		t.closeStream(s, true, http2.ErrCodeInternal, false)
		return ErrHeaderListSizeLimitViolation
	}
	// Send a RST_STREAM after the trailers if the client has not already half-closed.
	rst := s.getState() == streamActive
	t.finishStream(s, rst, http2.ErrCodeNo, trailingHeader, true)
	if t.stats != nil {
		// Note: The trailer fields are compressed with hpack after this call returns.
		// No WireLength field is set here.
		t.stats.HandleRPC(s.Context(), &stats.OutTrailer{
			Trailer: s.trailer.Copy(),
		})
	}
	return nil
}

// Write converts the data into HTTP2 data frame and sends it out. Non-nil error
// is returns if it fails (e.g., framing error, transport error).
func (t *http2Server) Write(s *Stream, hdr []byte, data []byte, opts *Options) error {
	if !s.isHeaderSent() { // Headers haven't been written yet.
		if err := t.WriteHeader(s, nil); err != nil {
			if _, ok := err.(ConnectionError); ok {
				return err
			}
			// TODO(mmukhi, dfawley): Make sure this is the right code to return.
			return status.Errorf(codes.Internal, "transport: %v", err)
		}
	} else {
		// Writing headers checks for this condition.
		if s.getState() == streamDone {
			// TODO(mmukhi, dfawley): Should the server write also return io.EOF?
			s.cancel()
			select {
			case <-t.done:
				return ErrConnClosing
			default:
			}
			return ContextErr(s.ctx.Err())
		}
	}
	df := &dataFrame{
		streamID:    s.id,
		h:           hdr,
		d:           data,
		onEachWrite: t.setResetPingStrikes,
	}
	if err := s.wq.get(int32(len(hdr) + len(data))); err != nil {
		select {
		case <-t.done:
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
	// True iff a ping has been sent, and no data has been received since then.
	outstandingPing := false
	// Amount of time remaining before which we should receive an ACK for the
	// last sent ping.
	kpTimeoutLeft := time.Duration(0)
	// Records the last value of t.lastRead before we go block on the timer.
	// This is required to check for read activity since then.
	prevNano := time.Now().UnixNano()
	// Initialize the different timers to their default values.
	idleTimer := time.NewTimer(t.kp.MaxConnectionIdle)
	ageTimer := time.NewTimer(t.kp.MaxConnectionAge)
	kpTimer := time.NewTimer(t.kp.Time)
	defer func() {
		// We need to drain the underlying channel in these timers after a call
		// to Stop(), only if we are interested in resetting them. Clearly we
		// are not interested in resetting them here.
		idleTimer.Stop()
		ageTimer.Stop()
		kpTimer.Stop()
	}()

	for {
		select {
		case <-idleTimer.C:
			t.mu.Lock()
			idle := t.idle
			if idle.IsZero() { // The connection is non-idle.
				t.mu.Unlock()
				idleTimer.Reset(t.kp.MaxConnectionIdle)
				continue
			}
			val := t.kp.MaxConnectionIdle - time.Since(idle)
			t.mu.Unlock()
			if val <= 0 {
				// The connection has been idle for a duration of keepalive.MaxConnectionIdle or more.
				// Gracefully close the connection.
				t.Drain()
				return
			}
			idleTimer.Reset(val)
		case <-ageTimer.C:
			t.Drain()
			ageTimer.Reset(t.kp.MaxConnectionAgeGrace)
			select {
			case <-ageTimer.C:
				// Close the connection after grace period.
				if logger.V(logLevel) {
					logger.Infof("transport: closing server transport due to maximum connection age.")
				}
				t.Close()
			case <-t.done:
			}
			return
		case <-kpTimer.C:
			lastRead := atomic.LoadInt64(&t.lastRead)
			if lastRead > prevNano {
				// There has been read activity since the last time we were
				// here. Setup the timer to fire at kp.Time seconds from
				// lastRead time and continue.
				outstandingPing = false
				kpTimer.Reset(time.Duration(lastRead) + t.kp.Time - time.Duration(time.Now().UnixNano()))
				prevNano = lastRead
				continue
			}
			if outstandingPing && kpTimeoutLeft <= 0 {
				if logger.V(logLevel) {
					logger.Infof("transport: closing server transport due to idleness.")
				}
				t.Close()
				return
			}
			if !outstandingPing {
				if channelz.IsOn() {
					atomic.AddInt64(&t.czData.kpCount, 1)
				}
				t.controlBuf.put(p)
				kpTimeoutLeft = t.kp.Timeout
				outstandingPing = true
			}
			// The amount of time to sleep here is the minimum of kp.Time and
			// timeoutLeft. This will ensure that we wait only for kp.Time
			// before sending out the next ping (for cases where the ping is
			// acked).
			sleepDuration := minTime(t.kp.Time, kpTimeoutLeft)
			kpTimeoutLeft -= sleepDuration
			kpTimer.Reset(sleepDuration)
		case <-t.done:
			return
		}
	}
}

// Close starts shutting down the http2Server transport.
// TODO(zhaoq): Now the destruction is not blocked on any pending streams. This
// could cause some resource issue. Revisit this later.
func (t *http2Server) Close() {
	t.mu.Lock()
	if t.state == closing {
		t.mu.Unlock()
		return
	}
	t.state = closing
	streams := t.activeStreams
	t.activeStreams = nil
	t.mu.Unlock()
	t.controlBuf.finish()
	close(t.done)
	if err := t.conn.Close(); err != nil && logger.V(logLevel) {
		logger.Infof("transport: error closing conn during Close: %v", err)
	}
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
}

// deleteStream deletes the stream s from transport's active streams.
func (t *http2Server) deleteStream(s *Stream, eosReceived bool) {
	// In case stream sending and receiving are invoked in separate
	// goroutines (e.g., bi-directional streaming), cancel needs to be
	// called to interrupt the potential blocking on other goroutines.
	s.cancel()

	t.mu.Lock()
	if _, ok := t.activeStreams[s.id]; ok {
		delete(t.activeStreams, s.id)
		if len(t.activeStreams) == 0 {
			t.idle = time.Now()
		}
	}
	t.mu.Unlock()

	if channelz.IsOn() {
		if eosReceived {
			atomic.AddInt64(&t.czData.streamsSucceeded, 1)
		} else {
			atomic.AddInt64(&t.czData.streamsFailed, 1)
		}
	}
}

// finishStream closes the stream and puts the trailing headerFrame into controlbuf.
func (t *http2Server) finishStream(s *Stream, rst bool, rstCode http2.ErrCode, hdr *headerFrame, eosReceived bool) {
	oldState := s.swapState(streamDone)
	if oldState == streamDone {
		// If the stream was already done, return.
		return
	}

	hdr.cleanup = &cleanupStream{
		streamID: s.id,
		rst:      rst,
		rstCode:  rstCode,
		onWrite: func() {
			t.deleteStream(s, eosReceived)
		},
	}
	t.controlBuf.put(hdr)
}

// closeStream clears the footprint of a stream when the stream is not needed any more.
func (t *http2Server) closeStream(s *Stream, rst bool, rstCode http2.ErrCode, eosReceived bool) {
	s.swapState(streamDone)
	t.deleteStream(s, eosReceived)

	t.controlBuf.put(&cleanupStream{
		streamID: s.id,
		rst:      rst,
		rstCode:  rstCode,
		onWrite:  func() {},
	})
}

func (t *http2Server) RemoteAddr() net.Addr {
	return t.remoteAddr
}

func (t *http2Server) Drain() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.drainChan != nil {
		return
	}
	t.drainChan = make(chan struct{})
	t.controlBuf.put(&goAway{code: http2.ErrCodeNo, debugData: []byte{}, headsUp: true})
}

var goAwayPing = &ping{data: [8]byte{1, 6, 1, 8, 0, 3, 3, 9}}

// Handles outgoing GoAway and returns true if loopy needs to put itself
// in draining mode.
func (t *http2Server) outgoingGoAwayHandler(g *goAway) (bool, error) {
	t.maxStreamMu.Lock()
	t.mu.Lock()
	if t.state == closing { // TODO(mmukhi): This seems unnecessary.
		t.mu.Unlock()
		t.maxStreamMu.Unlock()
		// The transport is closing.
		return false, ErrConnClosing
	}
	if !g.headsUp {
		// Stop accepting more streams now.
		t.state = draining
		sid := t.maxStreamID
		if len(t.activeStreams) == 0 {
			g.closeConn = true
		}
		t.mu.Unlock()
		t.maxStreamMu.Unlock()
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
	t.maxStreamMu.Unlock()
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
		case <-t.done:
			return
		}
		t.controlBuf.put(&goAway{code: g.code, debugData: g.debugData})
	}()
	return false, nil
}

func (t *http2Server) ChannelzMetric() *channelz.SocketInternalMetric {
	s := channelz.SocketInternalMetric{
		StreamsStarted:                   atomic.LoadInt64(&t.czData.streamsStarted),
		StreamsSucceeded:                 atomic.LoadInt64(&t.czData.streamsSucceeded),
		StreamsFailed:                    atomic.LoadInt64(&t.czData.streamsFailed),
		MessagesSent:                     atomic.LoadInt64(&t.czData.msgSent),
		MessagesReceived:                 atomic.LoadInt64(&t.czData.msgRecv),
		KeepAlivesSent:                   atomic.LoadInt64(&t.czData.kpCount),
		LastRemoteStreamCreatedTimestamp: time.Unix(0, atomic.LoadInt64(&t.czData.lastStreamCreatedTime)),
		LastMessageSentTimestamp:         time.Unix(0, atomic.LoadInt64(&t.czData.lastMsgSentTime)),
		LastMessageReceivedTimestamp:     time.Unix(0, atomic.LoadInt64(&t.czData.lastMsgRecvTime)),
		LocalFlowControlWindow:           int64(t.fc.getSize()),
		SocketOptions:                    channelz.GetSocketOption(t.conn),
		LocalAddr:                        t.localAddr,
		RemoteAddr:                       t.remoteAddr,
		// RemoteName :
	}
	if au, ok := t.authInfo.(credentials.ChannelzSecurityInfo); ok {
		s.Security = au.GetSecurityValue()
	}
	s.RemoteFlowControlWindow = t.getOutFlowWindow()
	return &s
}

func (t *http2Server) IncrMsgSent() {
	atomic.AddInt64(&t.czData.msgSent, 1)
	atomic.StoreInt64(&t.czData.lastMsgSentTime, time.Now().UnixNano())
}

func (t *http2Server) IncrMsgRecv() {
	atomic.AddInt64(&t.czData.msgRecv, 1)
	atomic.StoreInt64(&t.czData.lastMsgRecvTime, time.Now().UnixNano())
}

func (t *http2Server) getOutFlowWindow() int64 {
	resp := make(chan uint32, 1)
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	t.controlBuf.put(&outFlowControlSizeRequest{resp})
	select {
	case sz := <-resp:
		return int64(sz)
	case <-t.done:
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

type connectionKey struct{}

// GetConnection gets the connection from the context.
func GetConnection(ctx context.Context) net.Conn {
	conn, _ := ctx.Value(connectionKey{}).(net.Conn)
	return conn
}

// SetConnection adds the connection to the context to be able to get
// information about the destination ip and port for an incoming RPC. This also
// allows any unary or streaming interceptors to see the connection.
func setConnection(ctx context.Context, conn net.Conn) context.Context {
	return context.WithValue(ctx, connectionKey{}, conn)
}
