package spdystream

import (
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/docker/spdystream/spdy"
)

var (
	ErrInvalidStreamId   = errors.New("Invalid stream id")
	ErrTimeout           = errors.New("Timeout occured")
	ErrReset             = errors.New("Stream reset")
	ErrWriteClosedStream = errors.New("Write on closed stream")
)

const (
	FRAME_WORKERS = 5
	QUEUE_SIZE    = 50
)

type StreamHandler func(stream *Stream)

type AuthHandler func(header http.Header, slot uint8, parent uint32) bool

type idleAwareFramer struct {
	f              *spdy.Framer
	conn           *Connection
	writeLock      sync.Mutex
	resetChan      chan struct{}
	setTimeoutChan chan time.Duration
	timeout        time.Duration
}

func newIdleAwareFramer(framer *spdy.Framer) *idleAwareFramer {
	iaf := &idleAwareFramer{
		f:              framer,
		resetChan:      make(chan struct{}, 2),
		setTimeoutChan: make(chan time.Duration),
	}
	return iaf
}

func (i *idleAwareFramer) monitor() {
	var (
		timer     *time.Timer
		expired   <-chan time.Time
		resetChan = i.resetChan
	)
Loop:
	for {
		select {
		case timeout := <-i.setTimeoutChan:
			i.timeout = timeout
			if timeout == 0 {
				if timer != nil {
					timer.Stop()
				}
			} else {
				if timer == nil {
					timer = time.NewTimer(timeout)
					expired = timer.C
				} else {
					timer.Reset(timeout)
				}
			}
		case <-resetChan:
			if timer != nil && i.timeout > 0 {
				timer.Reset(i.timeout)
			}
		case <-expired:
			i.conn.streamCond.L.Lock()
			streams := i.conn.streams
			i.conn.streams = make(map[spdy.StreamId]*Stream)
			i.conn.streamCond.Broadcast()
			i.conn.streamCond.L.Unlock()
			go func() {
				for _, stream := range streams {
					stream.resetStream()
				}
				i.conn.Close()
			}()
		case <-i.conn.closeChan:
			if timer != nil {
				timer.Stop()
			}

			// Start a goroutine to drain resetChan. This is needed because we've seen
			// some unit tests with large numbers of goroutines get into a situation
			// where resetChan fills up, at least 1 call to Write() is still trying to
			// send to resetChan, the connection gets closed, and this case statement
			// attempts to grab the write lock that Write() already has, causing a
			// deadlock.
			//
			// See https://github.com/docker/spdystream/issues/49 for more details.
			go func() {
				for _ = range resetChan {
				}
			}()

			i.writeLock.Lock()
			close(resetChan)
			i.resetChan = nil
			i.writeLock.Unlock()

			break Loop
		}
	}

	// Drain resetChan
	for _ = range resetChan {
	}
}

func (i *idleAwareFramer) WriteFrame(frame spdy.Frame) error {
	i.writeLock.Lock()
	defer i.writeLock.Unlock()
	if i.resetChan == nil {
		return io.EOF
	}
	err := i.f.WriteFrame(frame)
	if err != nil {
		return err
	}

	i.resetChan <- struct{}{}

	return nil
}

func (i *idleAwareFramer) ReadFrame() (spdy.Frame, error) {
	frame, err := i.f.ReadFrame()
	if err != nil {
		return nil, err
	}

	// resetChan should never be closed since it is only closed
	// when the connection has closed its closeChan. This closure
	// only occurs after all Reads have finished
	// TODO (dmcgowan): refactor relationship into connection
	i.resetChan <- struct{}{}

	return frame, nil
}

type Connection struct {
	conn   net.Conn
	framer *idleAwareFramer

	closeChan      chan bool
	goneAway       bool
	lastStreamChan chan<- *Stream
	goAwayTimeout  time.Duration
	closeTimeout   time.Duration

	streamLock *sync.RWMutex
	streamCond *sync.Cond
	streams    map[spdy.StreamId]*Stream

	nextIdLock       sync.Mutex
	receiveIdLock    sync.Mutex
	nextStreamId     spdy.StreamId
	receivedStreamId spdy.StreamId

	pingIdLock sync.Mutex
	pingId     uint32
	pingChans  map[uint32]chan error

	shutdownLock sync.Mutex
	shutdownChan chan error
	hasShutdown  bool
}

// NewConnection creates a new spdy connection from an existing
// network connection.
func NewConnection(conn net.Conn, server bool) (*Connection, error) {
	framer, framerErr := spdy.NewFramer(conn, conn)
	if framerErr != nil {
		return nil, framerErr
	}
	idleAwareFramer := newIdleAwareFramer(framer)
	var sid spdy.StreamId
	var rid spdy.StreamId
	var pid uint32
	if server {
		sid = 2
		rid = 1
		pid = 2
	} else {
		sid = 1
		rid = 2
		pid = 1
	}

	streamLock := new(sync.RWMutex)
	streamCond := sync.NewCond(streamLock)

	session := &Connection{
		conn:   conn,
		framer: idleAwareFramer,

		closeChan:     make(chan bool),
		goAwayTimeout: time.Duration(0),
		closeTimeout:  time.Duration(0),

		streamLock:       streamLock,
		streamCond:       streamCond,
		streams:          make(map[spdy.StreamId]*Stream),
		nextStreamId:     sid,
		receivedStreamId: rid,

		pingId:    pid,
		pingChans: make(map[uint32]chan error),

		shutdownChan: make(chan error),
	}
	idleAwareFramer.conn = session
	go idleAwareFramer.monitor()

	return session, nil
}

// Ping sends a ping frame across the connection and
// returns the response time
func (s *Connection) Ping() (time.Duration, error) {
	pid := s.pingId
	s.pingIdLock.Lock()
	if s.pingId > 0x7ffffffe {
		s.pingId = s.pingId - 0x7ffffffe
	} else {
		s.pingId = s.pingId + 2
	}
	s.pingIdLock.Unlock()
	pingChan := make(chan error)
	s.pingChans[pid] = pingChan
	defer delete(s.pingChans, pid)

	frame := &spdy.PingFrame{Id: pid}
	startTime := time.Now()
	writeErr := s.framer.WriteFrame(frame)
	if writeErr != nil {
		return time.Duration(0), writeErr
	}
	select {
	case <-s.closeChan:
		return time.Duration(0), errors.New("connection closed")
	case err, ok := <-pingChan:
		if ok && err != nil {
			return time.Duration(0), err
		}
		break
	}
	return time.Now().Sub(startTime), nil
}

// Serve handles frames sent from the server, including reply frames
// which are needed to fully initiate connections.  Both clients and servers
// should call Serve in a separate goroutine before creating streams.
func (s *Connection) Serve(newHandler StreamHandler) {
	// Parition queues to ensure stream frames are handled
	// by the same worker, ensuring order is maintained
	frameQueues := make([]*PriorityFrameQueue, FRAME_WORKERS)
	for i := 0; i < FRAME_WORKERS; i++ {
		frameQueues[i] = NewPriorityFrameQueue(QUEUE_SIZE)
		// Ensure frame queue is drained when connection is closed
		go func(frameQueue *PriorityFrameQueue) {
			<-s.closeChan
			frameQueue.Drain()
		}(frameQueues[i])

		go s.frameHandler(frameQueues[i], newHandler)
	}

	var partitionRoundRobin int
	for {
		readFrame, err := s.framer.ReadFrame()
		if err != nil {
			if err != io.EOF {
				fmt.Errorf("frame read error: %s", err)
			} else {
				debugMessage("EOF received")
			}
			break
		}
		var priority uint8
		var partition int
		switch frame := readFrame.(type) {
		case *spdy.SynStreamFrame:
			if s.checkStreamFrame(frame) {
				priority = frame.Priority
				partition = int(frame.StreamId % FRAME_WORKERS)
				debugMessage("(%p) Add stream frame: %d ", s, frame.StreamId)
				s.addStreamFrame(frame)
			} else {
				debugMessage("(%p) Rejected stream frame: %d ", s, frame.StreamId)
				continue
			}
		case *spdy.SynReplyFrame:
			priority = s.getStreamPriority(frame.StreamId)
			partition = int(frame.StreamId % FRAME_WORKERS)
		case *spdy.DataFrame:
			priority = s.getStreamPriority(frame.StreamId)
			partition = int(frame.StreamId % FRAME_WORKERS)
		case *spdy.RstStreamFrame:
			priority = s.getStreamPriority(frame.StreamId)
			partition = int(frame.StreamId % FRAME_WORKERS)
		case *spdy.HeadersFrame:
			priority = s.getStreamPriority(frame.StreamId)
			partition = int(frame.StreamId % FRAME_WORKERS)
		case *spdy.PingFrame:
			priority = 0
			partition = partitionRoundRobin
			partitionRoundRobin = (partitionRoundRobin + 1) % FRAME_WORKERS
		case *spdy.GoAwayFrame:
			priority = 0
			partition = partitionRoundRobin
			partitionRoundRobin = (partitionRoundRobin + 1) % FRAME_WORKERS
		default:
			priority = 7
			partition = partitionRoundRobin
			partitionRoundRobin = (partitionRoundRobin + 1) % FRAME_WORKERS
		}
		frameQueues[partition].Push(readFrame, priority)
	}
	close(s.closeChan)

	s.streamCond.L.Lock()
	// notify streams that they're now closed, which will
	// unblock any stream Read() calls
	for _, stream := range s.streams {
		stream.closeRemoteChannels()
	}
	s.streams = make(map[spdy.StreamId]*Stream)
	s.streamCond.Broadcast()
	s.streamCond.L.Unlock()
}

func (s *Connection) frameHandler(frameQueue *PriorityFrameQueue, newHandler StreamHandler) {
	for {
		popFrame := frameQueue.Pop()
		if popFrame == nil {
			return
		}

		var frameErr error
		switch frame := popFrame.(type) {
		case *spdy.SynStreamFrame:
			frameErr = s.handleStreamFrame(frame, newHandler)
		case *spdy.SynReplyFrame:
			frameErr = s.handleReplyFrame(frame)
		case *spdy.DataFrame:
			frameErr = s.handleDataFrame(frame)
		case *spdy.RstStreamFrame:
			frameErr = s.handleResetFrame(frame)
		case *spdy.HeadersFrame:
			frameErr = s.handleHeaderFrame(frame)
		case *spdy.PingFrame:
			frameErr = s.handlePingFrame(frame)
		case *spdy.GoAwayFrame:
			frameErr = s.handleGoAwayFrame(frame)
		default:
			frameErr = fmt.Errorf("unhandled frame type: %T", frame)
		}

		if frameErr != nil {
			fmt.Errorf("frame handling error: %s", frameErr)
		}
	}
}

func (s *Connection) getStreamPriority(streamId spdy.StreamId) uint8 {
	stream, streamOk := s.getStream(streamId)
	if !streamOk {
		return 7
	}
	return stream.priority
}

func (s *Connection) addStreamFrame(frame *spdy.SynStreamFrame) {
	var parent *Stream
	if frame.AssociatedToStreamId != spdy.StreamId(0) {
		parent, _ = s.getStream(frame.AssociatedToStreamId)
	}

	stream := &Stream{
		streamId:   frame.StreamId,
		parent:     parent,
		conn:       s,
		startChan:  make(chan error),
		headers:    frame.Headers,
		finished:   (frame.CFHeader.Flags & spdy.ControlFlagUnidirectional) != 0x00,
		replyCond:  sync.NewCond(new(sync.Mutex)),
		dataChan:   make(chan []byte),
		headerChan: make(chan http.Header),
		closeChan:  make(chan bool),
	}
	if frame.CFHeader.Flags&spdy.ControlFlagFin != 0x00 {
		stream.closeRemoteChannels()
	}

	s.addStream(stream)
}

// checkStreamFrame checks to see if a stream frame is allowed.
// If the stream is invalid, then a reset frame with protocol error
// will be returned.
func (s *Connection) checkStreamFrame(frame *spdy.SynStreamFrame) bool {
	s.receiveIdLock.Lock()
	defer s.receiveIdLock.Unlock()
	if s.goneAway {
		return false
	}
	validationErr := s.validateStreamId(frame.StreamId)
	if validationErr != nil {
		go func() {
			resetErr := s.sendResetFrame(spdy.ProtocolError, frame.StreamId)
			if resetErr != nil {
				fmt.Errorf("reset error: %s", resetErr)
			}
		}()
		return false
	}
	return true
}

func (s *Connection) handleStreamFrame(frame *spdy.SynStreamFrame, newHandler StreamHandler) error {
	stream, ok := s.getStream(frame.StreamId)
	if !ok {
		return fmt.Errorf("Missing stream: %d", frame.StreamId)
	}

	newHandler(stream)

	return nil
}

func (s *Connection) handleReplyFrame(frame *spdy.SynReplyFrame) error {
	debugMessage("(%p) Reply frame received for %d", s, frame.StreamId)
	stream, streamOk := s.getStream(frame.StreamId)
	if !streamOk {
		debugMessage("Reply frame gone away for %d", frame.StreamId)
		// Stream has already gone away
		return nil
	}
	if stream.replied {
		// Stream has already received reply
		return nil
	}
	stream.replied = true

	// TODO Check for error
	if (frame.CFHeader.Flags & spdy.ControlFlagFin) != 0x00 {
		s.remoteStreamFinish(stream)
	}

	close(stream.startChan)

	return nil
}

func (s *Connection) handleResetFrame(frame *spdy.RstStreamFrame) error {
	stream, streamOk := s.getStream(frame.StreamId)
	if !streamOk {
		// Stream has already been removed
		return nil
	}
	s.removeStream(stream)
	stream.closeRemoteChannels()

	if !stream.replied {
		stream.replied = true
		stream.startChan <- ErrReset
		close(stream.startChan)
	}

	stream.finishLock.Lock()
	stream.finished = true
	stream.finishLock.Unlock()

	return nil
}

func (s *Connection) handleHeaderFrame(frame *spdy.HeadersFrame) error {
	stream, streamOk := s.getStream(frame.StreamId)
	if !streamOk {
		// Stream has already gone away
		return nil
	}
	if !stream.replied {
		// No reply received...Protocol error?
		return nil
	}

	// TODO limit headers while not blocking (use buffered chan or goroutine?)
	select {
	case <-stream.closeChan:
		return nil
	case stream.headerChan <- frame.Headers:
	}

	if (frame.CFHeader.Flags & spdy.ControlFlagFin) != 0x00 {
		s.remoteStreamFinish(stream)
	}

	return nil
}

func (s *Connection) handleDataFrame(frame *spdy.DataFrame) error {
	debugMessage("(%p) Data frame received for %d", s, frame.StreamId)
	stream, streamOk := s.getStream(frame.StreamId)
	if !streamOk {
		debugMessage("Data frame gone away for %d", frame.StreamId)
		// Stream has already gone away
		return nil
	}
	if !stream.replied {
		debugMessage("Data frame not replied %d", frame.StreamId)
		// No reply received...Protocol error?
		return nil
	}

	debugMessage("(%p) (%d) Data frame handling", stream, stream.streamId)
	if len(frame.Data) > 0 {
		stream.dataLock.RLock()
		select {
		case <-stream.closeChan:
			debugMessage("(%p) (%d) Data frame not sent (stream shut down)", stream, stream.streamId)
		case stream.dataChan <- frame.Data:
			debugMessage("(%p) (%d) Data frame sent", stream, stream.streamId)
		}
		stream.dataLock.RUnlock()
	}
	if (frame.Flags & spdy.DataFlagFin) != 0x00 {
		s.remoteStreamFinish(stream)
	}
	return nil
}

func (s *Connection) handlePingFrame(frame *spdy.PingFrame) error {
	if s.pingId&0x01 != frame.Id&0x01 {
		return s.framer.WriteFrame(frame)
	}
	pingChan, pingOk := s.pingChans[frame.Id]
	if pingOk {
		close(pingChan)
	}
	return nil
}

func (s *Connection) handleGoAwayFrame(frame *spdy.GoAwayFrame) error {
	debugMessage("(%p) Go away received", s)
	s.receiveIdLock.Lock()
	if s.goneAway {
		s.receiveIdLock.Unlock()
		return nil
	}
	s.goneAway = true
	s.receiveIdLock.Unlock()

	if s.lastStreamChan != nil {
		stream, _ := s.getStream(frame.LastGoodStreamId)
		go func() {
			s.lastStreamChan <- stream
		}()
	}

	// Do not block frame handler waiting for closure
	go s.shutdown(s.goAwayTimeout)

	return nil
}

func (s *Connection) remoteStreamFinish(stream *Stream) {
	stream.closeRemoteChannels()

	stream.finishLock.Lock()
	if stream.finished {
		// Stream is fully closed, cleanup
		s.removeStream(stream)
	}
	stream.finishLock.Unlock()
}

// CreateStream creates a new spdy stream using the parameters for
// creating the stream frame.  The stream frame will be sent upon
// calling this function, however this function does not wait for
// the reply frame.  If waiting for the reply is desired, use
// the stream Wait or WaitTimeout function on the stream returned
// by this function.
func (s *Connection) CreateStream(headers http.Header, parent *Stream, fin bool) (*Stream, error) {
	// MUST synchronize stream creation (all the way to writing the frame)
	// as stream IDs **MUST** increase monotonically.
	s.nextIdLock.Lock()
	defer s.nextIdLock.Unlock()

	streamId := s.getNextStreamId()
	if streamId == 0 {
		return nil, fmt.Errorf("Unable to get new stream id")
	}

	stream := &Stream{
		streamId:   streamId,
		parent:     parent,
		conn:       s,
		startChan:  make(chan error),
		headers:    headers,
		dataChan:   make(chan []byte),
		headerChan: make(chan http.Header),
		closeChan:  make(chan bool),
	}

	debugMessage("(%p) (%p) Create stream", s, stream)

	s.addStream(stream)

	return stream, s.sendStream(stream, fin)
}

func (s *Connection) shutdown(closeTimeout time.Duration) {
	// TODO Ensure this isn't called multiple times
	s.shutdownLock.Lock()
	if s.hasShutdown {
		s.shutdownLock.Unlock()
		return
	}
	s.hasShutdown = true
	s.shutdownLock.Unlock()

	var timeout <-chan time.Time
	if closeTimeout > time.Duration(0) {
		timeout = time.After(closeTimeout)
	}
	streamsClosed := make(chan bool)

	go func() {
		s.streamCond.L.Lock()
		for len(s.streams) > 0 {
			debugMessage("Streams opened: %d, %#v", len(s.streams), s.streams)
			s.streamCond.Wait()
		}
		s.streamCond.L.Unlock()
		close(streamsClosed)
	}()

	var err error
	select {
	case <-streamsClosed:
		// No active streams, close should be safe
		err = s.conn.Close()
	case <-timeout:
		// Force ungraceful close
		err = s.conn.Close()
		// Wait for cleanup to clear active streams
		<-streamsClosed
	}

	if err != nil {
		duration := 10 * time.Minute
		time.AfterFunc(duration, func() {
			select {
			case err, ok := <-s.shutdownChan:
				if ok {
					fmt.Errorf("Unhandled close error after %s: %s", duration, err)
				}
			default:
			}
		})
		s.shutdownChan <- err
	}
	close(s.shutdownChan)

	return
}

// Closes spdy connection by sending GoAway frame and initiating shutdown
func (s *Connection) Close() error {
	s.receiveIdLock.Lock()
	if s.goneAway {
		s.receiveIdLock.Unlock()
		return nil
	}
	s.goneAway = true
	s.receiveIdLock.Unlock()

	var lastStreamId spdy.StreamId
	if s.receivedStreamId > 2 {
		lastStreamId = s.receivedStreamId - 2
	}

	goAwayFrame := &spdy.GoAwayFrame{
		LastGoodStreamId: lastStreamId,
		Status:           spdy.GoAwayOK,
	}

	err := s.framer.WriteFrame(goAwayFrame)
	if err != nil {
		return err
	}

	go s.shutdown(s.closeTimeout)

	return nil
}

// CloseWait closes the connection and waits for shutdown
// to finish.  Note the underlying network Connection
// is not closed until the end of shutdown.
func (s *Connection) CloseWait() error {
	closeErr := s.Close()
	if closeErr != nil {
		return closeErr
	}
	shutdownErr, ok := <-s.shutdownChan
	if ok {
		return shutdownErr
	}
	return nil
}

// Wait waits for the connection to finish shutdown or for
// the wait timeout duration to expire.  This needs to be
// called either after Close has been called or the GOAWAYFRAME
// has been received.  If the wait timeout is 0, this function
// will block until shutdown finishes.  If wait is never called
// and a shutdown error occurs, that error will be logged as an
// unhandled error.
func (s *Connection) Wait(waitTimeout time.Duration) error {
	var timeout <-chan time.Time
	if waitTimeout > time.Duration(0) {
		timeout = time.After(waitTimeout)
	}

	select {
	case err, ok := <-s.shutdownChan:
		if ok {
			return err
		}
	case <-timeout:
		return ErrTimeout
	}
	return nil
}

// NotifyClose registers a channel to be called when the remote
// peer inidicates connection closure.  The last stream to be
// received by the remote will be sent on the channel.  The notify
// timeout will determine the duration between go away received
// and the connection being closed.
func (s *Connection) NotifyClose(c chan<- *Stream, timeout time.Duration) {
	s.goAwayTimeout = timeout
	s.lastStreamChan = c
}

// SetCloseTimeout sets the amount of time close will wait for
// streams to finish before terminating the underlying network
// connection.  Setting the timeout to 0 will cause close to
// wait forever, which is the default.
func (s *Connection) SetCloseTimeout(timeout time.Duration) {
	s.closeTimeout = timeout
}

// SetIdleTimeout sets the amount of time the connection may sit idle before
// it is forcefully terminated.
func (s *Connection) SetIdleTimeout(timeout time.Duration) {
	s.framer.setTimeoutChan <- timeout
}

func (s *Connection) sendHeaders(headers http.Header, stream *Stream, fin bool) error {
	var flags spdy.ControlFlags
	if fin {
		flags = spdy.ControlFlagFin
	}

	headerFrame := &spdy.HeadersFrame{
		StreamId: stream.streamId,
		Headers:  headers,
		CFHeader: spdy.ControlFrameHeader{Flags: flags},
	}

	return s.framer.WriteFrame(headerFrame)
}

func (s *Connection) sendReply(headers http.Header, stream *Stream, fin bool) error {
	var flags spdy.ControlFlags
	if fin {
		flags = spdy.ControlFlagFin
	}

	replyFrame := &spdy.SynReplyFrame{
		StreamId: stream.streamId,
		Headers:  headers,
		CFHeader: spdy.ControlFrameHeader{Flags: flags},
	}

	return s.framer.WriteFrame(replyFrame)
}

func (s *Connection) sendResetFrame(status spdy.RstStreamStatus, streamId spdy.StreamId) error {
	resetFrame := &spdy.RstStreamFrame{
		StreamId: streamId,
		Status:   status,
	}

	return s.framer.WriteFrame(resetFrame)
}

func (s *Connection) sendReset(status spdy.RstStreamStatus, stream *Stream) error {
	return s.sendResetFrame(status, stream.streamId)
}

func (s *Connection) sendStream(stream *Stream, fin bool) error {
	var flags spdy.ControlFlags
	if fin {
		flags = spdy.ControlFlagFin
		stream.finished = true
	}

	var parentId spdy.StreamId
	if stream.parent != nil {
		parentId = stream.parent.streamId
	}

	streamFrame := &spdy.SynStreamFrame{
		StreamId:             spdy.StreamId(stream.streamId),
		AssociatedToStreamId: spdy.StreamId(parentId),
		Headers:              stream.headers,
		CFHeader:             spdy.ControlFrameHeader{Flags: flags},
	}

	return s.framer.WriteFrame(streamFrame)
}

// getNextStreamId returns the next sequential id
// every call should produce a unique value or an error
func (s *Connection) getNextStreamId() spdy.StreamId {
	sid := s.nextStreamId
	if sid > 0x7fffffff {
		return 0
	}
	s.nextStreamId = s.nextStreamId + 2
	return sid
}

// PeekNextStreamId returns the next sequential id and keeps the next id untouched
func (s *Connection) PeekNextStreamId() spdy.StreamId {
	sid := s.nextStreamId
	return sid
}

func (s *Connection) validateStreamId(rid spdy.StreamId) error {
	if rid > 0x7fffffff || rid < s.receivedStreamId {
		return ErrInvalidStreamId
	}
	s.receivedStreamId = rid + 2
	return nil
}

func (s *Connection) addStream(stream *Stream) {
	s.streamCond.L.Lock()
	s.streams[stream.streamId] = stream
	debugMessage("(%p) (%p) Stream added, broadcasting: %d", s, stream, stream.streamId)
	s.streamCond.Broadcast()
	s.streamCond.L.Unlock()
}

func (s *Connection) removeStream(stream *Stream) {
	s.streamCond.L.Lock()
	delete(s.streams, stream.streamId)
	debugMessage("Stream removed, broadcasting: %d", stream.streamId)
	s.streamCond.Broadcast()
	s.streamCond.L.Unlock()
}

func (s *Connection) getStream(streamId spdy.StreamId) (stream *Stream, ok bool) {
	s.streamLock.RLock()
	stream, ok = s.streams[streamId]
	s.streamLock.RUnlock()
	return
}

// FindStream looks up the given stream id and either waits for the
// stream to be found or returns nil if the stream id is no longer
// valid.
func (s *Connection) FindStream(streamId uint32) *Stream {
	var stream *Stream
	var ok bool
	s.streamCond.L.Lock()
	stream, ok = s.streams[spdy.StreamId(streamId)]
	debugMessage("(%p) Found stream %d? %t", s, spdy.StreamId(streamId), ok)
	for !ok && streamId >= uint32(s.receivedStreamId) {
		s.streamCond.Wait()
		stream, ok = s.streams[spdy.StreamId(streamId)]
	}
	s.streamCond.L.Unlock()
	return stream
}

func (s *Connection) CloseChan() <-chan bool {
	return s.closeChan
}
