package quic

import (
	"net"
	"os"
	"sync"
	"time"

	"github.com/lucas-clemente/quic-go/internal/ackhandler"
	"github.com/lucas-clemente/quic-go/internal/flowcontrol"
	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/wire"
)

type deadlineError struct{}

func (deadlineError) Error() string   { return "deadline exceeded" }
func (deadlineError) Temporary() bool { return true }
func (deadlineError) Timeout() bool   { return true }
func (deadlineError) Unwrap() error   { return os.ErrDeadlineExceeded }

var errDeadline net.Error = &deadlineError{}

// The streamSender is notified by the stream about various events.
type streamSender interface {
	queueControlFrame(wire.Frame)
	onHasStreamData(protocol.StreamID)
	// must be called without holding the mutex that is acquired by closeForShutdown
	onStreamCompleted(protocol.StreamID)
}

// Each of the both stream halves gets its own uniStreamSender.
// This is necessary in order to keep track when both halves have been completed.
type uniStreamSender struct {
	streamSender
	onStreamCompletedImpl func()
}

func (s *uniStreamSender) queueControlFrame(f wire.Frame) {
	s.streamSender.queueControlFrame(f)
}

func (s *uniStreamSender) onHasStreamData(id protocol.StreamID) {
	s.streamSender.onHasStreamData(id)
}

func (s *uniStreamSender) onStreamCompleted(protocol.StreamID) {
	s.onStreamCompletedImpl()
}

var _ streamSender = &uniStreamSender{}

type streamI interface {
	Stream
	closeForShutdown(error)
	// for receiving
	handleStreamFrame(*wire.StreamFrame) error
	handleResetStreamFrame(*wire.ResetStreamFrame) error
	getWindowUpdate() protocol.ByteCount
	// for sending
	hasData() bool
	handleStopSendingFrame(*wire.StopSendingFrame)
	popStreamFrame(maxBytes protocol.ByteCount) (*ackhandler.Frame, bool)
	updateSendWindow(protocol.ByteCount)
}

var (
	_ receiveStreamI = (streamI)(nil)
	_ sendStreamI    = (streamI)(nil)
)

// A Stream assembles the data from StreamFrames and provides a super-convenient Read-Interface
//
// Read() and Write() may be called concurrently, but multiple calls to Read() or Write() individually must be synchronized manually.
type stream struct {
	receiveStream
	sendStream

	completedMutex         sync.Mutex
	sender                 streamSender
	receiveStreamCompleted bool
	sendStreamCompleted    bool

	version protocol.VersionNumber
}

var _ Stream = &stream{}

// newStream creates a new Stream
func newStream(streamID protocol.StreamID,
	sender streamSender,
	flowController flowcontrol.StreamFlowController,
	version protocol.VersionNumber,
) *stream {
	s := &stream{sender: sender, version: version}
	senderForSendStream := &uniStreamSender{
		streamSender: sender,
		onStreamCompletedImpl: func() {
			s.completedMutex.Lock()
			s.sendStreamCompleted = true
			s.checkIfCompleted()
			s.completedMutex.Unlock()
		},
	}
	s.sendStream = *newSendStream(streamID, senderForSendStream, flowController, version)
	senderForReceiveStream := &uniStreamSender{
		streamSender: sender,
		onStreamCompletedImpl: func() {
			s.completedMutex.Lock()
			s.receiveStreamCompleted = true
			s.checkIfCompleted()
			s.completedMutex.Unlock()
		},
	}
	s.receiveStream = *newReceiveStream(streamID, senderForReceiveStream, flowController, version)
	return s
}

// need to define StreamID() here, since both receiveStream and readStream have a StreamID()
func (s *stream) StreamID() protocol.StreamID {
	// the result is same for receiveStream and sendStream
	return s.sendStream.StreamID()
}

func (s *stream) Close() error {
	return s.sendStream.Close()
}

func (s *stream) SetDeadline(t time.Time) error {
	_ = s.SetReadDeadline(t)  // SetReadDeadline never errors
	_ = s.SetWriteDeadline(t) // SetWriteDeadline never errors
	return nil
}

// CloseForShutdown closes a stream abruptly.
// It makes Read and Write unblock (and return the error) immediately.
// The peer will NOT be informed about this: the stream is closed without sending a FIN or RST.
func (s *stream) closeForShutdown(err error) {
	s.sendStream.closeForShutdown(err)
	s.receiveStream.closeForShutdown(err)
}

// checkIfCompleted is called from the uniStreamSender, when one of the stream halves is completed.
// It makes sure that the onStreamCompleted callback is only called if both receive and send side have completed.
func (s *stream) checkIfCompleted() {
	if s.sendStreamCompleted && s.receiveStreamCompleted {
		s.sender.onStreamCompleted(s.StreamID())
	}
}
