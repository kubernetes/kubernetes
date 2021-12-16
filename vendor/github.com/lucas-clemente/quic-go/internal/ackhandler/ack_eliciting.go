package ackhandler

import "github.com/lucas-clemente/quic-go/internal/wire"

// IsFrameAckEliciting returns true if the frame is ack-eliciting.
func IsFrameAckEliciting(f wire.Frame) bool {
	_, isAck := f.(*wire.AckFrame)
	_, isConnectionClose := f.(*wire.ConnectionCloseFrame)
	return !isAck && !isConnectionClose
}

// HasAckElicitingFrames returns true if at least one frame is ack-eliciting.
func HasAckElicitingFrames(fs []Frame) bool {
	for _, f := range fs {
		if IsFrameAckEliciting(f.Frame) {
			return true
		}
	}
	return false
}
