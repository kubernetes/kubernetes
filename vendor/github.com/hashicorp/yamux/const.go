package yamux

import (
	"encoding/binary"
	"fmt"
)

var (
	// ErrInvalidVersion means we received a frame with an
	// invalid version
	ErrInvalidVersion = fmt.Errorf("invalid protocol version")

	// ErrInvalidMsgType means we received a frame with an
	// invalid message type
	ErrInvalidMsgType = fmt.Errorf("invalid msg type")

	// ErrSessionShutdown is used if there is a shutdown during
	// an operation
	ErrSessionShutdown = fmt.Errorf("session shutdown")

	// ErrStreamsExhausted is returned if we have no more
	// stream ids to issue
	ErrStreamsExhausted = fmt.Errorf("streams exhausted")

	// ErrDuplicateStream is used if a duplicate stream is
	// opened inbound
	ErrDuplicateStream = fmt.Errorf("duplicate stream initiated")

	// ErrReceiveWindowExceeded indicates the window was exceeded
	ErrRecvWindowExceeded = fmt.Errorf("recv window exceeded")

	// ErrTimeout is used when we reach an IO deadline
	ErrTimeout = fmt.Errorf("i/o deadline reached")

	// ErrStreamClosed is returned when using a closed stream
	ErrStreamClosed = fmt.Errorf("stream closed")

	// ErrUnexpectedFlag is set when we get an unexpected flag
	ErrUnexpectedFlag = fmt.Errorf("unexpected flag")

	// ErrRemoteGoAway is used when we get a go away from the other side
	ErrRemoteGoAway = fmt.Errorf("remote end is not accepting connections")

	// ErrConnectionReset is sent if a stream is reset. This can happen
	// if the backlog is exceeded, or if there was a remote GoAway.
	ErrConnectionReset = fmt.Errorf("connection reset")

	// ErrConnectionWriteTimeout indicates that we hit the "safety valve"
	// timeout writing to the underlying stream connection.
	ErrConnectionWriteTimeout = fmt.Errorf("connection write timeout")

	// ErrKeepAliveTimeout is sent if a missed keepalive caused the stream close
	ErrKeepAliveTimeout = fmt.Errorf("keepalive timeout")
)

const (
	// protoVersion is the only version we support
	protoVersion uint8 = 0
)

const (
	// Data is used for data frames. They are followed
	// by length bytes worth of payload.
	typeData uint8 = iota

	// WindowUpdate is used to change the window of
	// a given stream. The length indicates the delta
	// update to the window.
	typeWindowUpdate

	// Ping is sent as a keep-alive or to measure
	// the RTT. The StreamID and Length value are echoed
	// back in the response.
	typePing

	// GoAway is sent to terminate a session. The StreamID
	// should be 0 and the length is an error code.
	typeGoAway
)

const (
	// SYN is sent to signal a new stream. May
	// be sent with a data payload
	flagSYN uint16 = 1 << iota

	// ACK is sent to acknowledge a new stream. May
	// be sent with a data payload
	flagACK

	// FIN is sent to half-close the given stream.
	// May be sent with a data payload.
	flagFIN

	// RST is used to hard close a given stream.
	flagRST
)

const (
	// initialStreamWindow is the initial stream window size
	initialStreamWindow uint32 = 256 * 1024
)

const (
	// goAwayNormal is sent on a normal termination
	goAwayNormal uint32 = iota

	// goAwayProtoErr sent on a protocol error
	goAwayProtoErr

	// goAwayInternalErr sent on an internal error
	goAwayInternalErr
)

const (
	sizeOfVersion  = 1
	sizeOfType     = 1
	sizeOfFlags    = 2
	sizeOfStreamID = 4
	sizeOfLength   = 4
	headerSize     = sizeOfVersion + sizeOfType + sizeOfFlags +
		sizeOfStreamID + sizeOfLength
)

type header []byte

func (h header) Version() uint8 {
	return h[0]
}

func (h header) MsgType() uint8 {
	return h[1]
}

func (h header) Flags() uint16 {
	return binary.BigEndian.Uint16(h[2:4])
}

func (h header) StreamID() uint32 {
	return binary.BigEndian.Uint32(h[4:8])
}

func (h header) Length() uint32 {
	return binary.BigEndian.Uint32(h[8:12])
}

func (h header) String() string {
	return fmt.Sprintf("Vsn:%d Type:%d Flags:%d StreamID:%d Length:%d",
		h.Version(), h.MsgType(), h.Flags(), h.StreamID(), h.Length())
}

func (h header) encode(msgType uint8, flags uint16, streamID uint32, length uint32) {
	h[0] = protoVersion
	h[1] = msgType
	binary.BigEndian.PutUint16(h[2:4], flags)
	binary.BigEndian.PutUint32(h[4:8], streamID)
	binary.BigEndian.PutUint32(h[8:12], length)
}
