package logging

import "github.com/lucas-clemente/quic-go/internal/wire"

// A Frame is a QUIC frame
type Frame interface{}

// The AckRange is used within the AckFrame.
// It is a range of packet numbers that is being acknowledged.
type AckRange = wire.AckRange

type (
	// An AckFrame is an ACK frame.
	AckFrame = wire.AckFrame
	// A ConnectionCloseFrame is a CONNECTION_CLOSE frame.
	ConnectionCloseFrame = wire.ConnectionCloseFrame
	// A DataBlockedFrame is a DATA_BLOCKED frame.
	DataBlockedFrame = wire.DataBlockedFrame
	// A HandshakeDoneFrame is a HANDSHAKE_DONE frame.
	HandshakeDoneFrame = wire.HandshakeDoneFrame
	// A MaxDataFrame is a MAX_DATA frame.
	MaxDataFrame = wire.MaxDataFrame
	// A MaxStreamDataFrame is a MAX_STREAM_DATA frame.
	MaxStreamDataFrame = wire.MaxStreamDataFrame
	// A MaxStreamsFrame is a MAX_STREAMS_FRAME.
	MaxStreamsFrame = wire.MaxStreamsFrame
	// A NewConnectionIDFrame is a NEW_CONNECTION_ID frame.
	NewConnectionIDFrame = wire.NewConnectionIDFrame
	// A NewTokenFrame is a NEW_TOKEN frame.
	NewTokenFrame = wire.NewTokenFrame
	// A PathChallengeFrame is a PATH_CHALLENGE frame.
	PathChallengeFrame = wire.PathChallengeFrame
	// A PathResponseFrame is a PATH_RESPONSE frame.
	PathResponseFrame = wire.PathResponseFrame
	// A PingFrame is a PING frame.
	PingFrame = wire.PingFrame
	// A ResetStreamFrame is a RESET_STREAM frame.
	ResetStreamFrame = wire.ResetStreamFrame
	// A RetireConnectionIDFrame is a RETIRE_CONNECTION_ID frame.
	RetireConnectionIDFrame = wire.RetireConnectionIDFrame
	// A StopSendingFrame is a STOP_SENDING frame.
	StopSendingFrame = wire.StopSendingFrame
	// A StreamsBlockedFrame is a STREAMS_BLOCKED frame.
	StreamsBlockedFrame = wire.StreamsBlockedFrame
	// A StreamDataBlockedFrame is a STREAM_DATA_BLOCKED frame.
	StreamDataBlockedFrame = wire.StreamDataBlockedFrame
)

// A CryptoFrame is a CRYPTO frame.
type CryptoFrame struct {
	Offset ByteCount
	Length ByteCount
}

// A StreamFrame is a STREAM frame.
type StreamFrame struct {
	StreamID StreamID
	Offset   ByteCount
	Length   ByteCount
	Fin      bool
}

// A DatagramFrame is a DATAGRAM frame.
type DatagramFrame struct {
	Length ByteCount
}
