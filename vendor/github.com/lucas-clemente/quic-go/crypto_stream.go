package quic

import (
	"fmt"
	"io"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/qerr"
	"github.com/lucas-clemente/quic-go/internal/utils"
	"github.com/lucas-clemente/quic-go/internal/wire"
)

type cryptoStream interface {
	// for receiving data
	HandleCryptoFrame(*wire.CryptoFrame) error
	GetCryptoData() []byte
	Finish() error
	// for sending data
	io.Writer
	HasData() bool
	PopCryptoFrame(protocol.ByteCount) *wire.CryptoFrame
}

type cryptoStreamImpl struct {
	queue  *frameSorter
	msgBuf []byte

	highestOffset protocol.ByteCount
	finished      bool

	writeOffset protocol.ByteCount
	writeBuf    []byte
}

func newCryptoStream() cryptoStream {
	return &cryptoStreamImpl{queue: newFrameSorter()}
}

func (s *cryptoStreamImpl) HandleCryptoFrame(f *wire.CryptoFrame) error {
	highestOffset := f.Offset + protocol.ByteCount(len(f.Data))
	if maxOffset := highestOffset; maxOffset > protocol.MaxCryptoStreamOffset {
		return &qerr.TransportError{
			ErrorCode:    qerr.CryptoBufferExceeded,
			ErrorMessage: fmt.Sprintf("received invalid offset %d on crypto stream, maximum allowed %d", maxOffset, protocol.MaxCryptoStreamOffset),
		}
	}
	if s.finished {
		if highestOffset > s.highestOffset {
			// reject crypto data received after this stream was already finished
			return &qerr.TransportError{
				ErrorCode:    qerr.ProtocolViolation,
				ErrorMessage: "received crypto data after change of encryption level",
			}
		}
		// ignore data with a smaller offset than the highest received
		// could e.g. be a retransmission
		return nil
	}
	s.highestOffset = utils.MaxByteCount(s.highestOffset, highestOffset)
	if err := s.queue.Push(f.Data, f.Offset, nil); err != nil {
		return err
	}
	for {
		_, data, _ := s.queue.Pop()
		if data == nil {
			return nil
		}
		s.msgBuf = append(s.msgBuf, data...)
	}
}

// GetCryptoData retrieves data that was received in CRYPTO frames
func (s *cryptoStreamImpl) GetCryptoData() []byte {
	if len(s.msgBuf) < 4 {
		return nil
	}
	msgLen := 4 + int(s.msgBuf[1])<<16 + int(s.msgBuf[2])<<8 + int(s.msgBuf[3])
	if len(s.msgBuf) < msgLen {
		return nil
	}
	msg := make([]byte, msgLen)
	copy(msg, s.msgBuf[:msgLen])
	s.msgBuf = s.msgBuf[msgLen:]
	return msg
}

func (s *cryptoStreamImpl) Finish() error {
	if s.queue.HasMoreData() {
		return &qerr.TransportError{
			ErrorCode:    qerr.ProtocolViolation,
			ErrorMessage: "encryption level changed, but crypto stream has more data to read",
		}
	}
	s.finished = true
	return nil
}

// Writes writes data that should be sent out in CRYPTO frames
func (s *cryptoStreamImpl) Write(p []byte) (int, error) {
	s.writeBuf = append(s.writeBuf, p...)
	return len(p), nil
}

func (s *cryptoStreamImpl) HasData() bool {
	return len(s.writeBuf) > 0
}

func (s *cryptoStreamImpl) PopCryptoFrame(maxLen protocol.ByteCount) *wire.CryptoFrame {
	f := &wire.CryptoFrame{Offset: s.writeOffset}
	n := utils.MinByteCount(f.MaxDataLen(maxLen), protocol.ByteCount(len(s.writeBuf)))
	f.Data = s.writeBuf[:n]
	s.writeBuf = s.writeBuf[n:]
	s.writeOffset += n
	return f
}
