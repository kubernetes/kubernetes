package wire

import (
	"bytes"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/quicvarint"
)

// A StreamDataBlockedFrame is a STREAM_DATA_BLOCKED frame
type StreamDataBlockedFrame struct {
	StreamID          protocol.StreamID
	MaximumStreamData protocol.ByteCount
}

func parseStreamDataBlockedFrame(r *bytes.Reader, _ protocol.VersionNumber) (*StreamDataBlockedFrame, error) {
	if _, err := r.ReadByte(); err != nil {
		return nil, err
	}

	sid, err := quicvarint.Read(r)
	if err != nil {
		return nil, err
	}
	offset, err := quicvarint.Read(r)
	if err != nil {
		return nil, err
	}

	return &StreamDataBlockedFrame{
		StreamID:          protocol.StreamID(sid),
		MaximumStreamData: protocol.ByteCount(offset),
	}, nil
}

func (f *StreamDataBlockedFrame) Write(b *bytes.Buffer, version protocol.VersionNumber) error {
	b.WriteByte(0x15)
	quicvarint.Write(b, uint64(f.StreamID))
	quicvarint.Write(b, uint64(f.MaximumStreamData))
	return nil
}

// Length of a written frame
func (f *StreamDataBlockedFrame) Length(version protocol.VersionNumber) protocol.ByteCount {
	return 1 + quicvarint.Len(uint64(f.StreamID)) + quicvarint.Len(uint64(f.MaximumStreamData))
}
