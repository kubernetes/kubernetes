package wire

import (
	"bytes"
	"fmt"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/quicvarint"
)

// A MaxStreamsFrame is a MAX_STREAMS frame
type MaxStreamsFrame struct {
	Type         protocol.StreamType
	MaxStreamNum protocol.StreamNum
}

func parseMaxStreamsFrame(r *bytes.Reader, _ protocol.VersionNumber) (*MaxStreamsFrame, error) {
	typeByte, err := r.ReadByte()
	if err != nil {
		return nil, err
	}

	f := &MaxStreamsFrame{}
	switch typeByte {
	case 0x12:
		f.Type = protocol.StreamTypeBidi
	case 0x13:
		f.Type = protocol.StreamTypeUni
	}
	streamID, err := quicvarint.Read(r)
	if err != nil {
		return nil, err
	}
	f.MaxStreamNum = protocol.StreamNum(streamID)
	if f.MaxStreamNum > protocol.MaxStreamCount {
		return nil, fmt.Errorf("%d exceeds the maximum stream count", f.MaxStreamNum)
	}
	return f, nil
}

func (f *MaxStreamsFrame) Write(b *bytes.Buffer, _ protocol.VersionNumber) error {
	switch f.Type {
	case protocol.StreamTypeBidi:
		b.WriteByte(0x12)
	case protocol.StreamTypeUni:
		b.WriteByte(0x13)
	}
	quicvarint.Write(b, uint64(f.MaxStreamNum))
	return nil
}

// Length of a written frame
func (f *MaxStreamsFrame) Length(protocol.VersionNumber) protocol.ByteCount {
	return 1 + quicvarint.Len(uint64(f.MaxStreamNum))
}
