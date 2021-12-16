package wire

import (
	"bytes"

	"github.com/lucas-clemente/quic-go/internal/protocol"
)

// A PingFrame is a PING frame
type PingFrame struct{}

func parsePingFrame(r *bytes.Reader, _ protocol.VersionNumber) (*PingFrame, error) {
	if _, err := r.ReadByte(); err != nil {
		return nil, err
	}
	return &PingFrame{}, nil
}

func (f *PingFrame) Write(b *bytes.Buffer, version protocol.VersionNumber) error {
	b.WriteByte(0x1)
	return nil
}

// Length of a written frame
func (f *PingFrame) Length(version protocol.VersionNumber) protocol.ByteCount {
	return 1
}
