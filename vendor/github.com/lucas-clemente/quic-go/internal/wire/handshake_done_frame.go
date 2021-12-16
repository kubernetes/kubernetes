package wire

import (
	"bytes"

	"github.com/lucas-clemente/quic-go/internal/protocol"
)

// A HandshakeDoneFrame is a HANDSHAKE_DONE frame
type HandshakeDoneFrame struct{}

// ParseHandshakeDoneFrame parses a HandshakeDone frame
func parseHandshakeDoneFrame(r *bytes.Reader, _ protocol.VersionNumber) (*HandshakeDoneFrame, error) {
	if _, err := r.ReadByte(); err != nil {
		return nil, err
	}
	return &HandshakeDoneFrame{}, nil
}

func (f *HandshakeDoneFrame) Write(b *bytes.Buffer, _ protocol.VersionNumber) error {
	b.WriteByte(0x1e)
	return nil
}

// Length of a written frame
func (f *HandshakeDoneFrame) Length(_ protocol.VersionNumber) protocol.ByteCount {
	return 1
}
