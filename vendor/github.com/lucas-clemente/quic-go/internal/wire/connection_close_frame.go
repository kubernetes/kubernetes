package wire

import (
	"bytes"
	"io"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/quicvarint"
)

// A ConnectionCloseFrame is a CONNECTION_CLOSE frame
type ConnectionCloseFrame struct {
	IsApplicationError bool
	ErrorCode          uint64
	FrameType          uint64
	ReasonPhrase       string
}

func parseConnectionCloseFrame(r *bytes.Reader, _ protocol.VersionNumber) (*ConnectionCloseFrame, error) {
	typeByte, err := r.ReadByte()
	if err != nil {
		return nil, err
	}

	f := &ConnectionCloseFrame{IsApplicationError: typeByte == 0x1d}
	ec, err := quicvarint.Read(r)
	if err != nil {
		return nil, err
	}
	f.ErrorCode = ec
	// read the Frame Type, if this is not an application error
	if !f.IsApplicationError {
		ft, err := quicvarint.Read(r)
		if err != nil {
			return nil, err
		}
		f.FrameType = ft
	}
	var reasonPhraseLen uint64
	reasonPhraseLen, err = quicvarint.Read(r)
	if err != nil {
		return nil, err
	}
	// shortcut to prevent the unnecessary allocation of dataLen bytes
	// if the dataLen is larger than the remaining length of the packet
	// reading the whole reason phrase would result in EOF when attempting to READ
	if int(reasonPhraseLen) > r.Len() {
		return nil, io.EOF
	}

	reasonPhrase := make([]byte, reasonPhraseLen)
	if _, err := io.ReadFull(r, reasonPhrase); err != nil {
		// this should never happen, since we already checked the reasonPhraseLen earlier
		return nil, err
	}
	f.ReasonPhrase = string(reasonPhrase)
	return f, nil
}

// Length of a written frame
func (f *ConnectionCloseFrame) Length(protocol.VersionNumber) protocol.ByteCount {
	length := 1 + quicvarint.Len(f.ErrorCode) + quicvarint.Len(uint64(len(f.ReasonPhrase))) + protocol.ByteCount(len(f.ReasonPhrase))
	if !f.IsApplicationError {
		length += quicvarint.Len(f.FrameType) // for the frame type
	}
	return length
}

func (f *ConnectionCloseFrame) Write(b *bytes.Buffer, version protocol.VersionNumber) error {
	if f.IsApplicationError {
		b.WriteByte(0x1d)
	} else {
		b.WriteByte(0x1c)
	}

	quicvarint.Write(b, f.ErrorCode)
	if !f.IsApplicationError {
		quicvarint.Write(b, f.FrameType)
	}
	quicvarint.Write(b, uint64(len(f.ReasonPhrase)))
	b.WriteString(f.ReasonPhrase)
	return nil
}
