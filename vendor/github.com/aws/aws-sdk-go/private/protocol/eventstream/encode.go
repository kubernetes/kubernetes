package eventstream

import (
	"bytes"
	"encoding/binary"
	"hash"
	"hash/crc32"
	"io"
)

// Encoder provides EventStream message encoding.
type Encoder struct {
	w io.Writer

	headersBuf *bytes.Buffer
}

// NewEncoder initializes and returns an Encoder to encode Event Stream
// messages to an io.Writer.
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{
		w:          w,
		headersBuf: bytes.NewBuffer(nil),
	}
}

// Encode encodes a single EventStream message to the io.Writer the Encoder
// was created with. An error is returned if writing the message fails.
func (e *Encoder) Encode(msg Message) error {
	e.headersBuf.Reset()

	err := encodeHeaders(e.headersBuf, msg.Headers)
	if err != nil {
		return err
	}

	crc := crc32.New(crc32IEEETable)
	hashWriter := io.MultiWriter(e.w, crc)

	headersLen := uint32(e.headersBuf.Len())
	payloadLen := uint32(len(msg.Payload))

	if err := encodePrelude(hashWriter, crc, headersLen, payloadLen); err != nil {
		return err
	}

	if headersLen > 0 {
		if _, err := io.Copy(hashWriter, e.headersBuf); err != nil {
			return err
		}
	}

	if payloadLen > 0 {
		if _, err := hashWriter.Write(msg.Payload); err != nil {
			return err
		}
	}

	msgCRC := crc.Sum32()
	return binary.Write(e.w, binary.BigEndian, msgCRC)
}

func encodePrelude(w io.Writer, crc hash.Hash32, headersLen, payloadLen uint32) error {
	p := messagePrelude{
		Length:     minMsgLen + headersLen + payloadLen,
		HeadersLen: headersLen,
	}
	if err := p.ValidateLens(); err != nil {
		return err
	}

	err := binaryWriteFields(w, binary.BigEndian,
		p.Length,
		p.HeadersLen,
	)
	if err != nil {
		return err
	}

	p.PreludeCRC = crc.Sum32()
	err = binary.Write(w, binary.BigEndian, p.PreludeCRC)
	if err != nil {
		return err
	}

	return nil
}

func encodeHeaders(w io.Writer, headers Headers) error {
	for _, h := range headers {
		hn := headerName{
			Len: uint8(len(h.Name)),
		}
		copy(hn.Name[:hn.Len], h.Name)
		if err := hn.encode(w); err != nil {
			return err
		}

		if err := h.Value.encode(w); err != nil {
			return err
		}
	}

	return nil
}

func binaryWriteFields(w io.Writer, order binary.ByteOrder, vs ...interface{}) error {
	for _, v := range vs {
		if err := binary.Write(w, order, v); err != nil {
			return err
		}
	}
	return nil
}
