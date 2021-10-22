package eventstream

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"hash/crc32"
	"io"

	"github.com/aws/aws-sdk-go/aws"
)

// Encoder provides EventStream message encoding.
type Encoder struct {
	w      io.Writer
	logger aws.Logger

	headersBuf *bytes.Buffer
}

// NewEncoder initializes and returns an Encoder to encode Event Stream
// messages to an io.Writer.
func NewEncoder(w io.Writer, opts ...func(*Encoder)) *Encoder {
	e := &Encoder{
		w:          w,
		headersBuf: bytes.NewBuffer(nil),
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// EncodeWithLogger adds a logger to be used by the encode when decoding
// stream events.
func EncodeWithLogger(logger aws.Logger) func(*Encoder) {
	return func(d *Encoder) {
		d.logger = logger
	}
}

// Encode encodes a single EventStream message to the io.Writer the Encoder
// was created with. An error is returned if writing the message fails.
func (e *Encoder) Encode(msg Message) (err error) {
	e.headersBuf.Reset()

	writer := e.w
	if e.logger != nil {
		encodeMsgBuf := bytes.NewBuffer(nil)
		writer = io.MultiWriter(writer, encodeMsgBuf)
		defer func() {
			logMessageEncode(e.logger, encodeMsgBuf, msg, err)
		}()
	}

	if err = EncodeHeaders(e.headersBuf, msg.Headers); err != nil {
		return err
	}

	crc := crc32.New(crc32IEEETable)
	hashWriter := io.MultiWriter(writer, crc)

	headersLen := uint32(e.headersBuf.Len())
	payloadLen := uint32(len(msg.Payload))

	if err = encodePrelude(hashWriter, crc, headersLen, payloadLen); err != nil {
		return err
	}

	if headersLen > 0 {
		if _, err = io.Copy(hashWriter, e.headersBuf); err != nil {
			return err
		}
	}

	if payloadLen > 0 {
		if _, err = hashWriter.Write(msg.Payload); err != nil {
			return err
		}
	}

	msgCRC := crc.Sum32()
	return binary.Write(writer, binary.BigEndian, msgCRC)
}

func logMessageEncode(logger aws.Logger, msgBuf *bytes.Buffer, msg Message, encodeErr error) {
	w := bytes.NewBuffer(nil)
	defer func() { logger.Log(w.String()) }()

	fmt.Fprintf(w, "Message to encode:\n")
	encoder := json.NewEncoder(w)
	if err := encoder.Encode(msg); err != nil {
		fmt.Fprintf(w, "Failed to get encoded message, %v\n", err)
	}

	if encodeErr != nil {
		fmt.Fprintf(w, "Encode error: %v\n", encodeErr)
		return
	}

	fmt.Fprintf(w, "Raw message:\n%s\n", hex.Dump(msgBuf.Bytes()))
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

// EncodeHeaders writes the header values to the writer encoded in the event
// stream format. Returns an error if a header fails to encode.
func EncodeHeaders(w io.Writer, headers Headers) error {
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
