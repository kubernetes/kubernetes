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

// Decoder provides decoding of an Event Stream messages.
type Decoder struct {
	r      io.Reader
	logger aws.Logger
}

// NewDecoder initializes and returns a Decoder for decoding event
// stream messages from the reader provided.
func NewDecoder(r io.Reader, opts ...func(*Decoder)) *Decoder {
	d := &Decoder{
		r: r,
	}

	for _, opt := range opts {
		opt(d)
	}

	return d
}

// DecodeWithLogger adds a logger to be used by the decoder when decoding
// stream events.
func DecodeWithLogger(logger aws.Logger) func(*Decoder) {
	return func(d *Decoder) {
		d.logger = logger
	}
}

// Decode attempts to decode a single message from the event stream reader.
// Will return the event stream message, or error if Decode fails to read
// the message from the stream.
func (d *Decoder) Decode(payloadBuf []byte) (m Message, err error) {
	reader := d.r
	if d.logger != nil {
		debugMsgBuf := bytes.NewBuffer(nil)
		reader = io.TeeReader(reader, debugMsgBuf)
		defer func() {
			logMessageDecode(d.logger, debugMsgBuf, m, err)
		}()
	}

	m, err = Decode(reader, payloadBuf)

	return m, err
}

// Decode attempts to decode a single message from the event stream reader.
// Will return the event stream message, or error if Decode fails to read
// the message from the reader.
func Decode(reader io.Reader, payloadBuf []byte) (m Message, err error) {
	crc := crc32.New(crc32IEEETable)
	hashReader := io.TeeReader(reader, crc)

	prelude, err := decodePrelude(hashReader, crc)
	if err != nil {
		return Message{}, err
	}

	if prelude.HeadersLen > 0 {
		lr := io.LimitReader(hashReader, int64(prelude.HeadersLen))
		m.Headers, err = decodeHeaders(lr)
		if err != nil {
			return Message{}, err
		}
	}

	if payloadLen := prelude.PayloadLen(); payloadLen > 0 {
		buf, err := decodePayload(payloadBuf, io.LimitReader(hashReader, int64(payloadLen)))
		if err != nil {
			return Message{}, err
		}
		m.Payload = buf
	}

	msgCRC := crc.Sum32()
	if err := validateCRC(reader, msgCRC); err != nil {
		return Message{}, err
	}

	return m, nil
}

func logMessageDecode(logger aws.Logger, msgBuf *bytes.Buffer, msg Message, decodeErr error) {
	w := bytes.NewBuffer(nil)
	defer func() { logger.Log(w.String()) }()

	fmt.Fprintf(w, "Raw message:\n%s\n",
		hex.Dump(msgBuf.Bytes()))

	if decodeErr != nil {
		fmt.Fprintf(w, "Decode error: %v\n", decodeErr)
		return
	}

	rawMsg, err := msg.rawMessage()
	if err != nil {
		fmt.Fprintf(w, "failed to create raw message, %v\n", err)
		return
	}

	decodedMsg := decodedMessage{
		rawMessage: rawMsg,
		Headers:    decodedHeaders(msg.Headers),
	}

	fmt.Fprintf(w, "Decoded message:\n")
	encoder := json.NewEncoder(w)
	if err := encoder.Encode(decodedMsg); err != nil {
		fmt.Fprintf(w, "failed to generate decoded message, %v\n", err)
	}
}

func decodePrelude(r io.Reader, crc hash.Hash32) (messagePrelude, error) {
	var p messagePrelude

	var err error
	p.Length, err = decodeUint32(r)
	if err != nil {
		return messagePrelude{}, err
	}

	p.HeadersLen, err = decodeUint32(r)
	if err != nil {
		return messagePrelude{}, err
	}

	if err := p.ValidateLens(); err != nil {
		return messagePrelude{}, err
	}

	preludeCRC := crc.Sum32()
	if err := validateCRC(r, preludeCRC); err != nil {
		return messagePrelude{}, err
	}

	p.PreludeCRC = preludeCRC

	return p, nil
}

func decodePayload(buf []byte, r io.Reader) ([]byte, error) {
	w := bytes.NewBuffer(buf[0:0])

	_, err := io.Copy(w, r)
	return w.Bytes(), err
}

func decodeUint8(r io.Reader) (uint8, error) {
	type byteReader interface {
		ReadByte() (byte, error)
	}

	if br, ok := r.(byteReader); ok {
		v, err := br.ReadByte()
		return uint8(v), err
	}

	var b [1]byte
	_, err := io.ReadFull(r, b[:])
	return uint8(b[0]), err
}
func decodeUint16(r io.Reader) (uint16, error) {
	var b [2]byte
	bs := b[:]
	_, err := io.ReadFull(r, bs)
	if err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint16(bs), nil
}
func decodeUint32(r io.Reader) (uint32, error) {
	var b [4]byte
	bs := b[:]
	_, err := io.ReadFull(r, bs)
	if err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint32(bs), nil
}
func decodeUint64(r io.Reader) (uint64, error) {
	var b [8]byte
	bs := b[:]
	_, err := io.ReadFull(r, bs)
	if err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint64(bs), nil
}

func validateCRC(r io.Reader, expect uint32) error {
	msgCRC, err := decodeUint32(r)
	if err != nil {
		return err
	}

	if msgCRC != expect {
		return ChecksumError{}
	}

	return nil
}
