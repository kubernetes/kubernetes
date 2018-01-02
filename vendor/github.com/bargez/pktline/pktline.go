// Package pktline implements pkt-line format encoding used in Git.
// The format is described in
// https://github.com/git/git/blob/master/Documentation/technical/protocol-common.txt
package pktline

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"strconv"
)

// Errors returned by methods in package pktline.
var (
	ErrShortRead = errors.New("input is too short")
	ErrInputExcess = errors.New("input is too long")
	ErrTooLong = errors.New("too long payload")
	ErrInvalidLen = errors.New("invalid length")
)

const (
	headLen = 4
	maxLen  = 65524 // 65520 bytes of data
)

// Decoder decodes input in pkt-line format.
type Decoder struct {
	r io.Reader
}

// NewDecoder constructs a new pkt-line decoder.
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{r: r}
}

// Decode reads a single pkt-line and stores its payload.
// Flush-pkt is causes *payload to be nil.
func (d *Decoder) Decode(payload *[]byte) error {
	head := make([]byte, headLen)
	_, err := io.ReadFull(d.r, head)
	if err == io.ErrUnexpectedEOF {
		return ErrShortRead
	}
	if err != nil {
		return err
	}
	lineLen, err := strconv.ParseInt(string(head), 16, 16)
	if err != nil {
		return err
	}
	if lineLen == 0 { // flush-pkt
		*payload = nil
		return nil
	}
	if lineLen < headLen {
		return ErrInvalidLen
	}
	*payload = make([]byte, lineLen-headLen)
	if lineLen == headLen { // empty line
		return nil
	}
	_, err = io.ReadFull(d.r, *payload)
	if err == io.ErrUnexpectedEOF {
		return ErrShortRead
	}
	if err != nil {
		return err
	}
	return nil
}

// Decode parses a single pkt-line and returns it's payload.
// Input longer than a single pkt-line is considered an error.
func Decode(line []byte) (payload []byte, err error) {
	buffer := bytes.NewBuffer(line)
	decoder := NewDecoder(buffer)
	err = decoder.Decode(&payload)
	if err != nil {
		return
	}
	if buffer.Len() != 0 {
		err = ErrInputExcess
		return
	}
	return
}

// DecodeUntilFlush decodes pkt-line messages until it encounters flush-pkt.
// The flush-pkt is not included in output.
// If error is not nil, output contains data that was read before the error occured.
func (d *Decoder) DecodeUntilFlush(lines *[][]byte) (err error) {
	*lines = make([][]byte, 0)
	for {
		var l []byte
		err = d.Decode(&l)
		if err != nil {
			return
		}
		if l == nil { // flush-pkt
			return
		}
		*lines = append(*lines, l)
	}
	return
}

// Encoder encodes payloads in pkt-line format.
type Encoder struct {
	w io.Writer
}

// NewEncoder constructs a pkt-line encoder.
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{w: w}
}

// Encode encodes payload and writes it to encoder output.
// If payload is nil, writes flush-pkt.
func (e *Encoder) Encode(payload []byte) (error) {
	line, err := Encode(payload)
	if err != nil {
		return err
	}
	_, err = e.w.Write(line)
	if err != nil {
		return err
	}
	return nil
}

// Encode returns payload encoded in pkt-line format.
func Encode(payload []byte) ([]byte, error) {
	if payload == nil {
		return []byte("0000"), nil
	}
	if len(payload)+headLen > maxLen {
		return nil, ErrTooLong
	}
	head := []byte(fmt.Sprintf("%04x", len(payload)+headLen))
	return append(head, payload...), nil
}

// EncoderDecoder serves as both Encoder and Decoder.
type EncoderDecoder struct {
	Encoder
	Decoder
}

// NewEncoderDecoder constructs pkt-line encoder/decoder.
func NewEncoderDecoder(rw io.ReadWriter) *EncoderDecoder {
	return &EncoderDecoder{*NewEncoder(rw), *NewDecoder(rw)}
}
