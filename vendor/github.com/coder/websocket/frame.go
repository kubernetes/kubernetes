//go:build !js

package websocket

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/coder/websocket/internal/errd"
)

// opcode represents a WebSocket opcode.
type opcode int

// https://tools.ietf.org/html/rfc6455#section-11.8.
const (
	opContinuation opcode = iota
	opText
	opBinary
	// 3 - 7 are reserved for further non-control frames.
	_
	_
	_
	_
	_
	opClose
	opPing
	opPong
	// 11-16 are reserved for further control frames.
)

// header represents a WebSocket frame header.
// See https://tools.ietf.org/html/rfc6455#section-5.2.
type header struct {
	fin    bool
	rsv1   bool
	rsv2   bool
	rsv3   bool
	opcode opcode

	payloadLength int64

	masked  bool
	maskKey uint32
}

// readFrameHeader reads a header from the reader.
// See https://tools.ietf.org/html/rfc6455#section-5.2.
func readFrameHeader(r *bufio.Reader, readBuf []byte) (h header, err error) {
	defer errd.Wrap(&err, "failed to read frame header")

	b, err := r.ReadByte()
	if err != nil {
		return header{}, err
	}

	h.fin = b&(1<<7) != 0
	h.rsv1 = b&(1<<6) != 0
	h.rsv2 = b&(1<<5) != 0
	h.rsv3 = b&(1<<4) != 0

	h.opcode = opcode(b & 0xf)

	b, err = r.ReadByte()
	if err != nil {
		return header{}, err
	}

	h.masked = b&(1<<7) != 0

	payloadLength := b &^ (1 << 7)
	switch {
	case payloadLength < 126:
		h.payloadLength = int64(payloadLength)
	case payloadLength == 126:
		_, err = io.ReadFull(r, readBuf[:2])
		h.payloadLength = int64(binary.BigEndian.Uint16(readBuf))
	case payloadLength == 127:
		_, err = io.ReadFull(r, readBuf)
		h.payloadLength = int64(binary.BigEndian.Uint64(readBuf))
	}
	if err != nil {
		return header{}, err
	}

	if h.payloadLength < 0 {
		return header{}, fmt.Errorf("received negative payload length: %v", h.payloadLength)
	}

	if h.masked {
		_, err = io.ReadFull(r, readBuf[:4])
		if err != nil {
			return header{}, err
		}
		h.maskKey = binary.LittleEndian.Uint32(readBuf)
	}

	return h, nil
}

// maxControlPayload is the maximum length of a control frame payload.
// See https://tools.ietf.org/html/rfc6455#section-5.5.
const maxControlPayload = 125

// writeFrameHeader writes the bytes of the header to w.
// See https://tools.ietf.org/html/rfc6455#section-5.2
func writeFrameHeader(h header, w *bufio.Writer, buf []byte) (err error) {
	defer errd.Wrap(&err, "failed to write frame header")

	var b byte
	if h.fin {
		b |= 1 << 7
	}
	if h.rsv1 {
		b |= 1 << 6
	}
	if h.rsv2 {
		b |= 1 << 5
	}
	if h.rsv3 {
		b |= 1 << 4
	}

	b |= byte(h.opcode)

	err = w.WriteByte(b)
	if err != nil {
		return err
	}

	lengthByte := byte(0)
	if h.masked {
		lengthByte |= 1 << 7
	}

	switch {
	case h.payloadLength > math.MaxUint16:
		lengthByte |= 127
	case h.payloadLength > 125:
		lengthByte |= 126
	case h.payloadLength >= 0:
		lengthByte |= byte(h.payloadLength)
	}
	err = w.WriteByte(lengthByte)
	if err != nil {
		return err
	}

	switch {
	case h.payloadLength > math.MaxUint16:
		binary.BigEndian.PutUint64(buf, uint64(h.payloadLength))
		_, err = w.Write(buf)
	case h.payloadLength > 125:
		binary.BigEndian.PutUint16(buf, uint16(h.payloadLength))
		_, err = w.Write(buf[:2])
	}
	if err != nil {
		return err
	}

	if h.masked {
		binary.LittleEndian.PutUint32(buf, h.maskKey)
		_, err = w.Write(buf[:4])
		if err != nil {
			return err
		}
	}

	return nil
}
