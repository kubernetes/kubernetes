package binary

import (
	"encoding/binary"
	"io"
)

// Write writes the binary representation of data into w, using BigEndian order
// https://golang.org/pkg/encoding/binary/#Write
func Write(w io.Writer, data ...interface{}) error {
	for _, v := range data {
		if err := binary.Write(w, binary.BigEndian, v); err != nil {
			return err
		}
	}

	return nil
}

func WriteVariableWidthInt(w io.Writer, n int64) error {
	buf := []byte{byte(n & 0x7f)}
	n >>= 7
	for n != 0 {
		n--
		buf = append([]byte{0x80 | (byte(n & 0x7f))}, buf...)
		n >>= 7
	}

	_, err := w.Write(buf)

	return err
}

// WriteUint64 writes the binary representation of a uint64 into w, in BigEndian
// order
func WriteUint64(w io.Writer, value uint64) error {
	return binary.Write(w, binary.BigEndian, value)
}

// WriteUint32 writes the binary representation of a uint32 into w, in BigEndian
// order
func WriteUint32(w io.Writer, value uint32) error {
	return binary.Write(w, binary.BigEndian, value)
}

// WriteUint16 writes the binary representation of a uint16 into w, in BigEndian
// order
func WriteUint16(w io.Writer, value uint16) error {
	return binary.Write(w, binary.BigEndian, value)
}
