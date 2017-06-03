// +build !amd64,!386 appengine

package roaring

import (
	"encoding/binary"
	"io"
)

func (b *arrayContainer) writeTo(stream io.Writer) (int, error) {
	buf := make([]byte, 2*len(b.content))
	for i, v := range b.content {
		base := i * 2
		buf[base] = byte(v)
		buf[base+1] = byte(v >> 8)
	}
	return stream.Write(buf)
}

func (b *arrayContainer) readFrom(stream io.Reader) (int, error) {
	err := binary.Read(stream, binary.LittleEndian, b.content)
	if err != nil {
		return 0, err
	}
	return 2 * len(b.content), nil
}

func (b *bitmapContainer) writeTo(stream io.Writer) (int, error) {
	// Write set
	buf := make([]byte, 8*len(b.bitmap))
	for i, v := range b.bitmap {
		base := i * 8
		buf[base] = byte(v)
		buf[base+1] = byte(v >> 8)
		buf[base+2] = byte(v >> 16)
		buf[base+3] = byte(v >> 24)
		buf[base+4] = byte(v >> 32)
		buf[base+5] = byte(v >> 40)
		buf[base+6] = byte(v >> 48)
		buf[base+7] = byte(v >> 56)
	}
	return stream.Write(buf)
}

func (b *bitmapContainer) readFrom(stream io.Reader) (int, error) {
	err := binary.Read(stream, binary.LittleEndian, b.bitmap)
	if err != nil {
		return 0, err
	}
	return 8 * len(b.bitmap), nil
}

func (bc *bitmapContainer) asLittleEndianByteSlice() []byte {
	by := make([]byte, len(bc.bitmap)*8)
	for i := range bc.bitmap {
		binary.LittleEndian.PutUint64(by[i*8:], bc.bitmap[i])
	}
	return by
}
