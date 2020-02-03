// +build !amd64,!386 appengine

package roaring

import (
	"encoding/binary"
	"errors"
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
	if b.cardinality <= arrayDefaultMaxSize {
		return 0, errors.New("refusing to write bitmap container with cardinality of array container")
	}

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
	b.computeCardinality()
	return 8 * len(b.bitmap), nil
}

func (bc *bitmapContainer) asLittleEndianByteSlice() []byte {
	by := make([]byte, len(bc.bitmap)*8)
	for i := range bc.bitmap {
		binary.LittleEndian.PutUint64(by[i*8:], bc.bitmap[i])
	}
	return by
}

func uint64SliceAsByteSlice(slice []uint64) []byte {
	by := make([]byte, len(slice)*8)

	for i, v := range slice {
		binary.LittleEndian.PutUint64(by[i*8:], v)
	}

	return by
}

func uint16SliceAsByteSlice(slice []uint16) []byte {
	by := make([]byte, len(slice)*2)

	for i, v := range slice {
		binary.LittleEndian.PutUint16(by[i*2:], v)
	}

	return by
}

func byteSliceAsUint16Slice(slice []byte) []uint16 {
	if len(slice)%2 != 0 {
		panic("Slice size should be divisible by 2")
	}

	b := make([]uint16, len(slice)/2)

	for i := range b {
		b[i] = binary.LittleEndian.Uint16(slice[2*i:])
	}

	return b
}

func byteSliceAsUint64Slice(slice []byte) []uint64 {
	if len(slice)%8 != 0 {
		panic("Slice size should be divisible by 8")
	}

	b := make([]uint64, len(slice)/8)

	for i := range b {
		b[i] = binary.LittleEndian.Uint64(slice[8*i:])
	}

	return b
}

// Converts a byte slice to a interval16 slice.
// The function assumes that the slice byte buffer is run container data
// encoded according to Roaring Format Spec
func byteSliceAsInterval16Slice(byteSlice []byte) []interval16 {
	if len(byteSlice)%4 != 0 {
		panic("Slice size should be divisible by 4")
	}

	intervalSlice := make([]interval16, len(byteSlice)/4)

	for i := range intervalSlice {
		intervalSlice[i] = interval16{
			start:  binary.LittleEndian.Uint16(byteSlice[i*4:]),
			length: binary.LittleEndian.Uint16(byteSlice[i*4+2:]),
		}
	}

	return intervalSlice
}
