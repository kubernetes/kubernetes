// +build 386 amd64,!appengine

package roaring

import (
	"io"
	"reflect"
	"unsafe"
)

func (ac *arrayContainer) writeTo(stream io.Writer) (int, error) {
	buf := uint16SliceAsByteSlice(ac.content)
	return stream.Write(buf)
}

func (bc *bitmapContainer) writeTo(stream io.Writer) (int, error) {
	buf := uint64SliceAsByteSlice(bc.bitmap)
	return stream.Write(buf)
}

// readFrom reads an arrayContainer from stream.
// PRE-REQUISITE: you must size the arrayContainer correctly (allocate b.content)
// *before* you call readFrom. We can't guess the size in the stream
// by this point.
func (ac *arrayContainer) readFrom(stream io.Reader) (int, error) {
	buf := uint16SliceAsByteSlice(ac.content)
	return io.ReadFull(stream, buf)
}

func (bc *bitmapContainer) readFrom(stream io.Reader) (int, error) {
	buf := uint64SliceAsByteSlice(bc.bitmap)
	n, err := io.ReadFull(stream, buf)
	bc.computeCardinality()
	return n, err
}

func uint64SliceAsByteSlice(slice []uint64) []byte {
	// make a new slice header
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&slice))

	// update its capacity and length
	header.Len *= 8
	header.Cap *= 8

	// return it
	return *(*[]byte)(unsafe.Pointer(&header))
}

func uint16SliceAsByteSlice(slice []uint16) []byte {
	// make a new slice header
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&slice))

	// update its capacity and length
	header.Len *= 2
	header.Cap *= 2

	// return it
	return *(*[]byte)(unsafe.Pointer(&header))
}

func (bc *bitmapContainer) asLittleEndianByteSlice() []byte {
	return uint64SliceAsByteSlice(bc.bitmap)
}
