// +build 386 amd64,!appengine

package roaring

import (
	"errors"
	"io"
	"reflect"
	"runtime"
	"unsafe"
)

func (ac *arrayContainer) writeTo(stream io.Writer) (int, error) {
	buf := uint16SliceAsByteSlice(ac.content)
	return stream.Write(buf)
}

func (bc *bitmapContainer) writeTo(stream io.Writer) (int, error) {
	if bc.cardinality <= arrayDefaultMaxSize {
		return 0, errors.New("refusing to write bitmap container with cardinality of array container")
	}
	buf := uint64SliceAsByteSlice(bc.bitmap)
	return stream.Write(buf)
}

func uint64SliceAsByteSlice(slice []uint64) []byte {
	// make a new slice header
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&slice))

	// update its capacity and length
	header.Len *= 8
	header.Cap *= 8

	// instantiate result and use KeepAlive so data isn't unmapped.
	result := *(*[]byte)(unsafe.Pointer(&header))
	runtime.KeepAlive(&slice)

	// return it
	return result
}

func uint16SliceAsByteSlice(slice []uint16) []byte {
	// make a new slice header
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&slice))

	// update its capacity and length
	header.Len *= 2
	header.Cap *= 2

	// instantiate result and use KeepAlive so data isn't unmapped.
	result := *(*[]byte)(unsafe.Pointer(&header))
	runtime.KeepAlive(&slice)

	// return it
	return result
}

func (bc *bitmapContainer) asLittleEndianByteSlice() []byte {
	return uint64SliceAsByteSlice(bc.bitmap)
}

// Deserialization code follows

////
// These methods (byteSliceAsUint16Slice,...) do not make copies,
// they are pointer-based (unsafe). The caller is responsible to
// ensure that the input slice does not get garbage collected, deleted
// or modified while you hold the returned slince.
////
func byteSliceAsUint16Slice(slice []byte) (result []uint16) { // here we create a new slice holder
	if len(slice)%2 != 0 {
		panic("Slice size should be divisible by 2")
	}
	// reference: https://go101.org/article/unsafe.html

	// make a new slice header
	bHeader := (*reflect.SliceHeader)(unsafe.Pointer(&slice))
	rHeader := (*reflect.SliceHeader)(unsafe.Pointer(&result))

	// transfer the data from the given slice to a new variable (our result)
	rHeader.Data = bHeader.Data
	rHeader.Len = bHeader.Len / 2
	rHeader.Cap = bHeader.Cap / 2

	// instantiate result and use KeepAlive so data isn't unmapped.
	runtime.KeepAlive(&slice) // it is still crucial, GC can free it)

	// return result
	return
}

func byteSliceAsUint64Slice(slice []byte) (result []uint64) {
	if len(slice)%8 != 0 {
		panic("Slice size should be divisible by 8")
	}
	// reference: https://go101.org/article/unsafe.html

	// make a new slice header
	bHeader := (*reflect.SliceHeader)(unsafe.Pointer(&slice))
	rHeader := (*reflect.SliceHeader)(unsafe.Pointer(&result))

	// transfer the data from the given slice to a new variable (our result)
	rHeader.Data = bHeader.Data
	rHeader.Len = bHeader.Len / 8
	rHeader.Cap = bHeader.Cap / 8

	// instantiate result and use KeepAlive so data isn't unmapped.
	runtime.KeepAlive(&slice) // it is still crucial, GC can free it)

	// return result
	return
}

func byteSliceAsInterval16Slice(slice []byte) (result []interval16) {
	if len(slice)%4 != 0 {
		panic("Slice size should be divisible by 4")
	}
	// reference: https://go101.org/article/unsafe.html

	// make a new slice header
	bHeader := (*reflect.SliceHeader)(unsafe.Pointer(&slice))
	rHeader := (*reflect.SliceHeader)(unsafe.Pointer(&result))

	// transfer the data from the given slice to a new variable (our result)
	rHeader.Data = bHeader.Data
	rHeader.Len = bHeader.Len / 4
	rHeader.Cap = bHeader.Cap / 4

	// instantiate result and use KeepAlive so data isn't unmapped.
	runtime.KeepAlive(&slice) // it is still crucial, GC can free it)

	// return result
	return
}
