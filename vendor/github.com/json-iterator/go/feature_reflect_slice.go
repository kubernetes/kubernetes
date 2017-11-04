package jsoniter

import (
	"fmt"
	"io"
	"reflect"
	"unsafe"
)

func decoderOfSlice(cfg *frozenConfig, typ reflect.Type) (ValDecoder, error) {
	decoder, err := decoderOfType(cfg, typ.Elem())
	if err != nil {
		return nil, err
	}
	return &sliceDecoder{typ, typ.Elem(), decoder}, nil
}

func encoderOfSlice(cfg *frozenConfig, typ reflect.Type) (ValEncoder, error) {
	encoder, err := encoderOfType(cfg, typ.Elem())
	if err != nil {
		return nil, err
	}
	if typ.Elem().Kind() == reflect.Map {
		encoder = &optionalEncoder{encoder}
	}
	return &sliceEncoder{typ, typ.Elem(), encoder}, nil
}

type sliceEncoder struct {
	sliceType   reflect.Type
	elemType    reflect.Type
	elemEncoder ValEncoder
}

func (encoder *sliceEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	slice := (*sliceHeader)(ptr)
	if slice.Data == nil {
		stream.WriteNil()
		return
	}
	if slice.Len == 0 {
		stream.WriteEmptyArray()
		return
	}
	stream.WriteArrayStart()
	elemPtr := unsafe.Pointer(slice.Data)
	encoder.elemEncoder.Encode(unsafe.Pointer(elemPtr), stream)
	for i := 1; i < slice.Len; i++ {
		stream.WriteMore()
		elemPtr = unsafe.Pointer(uintptr(elemPtr) + encoder.elemType.Size())
		encoder.elemEncoder.Encode(unsafe.Pointer(elemPtr), stream)
	}
	stream.WriteArrayEnd()
	if stream.Error != nil && stream.Error != io.EOF {
		stream.Error = fmt.Errorf("%v: %s", encoder.sliceType, stream.Error.Error())
	}
}

func (encoder *sliceEncoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *sliceEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	slice := (*sliceHeader)(ptr)
	return slice.Len == 0
}

type sliceDecoder struct {
	sliceType   reflect.Type
	elemType    reflect.Type
	elemDecoder ValDecoder
}

// sliceHeader is a safe version of SliceHeader used within this package.
type sliceHeader struct {
	Data unsafe.Pointer
	Len  int
	Cap  int
}

func (decoder *sliceDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	decoder.doDecode(ptr, iter)
	if iter.Error != nil && iter.Error != io.EOF {
		iter.Error = fmt.Errorf("%v: %s", decoder.sliceType, iter.Error.Error())
	}
}

func (decoder *sliceDecoder) doDecode(ptr unsafe.Pointer, iter *Iterator) {
	slice := (*sliceHeader)(ptr)
	if iter.ReadNil() {
		slice.Len = 0
		slice.Cap = 0
		slice.Data = nil
		return
	}
	reuseSlice(slice, decoder.sliceType, 4)
	slice.Len = 0
	offset := uintptr(0)
	iter.ReadArrayCB(func(iter *Iterator) bool {
		growOne(slice, decoder.sliceType, decoder.elemType)
		decoder.elemDecoder.Decode(unsafe.Pointer(uintptr(slice.Data)+offset), iter)
		offset += decoder.elemType.Size()
		return true
	})
}

// grow grows the slice s so that it can hold extra more values, allocating
// more capacity if needed. It also returns the old and new slice lengths.
func growOne(slice *sliceHeader, sliceType reflect.Type, elementType reflect.Type) {
	newLen := slice.Len + 1
	if newLen <= slice.Cap {
		slice.Len = newLen
		return
	}
	newCap := slice.Cap
	if newCap == 0 {
		newCap = 1
	} else {
		for newCap < newLen {
			if slice.Len < 1024 {
				newCap += newCap
			} else {
				newCap += newCap / 4
			}
		}
	}
	newVal := reflect.MakeSlice(sliceType, newLen, newCap)
	dst := unsafe.Pointer(newVal.Pointer())
	// copy old array into new array
	originalBytesCount := uintptr(slice.Len) * elementType.Size()
	srcPtr := (*[1 << 30]byte)(slice.Data)
	dstPtr := (*[1 << 30]byte)(dst)
	for i := uintptr(0); i < originalBytesCount; i++ {
		dstPtr[i] = srcPtr[i]
	}
	slice.Data = dst
	slice.Len = newLen
	slice.Cap = newCap
}

func reuseSlice(slice *sliceHeader, sliceType reflect.Type, expectedCap int) {
	if expectedCap <= slice.Cap {
		return
	}
	newVal := reflect.MakeSlice(sliceType, 0, expectedCap)
	dst := unsafe.Pointer(newVal.Pointer())
	slice.Data = dst
	slice.Cap = expectedCap
}
