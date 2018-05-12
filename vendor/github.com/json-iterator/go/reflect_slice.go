package jsoniter

import (
	"fmt"
	"github.com/modern-go/reflect2"
	"io"
	"unsafe"
)

func decoderOfSlice(ctx *ctx, typ reflect2.Type) ValDecoder {
	sliceType := typ.(*reflect2.UnsafeSliceType)
	decoder := decoderOfType(ctx.append("[sliceElem]"), sliceType.Elem())
	return &sliceDecoder{sliceType, decoder}
}

func encoderOfSlice(ctx *ctx, typ reflect2.Type) ValEncoder {
	sliceType := typ.(*reflect2.UnsafeSliceType)
	encoder := encoderOfType(ctx.append("[sliceElem]"), sliceType.Elem())
	return &sliceEncoder{sliceType, encoder}
}

type sliceEncoder struct {
	sliceType   *reflect2.UnsafeSliceType
	elemEncoder ValEncoder
}

func (encoder *sliceEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	if encoder.sliceType.UnsafeIsNil(ptr) {
		stream.WriteNil()
		return
	}
	length := encoder.sliceType.UnsafeLengthOf(ptr)
	if length == 0 {
		stream.WriteEmptyArray()
		return
	}
	stream.WriteArrayStart()
	encoder.elemEncoder.Encode(encoder.sliceType.UnsafeGetIndex(ptr, 0), stream)
	for i := 1; i < length; i++ {
		stream.WriteMore()
		elemPtr := encoder.sliceType.UnsafeGetIndex(ptr, i)
		encoder.elemEncoder.Encode(elemPtr, stream)
	}
	stream.WriteArrayEnd()
	if stream.Error != nil && stream.Error != io.EOF {
		stream.Error = fmt.Errorf("%v: %s", encoder.sliceType, stream.Error.Error())
	}
}

func (encoder *sliceEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return encoder.sliceType.UnsafeLengthOf(ptr) == 0
}

type sliceDecoder struct {
	sliceType   *reflect2.UnsafeSliceType
	elemDecoder ValDecoder
}

func (decoder *sliceDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	decoder.doDecode(ptr, iter)
	if iter.Error != nil && iter.Error != io.EOF {
		iter.Error = fmt.Errorf("%v: %s", decoder.sliceType, iter.Error.Error())
	}
}

func (decoder *sliceDecoder) doDecode(ptr unsafe.Pointer, iter *Iterator) {
	c := iter.nextToken()
	sliceType := decoder.sliceType
	if c == 'n' {
		iter.skipThreeBytes('u', 'l', 'l')
		sliceType.UnsafeSetNil(ptr)
		return
	}
	if c != '[' {
		iter.ReportError("decode slice", "expect [ or n, but found "+string([]byte{c}))
		return
	}
	c = iter.nextToken()
	if c == ']' {
		sliceType.UnsafeSet(ptr, sliceType.UnsafeMakeSlice(0, 0))
		return
	}
	iter.unreadByte()
	sliceType.UnsafeGrow(ptr, 1)
	elemPtr := sliceType.UnsafeGetIndex(ptr, 0)
	decoder.elemDecoder.Decode(elemPtr, iter)
	length := 1
	for c = iter.nextToken(); c == ','; c = iter.nextToken() {
		idx := length
		length += 1
		sliceType.UnsafeGrow(ptr, length)
		elemPtr = sliceType.UnsafeGetIndex(ptr, idx)
		decoder.elemDecoder.Decode(elemPtr, iter)
	}
	if c != ']' {
		iter.ReportError("decode slice", "expect ], but found "+string([]byte{c}))
		return
	}
}
