package jsoniter

import (
	"fmt"
	"github.com/modern-go/reflect2"
	"io"
	"unsafe"
)

func decoderOfArray(ctx *ctx, typ reflect2.Type) ValDecoder {
	arrayType := typ.(*reflect2.UnsafeArrayType)
	decoder := decoderOfType(ctx.append("[arrayElem]"), arrayType.Elem())
	return &arrayDecoder{arrayType, decoder}
}

func encoderOfArray(ctx *ctx, typ reflect2.Type) ValEncoder {
	arrayType := typ.(*reflect2.UnsafeArrayType)
	if arrayType.Len() == 0 {
		return emptyArrayEncoder{}
	}
	encoder := encoderOfType(ctx.append("[arrayElem]"), arrayType.Elem())
	return &arrayEncoder{arrayType, encoder}
}

type emptyArrayEncoder struct{}

func (encoder emptyArrayEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteEmptyArray()
}

func (encoder emptyArrayEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return true
}

type arrayEncoder struct {
	arrayType   *reflect2.UnsafeArrayType
	elemEncoder ValEncoder
}

func (encoder *arrayEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteArrayStart()
	elemPtr := unsafe.Pointer(ptr)
	encoder.elemEncoder.Encode(elemPtr, stream)
	for i := 1; i < encoder.arrayType.Len(); i++ {
		stream.WriteMore()
		elemPtr = encoder.arrayType.UnsafeGetIndex(ptr, i)
		encoder.elemEncoder.Encode(elemPtr, stream)
	}
	stream.WriteArrayEnd()
	if stream.Error != nil && stream.Error != io.EOF {
		stream.Error = fmt.Errorf("%v: %s", encoder.arrayType, stream.Error.Error())
	}
}

func (encoder *arrayEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return false
}

type arrayDecoder struct {
	arrayType   *reflect2.UnsafeArrayType
	elemDecoder ValDecoder
}

func (decoder *arrayDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	decoder.doDecode(ptr, iter)
	if iter.Error != nil && iter.Error != io.EOF {
		iter.Error = fmt.Errorf("%v: %s", decoder.arrayType, iter.Error.Error())
	}
}

func (decoder *arrayDecoder) doDecode(ptr unsafe.Pointer, iter *Iterator) {
	c := iter.nextToken()
	arrayType := decoder.arrayType
	if c == 'n' {
		iter.skipThreeBytes('u', 'l', 'l')
		return
	}
	if c != '[' {
		iter.ReportError("decode array", "expect [ or n, but found "+string([]byte{c}))
		return
	}
	c = iter.nextToken()
	if c == ']' {
		return
	}
	iter.unreadByte()
	elemPtr := arrayType.UnsafeGetIndex(ptr, 0)
	decoder.elemDecoder.Decode(elemPtr, iter)
	length := 1
	for c = iter.nextToken(); c == ','; c = iter.nextToken() {
		if length >= arrayType.Len() {
			iter.Skip()
			continue
		}
		idx := length
		length += 1
		elemPtr = arrayType.UnsafeGetIndex(ptr, idx)
		decoder.elemDecoder.Decode(elemPtr, iter)
	}
	if c != ']' {
		iter.ReportError("decode array", "expect ], but found "+string([]byte{c}))
		return
	}
}
