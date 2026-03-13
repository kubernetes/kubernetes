package jsoniter

import (
	"github.com/modern-go/reflect2"
	"unsafe"
)

func decoderOfOptional(ctx *ctx, typ reflect2.Type) ValDecoder {
	ptrType := typ.(*reflect2.UnsafePtrType)
	elemType := ptrType.Elem()
	decoder := decoderOfType(ctx, elemType)
	return &OptionalDecoder{elemType, decoder}
}

func encoderOfOptional(ctx *ctx, typ reflect2.Type) ValEncoder {
	ptrType := typ.(*reflect2.UnsafePtrType)
	elemType := ptrType.Elem()
	elemEncoder := encoderOfType(ctx, elemType)
	encoder := &OptionalEncoder{elemEncoder}
	return encoder
}

type OptionalDecoder struct {
	ValueType    reflect2.Type
	ValueDecoder ValDecoder
}

func (decoder *OptionalDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if iter.ReadNil() {
		*((*unsafe.Pointer)(ptr)) = nil
	} else {
		if *((*unsafe.Pointer)(ptr)) == nil {
			//pointer to null, we have to allocate memory to hold the value
			newPtr := decoder.ValueType.UnsafeNew()
			decoder.ValueDecoder.Decode(newPtr, iter)
			*((*unsafe.Pointer)(ptr)) = newPtr
		} else {
			//reuse existing instance
			decoder.ValueDecoder.Decode(*((*unsafe.Pointer)(ptr)), iter)
		}
	}
}

type dereferenceDecoder struct {
	// only to deference a pointer
	valueType    reflect2.Type
	valueDecoder ValDecoder
}

func (decoder *dereferenceDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if *((*unsafe.Pointer)(ptr)) == nil {
		//pointer to null, we have to allocate memory to hold the value
		newPtr := decoder.valueType.UnsafeNew()
		decoder.valueDecoder.Decode(newPtr, iter)
		*((*unsafe.Pointer)(ptr)) = newPtr
	} else {
		//reuse existing instance
		decoder.valueDecoder.Decode(*((*unsafe.Pointer)(ptr)), iter)
	}
}

type OptionalEncoder struct {
	ValueEncoder ValEncoder
}

func (encoder *OptionalEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	if *((*unsafe.Pointer)(ptr)) == nil {
		stream.WriteNil()
	} else {
		encoder.ValueEncoder.Encode(*((*unsafe.Pointer)(ptr)), stream)
	}
}

func (encoder *OptionalEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*unsafe.Pointer)(ptr)) == nil
}

type dereferenceEncoder struct {
	ValueEncoder ValEncoder
}

func (encoder *dereferenceEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	if *((*unsafe.Pointer)(ptr)) == nil {
		stream.WriteNil()
	} else {
		encoder.ValueEncoder.Encode(*((*unsafe.Pointer)(ptr)), stream)
	}
}

func (encoder *dereferenceEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	dePtr := *((*unsafe.Pointer)(ptr))
	if dePtr == nil {
		return true
	}
	return encoder.ValueEncoder.IsEmpty(dePtr)
}

func (encoder *dereferenceEncoder) IsEmbeddedPtrNil(ptr unsafe.Pointer) bool {
	deReferenced := *((*unsafe.Pointer)(ptr))
	if deReferenced == nil {
		return true
	}
	isEmbeddedPtrNil, converted := encoder.ValueEncoder.(IsEmbeddedPtrNil)
	if !converted {
		return false
	}
	fieldPtr := unsafe.Pointer(deReferenced)
	return isEmbeddedPtrNil.IsEmbeddedPtrNil(fieldPtr)
}

type referenceEncoder struct {
	encoder ValEncoder
}

func (encoder *referenceEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	encoder.encoder.Encode(unsafe.Pointer(&ptr), stream)
}

func (encoder *referenceEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return encoder.encoder.IsEmpty(unsafe.Pointer(&ptr))
}

type referenceDecoder struct {
	decoder ValDecoder
}

func (decoder *referenceDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	decoder.decoder.Decode(unsafe.Pointer(&ptr), iter)
}
