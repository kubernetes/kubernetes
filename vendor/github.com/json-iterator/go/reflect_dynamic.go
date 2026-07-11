package jsoniter

import (
	"github.com/modern-go/reflect2"
	"reflect"
	"unsafe"
)

type dynamicEncoder struct {
	valType reflect2.Type
}

func (encoder *dynamicEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	obj := encoder.valType.UnsafeIndirect(ptr)
	stream.WriteVal(obj)
}

func (encoder *dynamicEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return encoder.valType.UnsafeIndirect(ptr) == nil
}

type efaceDecoder struct {
}

func (decoder *efaceDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	pObj := (*interface{})(ptr)
	obj := *pObj
	if obj == nil {
		*pObj = iter.Read()
		return
	}
	typ := reflect2.TypeOf(obj)
	if typ.Kind() != reflect.Ptr {
		*pObj = iter.Read()
		return
	}
	ptrType := typ.(*reflect2.UnsafePtrType)
	ptrElemType := ptrType.Elem()
	if iter.WhatIsNext() == NilValue {
		if ptrElemType.Kind() != reflect.Ptr {
			iter.skipFourBytes('n', 'u', 'l', 'l')
			*pObj = nil
			return
		}
	}
	if reflect2.IsNil(obj) {
		obj := ptrElemType.New()
		iter.ReadVal(obj)
		*pObj = obj
		return
	}
	iter.ReadVal(obj)
}

type ifaceDecoder struct {
	valType *reflect2.UnsafeIFaceType
}

func (decoder *ifaceDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if iter.ReadNil() {
		decoder.valType.UnsafeSet(ptr, decoder.valType.UnsafeNew())
		return
	}
	obj := decoder.valType.UnsafeIndirect(ptr)
	if reflect2.IsNil(obj) {
		iter.ReportError("decode non empty interface", "can not unmarshal into nil")
		return
	}
	iter.ReadVal(obj)
}
