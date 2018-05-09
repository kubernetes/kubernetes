package jsoniter

import (
	"bytes"
	"encoding/base64"
	"reflect"
	"strconv"
	"unsafe"

	"github.com/modern-go/reflect2"
)

const ptrSize = 32 << uintptr(^uintptr(0)>>63)

func createEncoderOfNative(ctx *ctx, typ reflect2.Type) ValEncoder {
	if typ.Kind() == reflect.Slice && typ.(reflect2.SliceType).Elem().Kind() == reflect.Uint8 {
		sliceDecoder := decoderOfSlice(ctx, typ)
		return &base64Codec{sliceDecoder: sliceDecoder}
	}
	typeName := typ.String()
	kind := typ.Kind()
	switch kind {
	case reflect.String:
		if typeName != "string" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*string)(nil)).Elem())
		}
		return &stringCodec{}
	case reflect.Int:
		if typeName != "int" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*int)(nil)).Elem())
		}
		if strconv.IntSize == 32 {
			return &int32Codec{}
		}
		return &int64Codec{}
	case reflect.Int8:
		if typeName != "int8" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*int8)(nil)).Elem())
		}
		return &int8Codec{}
	case reflect.Int16:
		if typeName != "int16" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*int16)(nil)).Elem())
		}
		return &int16Codec{}
	case reflect.Int32:
		if typeName != "int32" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*int32)(nil)).Elem())
		}
		return &int32Codec{}
	case reflect.Int64:
		if typeName != "int64" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*int64)(nil)).Elem())
		}
		return &int64Codec{}
	case reflect.Uint:
		if typeName != "uint" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*uint)(nil)).Elem())
		}
		if strconv.IntSize == 32 {
			return &uint32Codec{}
		}
		return &uint64Codec{}
	case reflect.Uint8:
		if typeName != "uint8" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*uint8)(nil)).Elem())
		}
		return &uint8Codec{}
	case reflect.Uint16:
		if typeName != "uint16" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*uint16)(nil)).Elem())
		}
		return &uint16Codec{}
	case reflect.Uint32:
		if typeName != "uint32" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*uint32)(nil)).Elem())
		}
		return &uint32Codec{}
	case reflect.Uintptr:
		if typeName != "uintptr" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*uintptr)(nil)).Elem())
		}
		if ptrSize == 32 {
			return &uint32Codec{}
		}
		return &uint64Codec{}
	case reflect.Uint64:
		if typeName != "uint64" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*uint64)(nil)).Elem())
		}
		return &uint64Codec{}
	case reflect.Float32:
		if typeName != "float32" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*float32)(nil)).Elem())
		}
		return &float32Codec{}
	case reflect.Float64:
		if typeName != "float64" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*float64)(nil)).Elem())
		}
		return &float64Codec{}
	case reflect.Bool:
		if typeName != "bool" {
			return encoderOfType(ctx, reflect2.TypeOfPtr((*bool)(nil)).Elem())
		}
		return &boolCodec{}
	}
	return nil
}

func createDecoderOfNative(ctx *ctx, typ reflect2.Type) ValDecoder {
	if typ.Kind() == reflect.Slice && typ.(reflect2.SliceType).Elem().Kind() == reflect.Uint8 {
		sliceDecoder := decoderOfSlice(ctx, typ)
		return &base64Codec{sliceDecoder: sliceDecoder}
	}
	typeName := typ.String()
	switch typ.Kind() {
	case reflect.String:
		if typeName != "string" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*string)(nil)).Elem())
		}
		return &stringCodec{}
	case reflect.Int:
		if typeName != "int" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*int)(nil)).Elem())
		}
		if strconv.IntSize == 32 {
			return &int32Codec{}
		}
		return &int64Codec{}
	case reflect.Int8:
		if typeName != "int8" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*int8)(nil)).Elem())
		}
		return &int8Codec{}
	case reflect.Int16:
		if typeName != "int16" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*int16)(nil)).Elem())
		}
		return &int16Codec{}
	case reflect.Int32:
		if typeName != "int32" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*int32)(nil)).Elem())
		}
		return &int32Codec{}
	case reflect.Int64:
		if typeName != "int64" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*int64)(nil)).Elem())
		}
		return &int64Codec{}
	case reflect.Uint:
		if typeName != "uint" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*uint)(nil)).Elem())
		}
		if strconv.IntSize == 32 {
			return &uint32Codec{}
		}
		return &uint64Codec{}
	case reflect.Uint8:
		if typeName != "uint8" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*uint8)(nil)).Elem())
		}
		return &uint8Codec{}
	case reflect.Uint16:
		if typeName != "uint16" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*uint16)(nil)).Elem())
		}
		return &uint16Codec{}
	case reflect.Uint32:
		if typeName != "uint32" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*uint32)(nil)).Elem())
		}
		return &uint32Codec{}
	case reflect.Uintptr:
		if typeName != "uintptr" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*uintptr)(nil)).Elem())
		}
		if ptrSize == 32 {
			return &uint32Codec{}
		}
		return &uint64Codec{}
	case reflect.Uint64:
		if typeName != "uint64" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*uint64)(nil)).Elem())
		}
		return &uint64Codec{}
	case reflect.Float32:
		if typeName != "float32" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*float32)(nil)).Elem())
		}
		return &float32Codec{}
	case reflect.Float64:
		if typeName != "float64" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*float64)(nil)).Elem())
		}
		return &float64Codec{}
	case reflect.Bool:
		if typeName != "bool" {
			return decoderOfType(ctx, reflect2.TypeOfPtr((*bool)(nil)).Elem())
		}
		return &boolCodec{}
	}
	return nil
}

type stringCodec struct {
}

func (codec *stringCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	*((*string)(ptr)) = iter.ReadString()
}

func (codec *stringCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	str := *((*string)(ptr))
	stream.WriteString(str)
}

func (codec *stringCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*string)(ptr)) == ""
}

type int8Codec struct {
}

func (codec *int8Codec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*int8)(ptr)) = iter.ReadInt8()
	}
}

func (codec *int8Codec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteInt8(*((*int8)(ptr)))
}

func (codec *int8Codec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*int8)(ptr)) == 0
}

type int16Codec struct {
}

func (codec *int16Codec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*int16)(ptr)) = iter.ReadInt16()
	}
}

func (codec *int16Codec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteInt16(*((*int16)(ptr)))
}

func (codec *int16Codec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*int16)(ptr)) == 0
}

type int32Codec struct {
}

func (codec *int32Codec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*int32)(ptr)) = iter.ReadInt32()
	}
}

func (codec *int32Codec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteInt32(*((*int32)(ptr)))
}

func (codec *int32Codec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*int32)(ptr)) == 0
}

type int64Codec struct {
}

func (codec *int64Codec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*int64)(ptr)) = iter.ReadInt64()
	}
}

func (codec *int64Codec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteInt64(*((*int64)(ptr)))
}

func (codec *int64Codec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*int64)(ptr)) == 0
}

type uint8Codec struct {
}

func (codec *uint8Codec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*uint8)(ptr)) = iter.ReadUint8()
	}
}

func (codec *uint8Codec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteUint8(*((*uint8)(ptr)))
}

func (codec *uint8Codec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*uint8)(ptr)) == 0
}

type uint16Codec struct {
}

func (codec *uint16Codec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*uint16)(ptr)) = iter.ReadUint16()
	}
}

func (codec *uint16Codec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteUint16(*((*uint16)(ptr)))
}

func (codec *uint16Codec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*uint16)(ptr)) == 0
}

type uint32Codec struct {
}

func (codec *uint32Codec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*uint32)(ptr)) = iter.ReadUint32()
	}
}

func (codec *uint32Codec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteUint32(*((*uint32)(ptr)))
}

func (codec *uint32Codec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*uint32)(ptr)) == 0
}

type uint64Codec struct {
}

func (codec *uint64Codec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*uint64)(ptr)) = iter.ReadUint64()
	}
}

func (codec *uint64Codec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteUint64(*((*uint64)(ptr)))
}

func (codec *uint64Codec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*uint64)(ptr)) == 0
}

type float32Codec struct {
}

func (codec *float32Codec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*float32)(ptr)) = iter.ReadFloat32()
	}
}

func (codec *float32Codec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteFloat32(*((*float32)(ptr)))
}

func (codec *float32Codec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*float32)(ptr)) == 0
}

type float64Codec struct {
}

func (codec *float64Codec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*float64)(ptr)) = iter.ReadFloat64()
	}
}

func (codec *float64Codec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteFloat64(*((*float64)(ptr)))
}

func (codec *float64Codec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*float64)(ptr)) == 0
}

type boolCodec struct {
}

func (codec *boolCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*bool)(ptr)) = iter.ReadBool()
	}
}

func (codec *boolCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteBool(*((*bool)(ptr)))
}

func (codec *boolCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return !(*((*bool)(ptr)))
}

type base64Codec struct {
	sliceType    *reflect2.UnsafeSliceType
	sliceDecoder ValDecoder
}

func (codec *base64Codec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if iter.ReadNil() {
		codec.sliceType.UnsafeSetNil(ptr)
		return
	}
	switch iter.WhatIsNext() {
	case StringValue:
		encoding := base64.StdEncoding
		src := iter.SkipAndReturnBytes()
		// New line characters (\r and \n) are ignored.
		// Refer to https://golang.org/pkg/encoding/base64/#Encoding.Decode
		src = bytes.Replace(src, []byte(`\r`), []byte{}, -1)
		src = bytes.Replace(src, []byte(`\n`), []byte{}, -1)
		src = src[1 : len(src)-1]
		decodedLen := encoding.DecodedLen(len(src))
		dst := make([]byte, decodedLen)
		len, err := encoding.Decode(dst, src)
		if err != nil {
			iter.ReportError("decode base64", err.Error())
		} else {
			dst = dst[:len]
			codec.sliceType.UnsafeSet(ptr, unsafe.Pointer(&dst))
		}
	case ArrayValue:
		codec.sliceDecoder.Decode(ptr, iter)
	default:
		iter.ReportError("base64Codec", "invalid input")
	}
}

func (codec *base64Codec) Encode(ptr unsafe.Pointer, stream *Stream) {
	src := *((*[]byte)(ptr))
	if len(src) == 0 {
		stream.WriteNil()
		return
	}
	encoding := base64.StdEncoding
	stream.writeByte('"')
	size := encoding.EncodedLen(len(src))
	buf := make([]byte, size)
	encoding.Encode(buf, src)
	stream.buf = append(stream.buf, buf...)
	stream.writeByte('"')
}

func (codec *base64Codec) IsEmpty(ptr unsafe.Pointer) bool {
	return len(*((*[]byte)(ptr))) == 0
}
