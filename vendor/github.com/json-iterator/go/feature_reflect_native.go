package jsoniter

import (
	"encoding"
	"encoding/base64"
	"encoding/json"
	"reflect"
	"unsafe"
)

type stringCodec struct {
}

func (codec *stringCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	*((*string)(ptr)) = iter.ReadString()
}

func (codec *stringCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	str := *((*string)(ptr))
	stream.WriteString(str)
}

func (codec *stringCodec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
}

func (codec *stringCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*string)(ptr)) == ""
}

type intCodec struct {
}

func (codec *intCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*int)(ptr)) = iter.ReadInt()
	}
}

func (codec *intCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteInt(*((*int)(ptr)))
}

func (codec *intCodec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
}

func (codec *intCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*int)(ptr)) == 0
}

type uintptrCodec struct {
}

func (codec *uintptrCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*uintptr)(ptr)) = uintptr(iter.ReadUint64())
	}
}

func (codec *uintptrCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteUint64(uint64(*((*uintptr)(ptr))))
}

func (codec *uintptrCodec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
}

func (codec *uintptrCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*uintptr)(ptr)) == 0
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

func (codec *int8Codec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
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

func (codec *int16Codec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
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

func (codec *int32Codec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
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

func (codec *int64Codec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
}

func (codec *int64Codec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*int64)(ptr)) == 0
}

type uintCodec struct {
}

func (codec *uintCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if !iter.ReadNil() {
		*((*uint)(ptr)) = iter.ReadUint()
		return
	}
}

func (codec *uintCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteUint(*((*uint)(ptr)))
}

func (codec *uintCodec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
}

func (codec *uintCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*uint)(ptr)) == 0
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

func (codec *uint8Codec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
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

func (codec *uint16Codec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
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

func (codec *uint32Codec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
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

func (codec *uint64Codec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
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

func (codec *float32Codec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
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

func (codec *float64Codec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
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

func (codec *boolCodec) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, codec)
}

func (codec *boolCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return !(*((*bool)(ptr)))
}

type emptyInterfaceCodec struct {
}

func (codec *emptyInterfaceCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	existing := *((*interface{})(ptr))

	// Checking for both typed and untyped nil pointers.
	if existing != nil &&
		reflect.TypeOf(existing).Kind() == reflect.Ptr &&
		!reflect.ValueOf(existing).IsNil() {

		var ptrToExisting interface{}
		for {
			elem := reflect.ValueOf(existing).Elem()
			if elem.Kind() != reflect.Ptr || elem.IsNil() {
				break
			}
			ptrToExisting = existing
			existing = elem.Interface()
		}

		if iter.ReadNil() {
			if ptrToExisting != nil {
				nilPtr := reflect.Zero(reflect.TypeOf(ptrToExisting).Elem())
				reflect.ValueOf(ptrToExisting).Elem().Set(nilPtr)
			} else {
				*((*interface{})(ptr)) = nil
			}
		} else {
			iter.ReadVal(existing)
		}

		return
	}

	if iter.ReadNil() {
		*((*interface{})(ptr)) = nil
	} else {
		*((*interface{})(ptr)) = iter.Read()
	}
}

func (codec *emptyInterfaceCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteVal(*((*interface{})(ptr)))
}

func (codec *emptyInterfaceCodec) EncodeInterface(val interface{}, stream *Stream) {
	stream.WriteVal(val)
}

func (codec *emptyInterfaceCodec) IsEmpty(ptr unsafe.Pointer) bool {
	emptyInterface := (*emptyInterface)(ptr)
	return emptyInterface.typ == nil
}

type nonEmptyInterfaceCodec struct {
}

func (codec *nonEmptyInterfaceCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	nonEmptyInterface := (*nonEmptyInterface)(ptr)
	if nonEmptyInterface.itab == nil {
		iter.ReportError("read non-empty interface", "do not know which concrete type to decode to")
		return
	}
	var i interface{}
	e := (*emptyInterface)(unsafe.Pointer(&i))
	e.typ = nonEmptyInterface.itab.typ
	e.word = nonEmptyInterface.word
	iter.ReadVal(&i)
	if e.word == nil {
		nonEmptyInterface.itab = nil
	}
	nonEmptyInterface.word = e.word
}

func (codec *nonEmptyInterfaceCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	nonEmptyInterface := (*nonEmptyInterface)(ptr)
	var i interface{}
	if nonEmptyInterface.itab != nil {
		e := (*emptyInterface)(unsafe.Pointer(&i))
		e.typ = nonEmptyInterface.itab.typ
		e.word = nonEmptyInterface.word
	}
	stream.WriteVal(i)
}

func (codec *nonEmptyInterfaceCodec) EncodeInterface(val interface{}, stream *Stream) {
	stream.WriteVal(val)
}

func (codec *nonEmptyInterfaceCodec) IsEmpty(ptr unsafe.Pointer) bool {
	nonEmptyInterface := (*nonEmptyInterface)(ptr)
	return nonEmptyInterface.word == nil
}

type anyCodec struct {
}

func (codec *anyCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	*((*Any)(ptr)) = iter.ReadAny()
}

func (codec *anyCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	(*((*Any)(ptr))).WriteTo(stream)
}

func (codec *anyCodec) EncodeInterface(val interface{}, stream *Stream) {
	(val.(Any)).WriteTo(stream)
}

func (codec *anyCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return (*((*Any)(ptr))).Size() == 0
}

type jsonNumberCodec struct {
}

func (codec *jsonNumberCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	switch iter.WhatIsNext() {
	case StringValue:
		*((*json.Number)(ptr)) = json.Number(iter.ReadString())
	case NilValue:
		iter.skipFourBytes('n', 'u', 'l', 'l')
		*((*json.Number)(ptr)) = ""
	default:
		*((*json.Number)(ptr)) = json.Number([]byte(iter.readNumberAsString()))
	}
}

func (codec *jsonNumberCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteRaw(string(*((*json.Number)(ptr))))
}

func (codec *jsonNumberCodec) EncodeInterface(val interface{}, stream *Stream) {
	stream.WriteRaw(string(val.(json.Number)))
}

func (codec *jsonNumberCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return len(*((*json.Number)(ptr))) == 0
}

type jsoniterNumberCodec struct {
}

func (codec *jsoniterNumberCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	switch iter.WhatIsNext() {
	case StringValue:
		*((*Number)(ptr)) = Number(iter.ReadString())
	case NilValue:
		iter.skipFourBytes('n', 'u', 'l', 'l')
		*((*Number)(ptr)) = ""
	default:
		*((*Number)(ptr)) = Number([]byte(iter.readNumberAsString()))
	}
}

func (codec *jsoniterNumberCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteRaw(string(*((*Number)(ptr))))
}

func (codec *jsoniterNumberCodec) EncodeInterface(val interface{}, stream *Stream) {
	stream.WriteRaw(string(val.(Number)))
}

func (codec *jsoniterNumberCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return len(*((*Number)(ptr))) == 0
}

type jsonRawMessageCodec struct {
}

func (codec *jsonRawMessageCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	*((*json.RawMessage)(ptr)) = json.RawMessage(iter.SkipAndReturnBytes())
}

func (codec *jsonRawMessageCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteRaw(string(*((*json.RawMessage)(ptr))))
}

func (codec *jsonRawMessageCodec) EncodeInterface(val interface{}, stream *Stream) {
	stream.WriteRaw(string(val.(json.RawMessage)))
}

func (codec *jsonRawMessageCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return len(*((*json.RawMessage)(ptr))) == 0
}

type jsoniterRawMessageCodec struct {
}

func (codec *jsoniterRawMessageCodec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	*((*RawMessage)(ptr)) = RawMessage(iter.SkipAndReturnBytes())
}

func (codec *jsoniterRawMessageCodec) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteRaw(string(*((*RawMessage)(ptr))))
}

func (codec *jsoniterRawMessageCodec) EncodeInterface(val interface{}, stream *Stream) {
	stream.WriteRaw(string(val.(RawMessage)))
}

func (codec *jsoniterRawMessageCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return len(*((*RawMessage)(ptr))) == 0
}

type base64Codec struct {
	sliceDecoder ValDecoder
}

func (codec *base64Codec) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if iter.ReadNil() {
		ptrSlice := (*sliceHeader)(ptr)
		ptrSlice.Len = 0
		ptrSlice.Cap = 0
		ptrSlice.Data = nil
		return
	}
	switch iter.WhatIsNext() {
	case StringValue:
		encoding := base64.StdEncoding
		src := iter.SkipAndReturnBytes()
		src = src[1 : len(src)-1]
		decodedLen := encoding.DecodedLen(len(src))
		dst := make([]byte, decodedLen)
		len, err := encoding.Decode(dst, src)
		if err != nil {
			iter.ReportError("decode base64", err.Error())
		} else {
			dst = dst[:len]
			dstSlice := (*sliceHeader)(unsafe.Pointer(&dst))
			ptrSlice := (*sliceHeader)(ptr)
			ptrSlice.Data = dstSlice.Data
			ptrSlice.Cap = dstSlice.Cap
			ptrSlice.Len = dstSlice.Len
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
	toGrow := encoding.EncodedLen(len(src))
	stream.ensure(toGrow)
	encoding.Encode(stream.buf[stream.n:], src)
	stream.n += toGrow
	stream.writeByte('"')
}

func (codec *base64Codec) EncodeInterface(val interface{}, stream *Stream) {
	ptr := extractInterface(val).word
	src := *((*[]byte)(ptr))
	if len(src) == 0 {
		stream.WriteNil()
		return
	}
	encoding := base64.StdEncoding
	stream.writeByte('"')
	toGrow := encoding.EncodedLen(len(src))
	stream.ensure(toGrow)
	encoding.Encode(stream.buf[stream.n:], src)
	stream.n += toGrow
	stream.writeByte('"')
}

func (codec *base64Codec) IsEmpty(ptr unsafe.Pointer) bool {
	return len(*((*[]byte)(ptr))) == 0
}

type stringModeNumberDecoder struct {
	elemDecoder ValDecoder
}

func (decoder *stringModeNumberDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	c := iter.nextToken()
	if c != '"' {
		iter.ReportError("stringModeNumberDecoder", `expect ", but found `+string([]byte{c}))
		return
	}
	decoder.elemDecoder.Decode(ptr, iter)
	if iter.Error != nil {
		return
	}
	c = iter.readByte()
	if c != '"' {
		iter.ReportError("stringModeNumberDecoder", `expect ", but found `+string([]byte{c}))
		return
	}
}

type stringModeStringDecoder struct {
	elemDecoder ValDecoder
	cfg         *frozenConfig
}

func (decoder *stringModeStringDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	decoder.elemDecoder.Decode(ptr, iter)
	str := *((*string)(ptr))
	tempIter := decoder.cfg.BorrowIterator([]byte(str))
	defer decoder.cfg.ReturnIterator(tempIter)
	*((*string)(ptr)) = tempIter.ReadString()
}

type stringModeNumberEncoder struct {
	elemEncoder ValEncoder
}

func (encoder *stringModeNumberEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.writeByte('"')
	encoder.elemEncoder.Encode(ptr, stream)
	stream.writeByte('"')
}

func (encoder *stringModeNumberEncoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *stringModeNumberEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return encoder.elemEncoder.IsEmpty(ptr)
}

type stringModeStringEncoder struct {
	elemEncoder ValEncoder
	cfg         *frozenConfig
}

func (encoder *stringModeStringEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	tempStream := encoder.cfg.BorrowStream(nil)
	defer encoder.cfg.ReturnStream(tempStream)
	encoder.elemEncoder.Encode(ptr, tempStream)
	stream.WriteString(string(tempStream.Buffer()))
}

func (encoder *stringModeStringEncoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *stringModeStringEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return encoder.elemEncoder.IsEmpty(ptr)
}

type marshalerEncoder struct {
	templateInterface emptyInterface
	checkIsEmpty      checkIsEmpty
}

func (encoder *marshalerEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	templateInterface := encoder.templateInterface
	templateInterface.word = ptr
	realInterface := (*interface{})(unsafe.Pointer(&templateInterface))
	marshaler, ok := (*realInterface).(json.Marshaler)
	if !ok {
		stream.WriteVal(nil)
		return
	}

	bytes, err := marshaler.MarshalJSON()
	if err != nil {
		stream.Error = err
	} else {
		stream.Write(bytes)
	}
}
func (encoder *marshalerEncoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *marshalerEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return encoder.checkIsEmpty.IsEmpty(ptr)
}

type textMarshalerEncoder struct {
	templateInterface emptyInterface
	checkIsEmpty      checkIsEmpty
}

func (encoder *textMarshalerEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	templateInterface := encoder.templateInterface
	templateInterface.word = ptr
	realInterface := (*interface{})(unsafe.Pointer(&templateInterface))
	marshaler := (*realInterface).(encoding.TextMarshaler)
	bytes, err := marshaler.MarshalText()
	if err != nil {
		stream.Error = err
	} else {
		stream.WriteString(string(bytes))
	}
}

func (encoder *textMarshalerEncoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *textMarshalerEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return encoder.checkIsEmpty.IsEmpty(ptr)
}

type unmarshalerDecoder struct {
	templateInterface emptyInterface
}

func (decoder *unmarshalerDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	templateInterface := decoder.templateInterface
	templateInterface.word = ptr
	realInterface := (*interface{})(unsafe.Pointer(&templateInterface))
	unmarshaler := (*realInterface).(json.Unmarshaler)
	iter.nextToken()
	iter.unreadByte() // skip spaces
	bytes := iter.SkipAndReturnBytes()
	err := unmarshaler.UnmarshalJSON(bytes)
	if err != nil {
		iter.ReportError("unmarshalerDecoder", err.Error())
	}
}

type textUnmarshalerDecoder struct {
	templateInterface emptyInterface
}

func (decoder *textUnmarshalerDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	templateInterface := decoder.templateInterface
	templateInterface.word = ptr
	realInterface := (*interface{})(unsafe.Pointer(&templateInterface))
	unmarshaler := (*realInterface).(encoding.TextUnmarshaler)
	str := iter.ReadString()
	err := unmarshaler.UnmarshalText([]byte(str))
	if err != nil {
		iter.ReportError("textUnmarshalerDecoder", err.Error())
	}
}
