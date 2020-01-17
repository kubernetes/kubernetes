package jsoniter

import (
	"fmt"
	"github.com/modern-go/reflect2"
	"io"
	"reflect"
	"sort"
	"unsafe"
)

func decoderOfMap(ctx *ctx, typ reflect2.Type) ValDecoder {
	mapType := typ.(*reflect2.UnsafeMapType)
	keyDecoder := decoderOfMapKey(ctx.append("[mapKey]"), mapType.Key())
	elemDecoder := decoderOfType(ctx.append("[mapElem]"), mapType.Elem())
	return &mapDecoder{
		mapType:     mapType,
		keyType:     mapType.Key(),
		elemType:    mapType.Elem(),
		keyDecoder:  keyDecoder,
		elemDecoder: elemDecoder,
	}
}

func encoderOfMap(ctx *ctx, typ reflect2.Type) ValEncoder {
	mapType := typ.(*reflect2.UnsafeMapType)
	if ctx.sortMapKeys {
		return &sortKeysMapEncoder{
			mapType:     mapType,
			keyEncoder:  encoderOfMapKey(ctx.append("[mapKey]"), mapType.Key()),
			elemEncoder: encoderOfType(ctx.append("[mapElem]"), mapType.Elem()),
		}
	}
	return &mapEncoder{
		mapType:     mapType,
		keyEncoder:  encoderOfMapKey(ctx.append("[mapKey]"), mapType.Key()),
		elemEncoder: encoderOfType(ctx.append("[mapElem]"), mapType.Elem()),
	}
}

func decoderOfMapKey(ctx *ctx, typ reflect2.Type) ValDecoder {
	decoder := ctx.decoderExtension.CreateMapKeyDecoder(typ)
	if decoder != nil {
		return decoder
	}
	for _, extension := range ctx.extraExtensions {
		decoder := extension.CreateMapKeyDecoder(typ)
		if decoder != nil {
			return decoder
		}
	}
	switch typ.Kind() {
	case reflect.String:
		return decoderOfType(ctx, reflect2.DefaultTypeOfKind(reflect.String))
	case reflect.Bool,
		reflect.Uint8, reflect.Int8,
		reflect.Uint16, reflect.Int16,
		reflect.Uint32, reflect.Int32,
		reflect.Uint64, reflect.Int64,
		reflect.Uint, reflect.Int,
		reflect.Float32, reflect.Float64,
		reflect.Uintptr:
		typ = reflect2.DefaultTypeOfKind(typ.Kind())
		return &numericMapKeyDecoder{decoderOfType(ctx, typ)}
	default:
		ptrType := reflect2.PtrTo(typ)
		if ptrType.Implements(unmarshalerType) {
			return &referenceDecoder{
				&unmarshalerDecoder{
					valType: ptrType,
				},
			}
		}
		if typ.Implements(unmarshalerType) {
			return &unmarshalerDecoder{
				valType: typ,
			}
		}
		if ptrType.Implements(textUnmarshalerType) {
			return &referenceDecoder{
				&textUnmarshalerDecoder{
					valType: ptrType,
				},
			}
		}
		if typ.Implements(textUnmarshalerType) {
			return &textUnmarshalerDecoder{
				valType: typ,
			}
		}
		return &lazyErrorDecoder{err: fmt.Errorf("unsupported map key type: %v", typ)}
	}
}

func encoderOfMapKey(ctx *ctx, typ reflect2.Type) ValEncoder {
	encoder := ctx.encoderExtension.CreateMapKeyEncoder(typ)
	if encoder != nil {
		return encoder
	}
	for _, extension := range ctx.extraExtensions {
		encoder := extension.CreateMapKeyEncoder(typ)
		if encoder != nil {
			return encoder
		}
	}
	switch typ.Kind() {
	case reflect.String:
		return encoderOfType(ctx, reflect2.DefaultTypeOfKind(reflect.String))
	case reflect.Bool,
		reflect.Uint8, reflect.Int8,
		reflect.Uint16, reflect.Int16,
		reflect.Uint32, reflect.Int32,
		reflect.Uint64, reflect.Int64,
		reflect.Uint, reflect.Int,
		reflect.Float32, reflect.Float64,
		reflect.Uintptr:
		typ = reflect2.DefaultTypeOfKind(typ.Kind())
		return &numericMapKeyEncoder{encoderOfType(ctx, typ)}
	default:
		if typ == textMarshalerType {
			return &directTextMarshalerEncoder{
				stringEncoder: ctx.EncoderOf(reflect2.TypeOf("")),
			}
		}
		if typ.Implements(textMarshalerType) {
			return &textMarshalerEncoder{
				valType:       typ,
				stringEncoder: ctx.EncoderOf(reflect2.TypeOf("")),
			}
		}
		if typ.Kind() == reflect.Interface {
			return &dynamicMapKeyEncoder{ctx, typ}
		}
		return &lazyErrorEncoder{err: fmt.Errorf("unsupported map key type: %v", typ)}
	}
}

type mapDecoder struct {
	mapType     *reflect2.UnsafeMapType
	keyType     reflect2.Type
	elemType    reflect2.Type
	keyDecoder  ValDecoder
	elemDecoder ValDecoder
}

func (decoder *mapDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	mapType := decoder.mapType
	c := iter.nextToken()
	if c == 'n' {
		iter.skipThreeBytes('u', 'l', 'l')
		*(*unsafe.Pointer)(ptr) = nil
		mapType.UnsafeSet(ptr, mapType.UnsafeNew())
		return
	}
	if mapType.UnsafeIsNil(ptr) {
		mapType.UnsafeSet(ptr, mapType.UnsafeMakeMap(0))
	}
	if c != '{' {
		iter.ReportError("ReadMapCB", `expect { or n, but found `+string([]byte{c}))
		return
	}
	c = iter.nextToken()
	if c == '}' {
		return
	}
	if c != '"' {
		iter.ReportError("ReadMapCB", `expect " after }, but found `+string([]byte{c}))
		return
	}
	iter.unreadByte()
	key := decoder.keyType.UnsafeNew()
	decoder.keyDecoder.Decode(key, iter)
	c = iter.nextToken()
	if c != ':' {
		iter.ReportError("ReadMapCB", "expect : after object field, but found "+string([]byte{c}))
		return
	}
	elem := decoder.elemType.UnsafeNew()
	decoder.elemDecoder.Decode(elem, iter)
	decoder.mapType.UnsafeSetIndex(ptr, key, elem)
	for c = iter.nextToken(); c == ','; c = iter.nextToken() {
		key := decoder.keyType.UnsafeNew()
		decoder.keyDecoder.Decode(key, iter)
		c = iter.nextToken()
		if c != ':' {
			iter.ReportError("ReadMapCB", "expect : after object field, but found "+string([]byte{c}))
			return
		}
		elem := decoder.elemType.UnsafeNew()
		decoder.elemDecoder.Decode(elem, iter)
		decoder.mapType.UnsafeSetIndex(ptr, key, elem)
	}
	if c != '}' {
		iter.ReportError("ReadMapCB", `expect }, but found `+string([]byte{c}))
	}
}

type numericMapKeyDecoder struct {
	decoder ValDecoder
}

func (decoder *numericMapKeyDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	c := iter.nextToken()
	if c != '"' {
		iter.ReportError("ReadMapCB", `expect ", but found `+string([]byte{c}))
		return
	}
	decoder.decoder.Decode(ptr, iter)
	c = iter.nextToken()
	if c != '"' {
		iter.ReportError("ReadMapCB", `expect ", but found `+string([]byte{c}))
		return
	}
}

type numericMapKeyEncoder struct {
	encoder ValEncoder
}

func (encoder *numericMapKeyEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.writeByte('"')
	encoder.encoder.Encode(ptr, stream)
	stream.writeByte('"')
}

func (encoder *numericMapKeyEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return false
}

type dynamicMapKeyEncoder struct {
	ctx     *ctx
	valType reflect2.Type
}

func (encoder *dynamicMapKeyEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	obj := encoder.valType.UnsafeIndirect(ptr)
	encoderOfMapKey(encoder.ctx, reflect2.TypeOf(obj)).Encode(reflect2.PtrOf(obj), stream)
}

func (encoder *dynamicMapKeyEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	obj := encoder.valType.UnsafeIndirect(ptr)
	return encoderOfMapKey(encoder.ctx, reflect2.TypeOf(obj)).IsEmpty(reflect2.PtrOf(obj))
}

type mapEncoder struct {
	mapType     *reflect2.UnsafeMapType
	keyEncoder  ValEncoder
	elemEncoder ValEncoder
}

func (encoder *mapEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteObjectStart()
	iter := encoder.mapType.UnsafeIterate(ptr)
	for i := 0; iter.HasNext(); i++ {
		if i != 0 {
			stream.WriteMore()
		}
		key, elem := iter.UnsafeNext()
		encoder.keyEncoder.Encode(key, stream)
		if stream.indention > 0 {
			stream.writeTwoBytes(byte(':'), byte(' '))
		} else {
			stream.writeByte(':')
		}
		encoder.elemEncoder.Encode(elem, stream)
	}
	stream.WriteObjectEnd()
}

func (encoder *mapEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	iter := encoder.mapType.UnsafeIterate(ptr)
	return !iter.HasNext()
}

type sortKeysMapEncoder struct {
	mapType     *reflect2.UnsafeMapType
	keyEncoder  ValEncoder
	elemEncoder ValEncoder
}

func (encoder *sortKeysMapEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	if *(*unsafe.Pointer)(ptr) == nil {
		stream.WriteNil()
		return
	}
	stream.WriteObjectStart()
	mapIter := encoder.mapType.UnsafeIterate(ptr)
	subStream := stream.cfg.BorrowStream(nil)
	subIter := stream.cfg.BorrowIterator(nil)
	keyValues := encodedKeyValues{}
	for mapIter.HasNext() {
		subStream.buf = make([]byte, 0, 64)
		key, elem := mapIter.UnsafeNext()
		encoder.keyEncoder.Encode(key, subStream)
		if subStream.Error != nil && subStream.Error != io.EOF && stream.Error == nil {
			stream.Error = subStream.Error
		}
		encodedKey := subStream.Buffer()
		subIter.ResetBytes(encodedKey)
		decodedKey := subIter.ReadString()
		if stream.indention > 0 {
			subStream.writeTwoBytes(byte(':'), byte(' '))
		} else {
			subStream.writeByte(':')
		}
		encoder.elemEncoder.Encode(elem, subStream)
		keyValues = append(keyValues, encodedKV{
			key:      decodedKey,
			keyValue: subStream.Buffer(),
		})
	}
	sort.Sort(keyValues)
	for i, keyValue := range keyValues {
		if i != 0 {
			stream.WriteMore()
		}
		stream.Write(keyValue.keyValue)
	}
	stream.WriteObjectEnd()
	stream.cfg.ReturnStream(subStream)
	stream.cfg.ReturnIterator(subIter)
}

func (encoder *sortKeysMapEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	iter := encoder.mapType.UnsafeIterate(ptr)
	return !iter.HasNext()
}

type encodedKeyValues []encodedKV

type encodedKV struct {
	key      string
	keyValue []byte
}

func (sv encodedKeyValues) Len() int           { return len(sv) }
func (sv encodedKeyValues) Swap(i, j int)      { sv[i], sv[j] = sv[j], sv[i] }
func (sv encodedKeyValues) Less(i, j int) bool { return sv[i].key < sv[j].key }
