package msgpack

import (
	"fmt"
	"io/ioutil"
	"reflect"
	"sync"
)

var (
	marshalerType   = reflect.TypeOf(new(Marshaler)).Elem()
	unmarshalerType = reflect.TypeOf(new(Unmarshaler)).Elem()

	encoderType = reflect.TypeOf(new(CustomEncoder)).Elem()
	decoderType = reflect.TypeOf(new(CustomDecoder)).Elem()
)

type (
	encoderFunc func(*Encoder, reflect.Value) error
	decoderFunc func(*Decoder, reflect.Value) error
)

var (
	typEncMap = make(map[reflect.Type]encoderFunc)
	typDecMap = make(map[reflect.Type]decoderFunc)
)

// Register registers encoder and decoder functions for a type.
// In most cases you should prefer implementing CustomEncoder and
// CustomDecoder interfaces.
func Register(typ reflect.Type, enc encoderFunc, dec decoderFunc) {
	typEncMap[typ] = enc
	typDecMap[typ] = dec
}

var structs = newStructCache()

var (
	valueEncoders []encoderFunc
	valueDecoders []decoderFunc
)

func init() {
	valueEncoders = []encoderFunc{
		reflect.Bool:          encodeBoolValue,
		reflect.Int:           encodeInt64Value,
		reflect.Int8:          encodeInt64Value,
		reflect.Int16:         encodeInt64Value,
		reflect.Int32:         encodeInt64Value,
		reflect.Int64:         encodeInt64Value,
		reflect.Uint:          encodeUint64Value,
		reflect.Uint8:         encodeUint64Value,
		reflect.Uint16:        encodeUint64Value,
		reflect.Uint32:        encodeUint64Value,
		reflect.Uint64:        encodeUint64Value,
		reflect.Float32:       encodeFloat32Value,
		reflect.Float64:       encodeFloat64Value,
		reflect.Complex64:     encodeUnsupportedValue,
		reflect.Complex128:    encodeUnsupportedValue,
		reflect.Array:         encodeArrayValue,
		reflect.Chan:          encodeUnsupportedValue,
		reflect.Func:          encodeUnsupportedValue,
		reflect.Interface:     encodeInterfaceValue,
		reflect.Map:           encodeMapValue,
		reflect.Ptr:           encodeUnsupportedValue,
		reflect.Slice:         encodeSliceValue,
		reflect.String:        encodeStringValue,
		reflect.Struct:        encodeStructValue,
		reflect.UnsafePointer: encodeUnsupportedValue,
	}
	valueDecoders = []decoderFunc{
		reflect.Bool:          decodeBoolValue,
		reflect.Int:           decodeInt64Value,
		reflect.Int8:          decodeInt64Value,
		reflect.Int16:         decodeInt64Value,
		reflect.Int32:         decodeInt64Value,
		reflect.Int64:         decodeInt64Value,
		reflect.Uint:          decodeUint64Value,
		reflect.Uint8:         decodeUint64Value,
		reflect.Uint16:        decodeUint64Value,
		reflect.Uint32:        decodeUint64Value,
		reflect.Uint64:        decodeUint64Value,
		reflect.Float32:       decodeFloat64Value,
		reflect.Float64:       decodeFloat64Value,
		reflect.Complex64:     decodeUnsupportedValue,
		reflect.Complex128:    decodeUnsupportedValue,
		reflect.Array:         decodeArrayValue,
		reflect.Chan:          decodeUnsupportedValue,
		reflect.Func:          decodeUnsupportedValue,
		reflect.Interface:     decodeInterfaceValue,
		reflect.Map:           decodeMapValue,
		reflect.Ptr:           decodeUnsupportedValue,
		reflect.Slice:         decodeSliceValue,
		reflect.String:        decodeStringValue,
		reflect.Struct:        decodeStructValue,
		reflect.UnsafePointer: decodeUnsupportedValue,
	}
}

//------------------------------------------------------------------------------

func encodeUnsupportedValue(e *Encoder, v reflect.Value) error {
	return fmt.Errorf("msgpack: Encode(unsupported %T)", v.Interface())
}

func decodeUnsupportedValue(d *Decoder, v reflect.Value) error {
	return fmt.Errorf("msgpack: Decode(unsupported %T)", v.Interface())
}

//------------------------------------------------------------------------------

func encodeBoolValue(e *Encoder, v reflect.Value) error {
	return e.EncodeBool(v.Bool())
}

func decodeBoolValue(d *Decoder, v reflect.Value) error {
	return d.boolValue(v)
}

//------------------------------------------------------------------------------

func encodeFloat32Value(e *Encoder, v reflect.Value) error {
	return e.EncodeFloat32(float32(v.Float()))
}

func encodeFloat64Value(e *Encoder, v reflect.Value) error {
	return e.EncodeFloat64(v.Float())
}

func decodeFloat64Value(d *Decoder, v reflect.Value) error {
	return d.float64Value(v)
}

//------------------------------------------------------------------------------

func encodeStringValue(e *Encoder, v reflect.Value) error {
	return e.EncodeString(v.String())
}

func decodeStringValue(d *Decoder, v reflect.Value) error {
	return d.stringValue(v)
}

//------------------------------------------------------------------------------

func encodeByteSliceValue(e *Encoder, v reflect.Value) error {
	return e.EncodeBytes(v.Bytes())
}

func encodeByteArrayValue(e *Encoder, v reflect.Value) error {
	if err := e.encodeBytesLen(v.Len()); err != nil {
		return err
	}

	if v.CanAddr() {
		b := v.Slice(0, v.Len()).Bytes()
		return e.write(b)
	}

	b := make([]byte, v.Len())
	reflect.Copy(reflect.ValueOf(b), v)
	return e.write(b)
}

func decodeByteSliceValue(d *Decoder, v reflect.Value) error {
	return d.byteSliceValue(v)
}

func decodeByteArrayValue(d *Decoder, v reflect.Value) error {
	return d.byteArrayValue(v)
}

//------------------------------------------------------------------------------

func encodeInt64Value(e *Encoder, v reflect.Value) error {
	return e.EncodeInt64(v.Int())
}

func decodeInt64Value(d *Decoder, v reflect.Value) error {
	return d.int64Value(v)
}

//------------------------------------------------------------------------------

func encodeUint64Value(e *Encoder, v reflect.Value) error {
	return e.EncodeUint64(v.Uint())
}

func decodeUint64Value(d *Decoder, v reflect.Value) error {
	return d.uint64Value(v)
}

//------------------------------------------------------------------------------

func encodeSliceValue(e *Encoder, v reflect.Value) error {
	return e.encodeSlice(v)
}

func decodeSliceValue(d *Decoder, v reflect.Value) error {
	return d.sliceValue(v)
}

//------------------------------------------------------------------------------

func encodeArrayValue(e *Encoder, v reflect.Value) error {
	return e.encodeArray(v)
}

func decodeArrayValue(d *Decoder, v reflect.Value) error {
	return d.sliceValue(v)
}

//------------------------------------------------------------------------------

func encodeInterfaceValue(e *Encoder, v reflect.Value) error {
	if v.IsNil() {
		return e.EncodeNil()
	}
	return e.EncodeValue(v.Elem())
}

func decodeInterfaceValue(d *Decoder, v reflect.Value) error {
	if v.IsNil() {
		return d.interfaceValue(v)
	}
	return d.DecodeValue(v.Elem())
}

//------------------------------------------------------------------------------

func encodeMapValue(e *Encoder, v reflect.Value) error {
	return e.encodeMap(v)
}

func decodeMapValue(d *Decoder, v reflect.Value) error {
	return d.mapValue(v)
}

//------------------------------------------------------------------------------

func ptrEncoderFunc(typ reflect.Type) encoderFunc {
	encoder := getEncoder(typ.Elem())
	return func(e *Encoder, v reflect.Value) error {
		if v.IsNil() {
			return e.EncodeNil()
		}
		return encoder(e, v.Elem())
	}
}

func ptrDecoderFunc(typ reflect.Type) decoderFunc {
	decoder := getDecoder(typ.Elem())
	return func(d *Decoder, v reflect.Value) error {
		if d.gotNilCode() {
			v.Set(reflect.Zero(v.Type()))
			return d.DecodeNil()
		}
		if v.IsNil() {
			if !v.CanSet() {
				return fmt.Errorf("msgpack: Decode(nonsettable %T)", v.Interface())
			}
			v.Set(reflect.New(v.Type().Elem()))
		}
		return decoder(d, v.Elem())
	}
}

//------------------------------------------------------------------------------

func encodeStructValue(e *Encoder, v reflect.Value) error {
	return e.encodeStruct(v)
}

func decodeStructValue(d *Decoder, v reflect.Value) error {
	return d.structValue(v)
}

//------------------------------------------------------------------------------

func encodeCustomValuePtr(e *Encoder, v reflect.Value) error {
	if !v.CanAddr() {
		return fmt.Errorf("msgpack: Encode(non-addressable %T)", v.Interface())
	}
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		if v.IsNil() {
			return e.EncodeNil()
		}
	}
	encoder := v.Addr().Interface().(CustomEncoder)
	return encoder.EncodeMsgpack(e)
}

func encodeCustomValue(e *Encoder, v reflect.Value) error {
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		if v.IsNil() {
			return e.EncodeNil()
		}
	}
	encoder := v.Interface().(CustomEncoder)
	return encoder.EncodeMsgpack(e)
}

func decodeCustomValuePtr(d *Decoder, v reflect.Value) error {
	if !v.CanAddr() {
		return fmt.Errorf("msgpack: Decode(nonsettable %T)", v.Interface())
	}
	if d.gotNilCode() {
		return d.DecodeNil()
	}
	decoder := v.Addr().Interface().(CustomDecoder)
	return decoder.DecodeMsgpack(d)
}

func decodeCustomValue(d *Decoder, v reflect.Value) error {
	if d.gotNilCode() {
		return d.DecodeNil()
	}
	if v.IsNil() {
		v.Set(reflect.New(v.Type().Elem()))
	}
	decoder := v.Interface().(CustomDecoder)
	return decoder.DecodeMsgpack(d)
}

//------------------------------------------------------------------------------

func marshalValue(e *Encoder, v reflect.Value) error {
	marshaler := v.Interface().(Marshaler)
	b, err := marshaler.MarshalMsgpack()
	if err != nil {
		return err
	}
	_, err = e.w.Write(b)
	return err
}

func unmarshalValue(d *Decoder, v reflect.Value) error {
	if v.IsNil() {
		v.Set(reflect.New(v.Type().Elem()))
	}
	b, err := ioutil.ReadAll(d.r)
	if err != nil {
		return err
	}
	unmarshaler := v.Interface().(Unmarshaler)
	return unmarshaler.UnmarshalMsgpack(b)
}

//------------------------------------------------------------------------------

type structCache struct {
	l sync.RWMutex
	m map[reflect.Type]*fields
}

func newStructCache() *structCache {
	return &structCache{
		m: make(map[reflect.Type]*fields),
	}
}

func (m *structCache) Fields(typ reflect.Type) *fields {
	m.l.RLock()
	fs, ok := m.m[typ]
	m.l.RUnlock()
	if !ok {
		m.l.Lock()
		fs, ok = m.m[typ]
		if !ok {
			fs = getFields(typ)
			m.m[typ] = fs
		}
		m.l.Unlock()
	}

	return fs
}

func getEncoder(typ reflect.Type) encoderFunc {
	enc := _getEncoder(typ)
	if id := extTypeId(typ); id != -1 {
		return makeExtEncoder(id, enc)
	}
	return enc
}

func _getEncoder(typ reflect.Type) encoderFunc {
	kind := typ.Kind()

	if typ.Implements(encoderType) {
		return encodeCustomValue
	}

	// Addressable struct field value.
	if reflect.PtrTo(typ).Implements(encoderType) {
		return encodeCustomValuePtr
	}

	if typ.Implements(marshalerType) {
		return marshalValue
	}
	if encoder, ok := typEncMap[typ]; ok {
		return encoder
	}

	switch kind {
	case reflect.Ptr:
		return ptrEncoderFunc(typ)
	case reflect.Slice:
		if typ.Elem().Kind() == reflect.Uint8 {
			return encodeByteSliceValue
		}
	case reflect.Array:
		if typ.Elem().Kind() == reflect.Uint8 {
			return encodeByteArrayValue
		}
	}
	return valueEncoders[kind]
}

func getDecoder(typ reflect.Type) decoderFunc {
	kind := typ.Kind()

	// Addressable struct field value.
	if kind != reflect.Ptr && reflect.PtrTo(typ).Implements(decoderType) {
		return decodeCustomValuePtr
	}

	if typ.Implements(decoderType) {
		return decodeCustomValue
	}

	if typ.Implements(unmarshalerType) {
		return unmarshalValue
	}

	if decoder, ok := typDecMap[typ]; ok {
		return decoder
	}

	switch kind {
	case reflect.Ptr:
		return ptrDecoderFunc(typ)
	case reflect.Slice:
		if typ.Elem().Kind() == reflect.Uint8 {
			return decodeByteSliceValue
		}
	case reflect.Array:
		if typ.Elem().Kind() == reflect.Uint8 {
			return decodeByteArrayValue
		}
	}
	return valueDecoders[kind]
}
