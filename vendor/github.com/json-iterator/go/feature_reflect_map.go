package jsoniter

import (
	"encoding"
	"encoding/json"
	"reflect"
	"sort"
	"strconv"
	"unsafe"
)

type mapDecoder struct {
	mapType      reflect.Type
	keyType      reflect.Type
	elemType     reflect.Type
	elemDecoder  ValDecoder
	mapInterface emptyInterface
}

func (decoder *mapDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	// dark magic to cast unsafe.Pointer back to interface{} using reflect.Type
	mapInterface := decoder.mapInterface
	mapInterface.word = ptr
	realInterface := (*interface{})(unsafe.Pointer(&mapInterface))
	realVal := reflect.ValueOf(*realInterface).Elem()
	if iter.ReadNil() {
		realVal.Set(reflect.Zero(decoder.mapType))
		return
	}
	if realVal.IsNil() {
		realVal.Set(reflect.MakeMap(realVal.Type()))
	}
	iter.ReadMapCB(func(iter *Iterator, keyStr string) bool {
		elem := reflect.New(decoder.elemType)
		decoder.elemDecoder.Decode(unsafe.Pointer(elem.Pointer()), iter)
		// to put into map, we have to use reflection
		keyType := decoder.keyType
		// TODO: remove this from loop
		switch {
		case keyType.Kind() == reflect.String:
			realVal.SetMapIndex(reflect.ValueOf(keyStr).Convert(keyType), elem.Elem())
			return true
		case keyType.Implements(textUnmarshalerType):
			textUnmarshaler := reflect.New(keyType.Elem()).Interface().(encoding.TextUnmarshaler)
			err := textUnmarshaler.UnmarshalText([]byte(keyStr))
			if err != nil {
				iter.ReportError("read map key as TextUnmarshaler", err.Error())
				return false
			}
			realVal.SetMapIndex(reflect.ValueOf(textUnmarshaler), elem.Elem())
			return true
		case reflect.PtrTo(keyType).Implements(textUnmarshalerType):
			textUnmarshaler := reflect.New(keyType).Interface().(encoding.TextUnmarshaler)
			err := textUnmarshaler.UnmarshalText([]byte(keyStr))
			if err != nil {
				iter.ReportError("read map key as TextUnmarshaler", err.Error())
				return false
			}
			realVal.SetMapIndex(reflect.ValueOf(textUnmarshaler).Elem(), elem.Elem())
			return true
		default:
			switch keyType.Kind() {
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				n, err := strconv.ParseInt(keyStr, 10, 64)
				if err != nil || reflect.Zero(keyType).OverflowInt(n) {
					iter.ReportError("read map key as int64", "read int64 failed")
					return false
				}
				realVal.SetMapIndex(reflect.ValueOf(n).Convert(keyType), elem.Elem())
				return true
			case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
				n, err := strconv.ParseUint(keyStr, 10, 64)
				if err != nil || reflect.Zero(keyType).OverflowUint(n) {
					iter.ReportError("read map key as uint64", "read uint64 failed")
					return false
				}
				realVal.SetMapIndex(reflect.ValueOf(n).Convert(keyType), elem.Elem())
				return true
			}
		}
		iter.ReportError("read map key", "unexpected map key type "+keyType.String())
		return true
	})
}

type mapEncoder struct {
	mapType      reflect.Type
	elemType     reflect.Type
	elemEncoder  ValEncoder
	mapInterface emptyInterface
}

func (encoder *mapEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	mapInterface := encoder.mapInterface
	mapInterface.word = ptr
	realInterface := (*interface{})(unsafe.Pointer(&mapInterface))
	realVal := reflect.ValueOf(*realInterface)
	stream.WriteObjectStart()
	for i, key := range realVal.MapKeys() {
		if i != 0 {
			stream.WriteMore()
		}
		encodeMapKey(key, stream)
		if stream.indention > 0 {
			stream.writeTwoBytes(byte(':'), byte(' '))
		} else {
			stream.writeByte(':')
		}
		val := realVal.MapIndex(key).Interface()
		encoder.elemEncoder.EncodeInterface(val, stream)
	}
	stream.WriteObjectEnd()
}

func encodeMapKey(key reflect.Value, stream *Stream) {
	if key.Kind() == reflect.String {
		stream.WriteString(key.String())
		return
	}
	if tm, ok := key.Interface().(encoding.TextMarshaler); ok {
		buf, err := tm.MarshalText()
		if err != nil {
			stream.Error = err
			return
		}
		stream.writeByte('"')
		stream.Write(buf)
		stream.writeByte('"')
		return
	}
	switch key.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		stream.writeByte('"')
		stream.WriteInt64(key.Int())
		stream.writeByte('"')
		return
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		stream.writeByte('"')
		stream.WriteUint64(key.Uint())
		stream.writeByte('"')
		return
	}
	stream.Error = &json.UnsupportedTypeError{Type: key.Type()}
}

func (encoder *mapEncoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *mapEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	mapInterface := encoder.mapInterface
	mapInterface.word = ptr
	realInterface := (*interface{})(unsafe.Pointer(&mapInterface))
	realVal := reflect.ValueOf(*realInterface)
	return realVal.Len() == 0
}

type sortKeysMapEncoder struct {
	mapType      reflect.Type
	elemType     reflect.Type
	elemEncoder  ValEncoder
	mapInterface emptyInterface
}

func (encoder *sortKeysMapEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	mapInterface := encoder.mapInterface
	mapInterface.word = ptr
	realInterface := (*interface{})(unsafe.Pointer(&mapInterface))
	realVal := reflect.ValueOf(*realInterface)

	// Extract and sort the keys.
	keys := realVal.MapKeys()
	sv := stringValues(make([]reflectWithString, len(keys)))
	for i, v := range keys {
		sv[i].v = v
		if err := sv[i].resolve(); err != nil {
			stream.Error = err
			return
		}
	}
	sort.Sort(sv)

	stream.WriteObjectStart()
	for i, key := range sv {
		if i != 0 {
			stream.WriteMore()
		}
		stream.WriteVal(key.s) // might need html escape, so can not WriteString directly
		if stream.indention > 0 {
			stream.writeTwoBytes(byte(':'), byte(' '))
		} else {
			stream.writeByte(':')
		}
		val := realVal.MapIndex(key.v).Interface()
		encoder.elemEncoder.EncodeInterface(val, stream)
	}
	stream.WriteObjectEnd()
}

// stringValues is a slice of reflect.Value holding *reflect.StringValue.
// It implements the methods to sort by string.
type stringValues []reflectWithString

type reflectWithString struct {
	v reflect.Value
	s string
}

func (w *reflectWithString) resolve() error {
	if w.v.Kind() == reflect.String {
		w.s = w.v.String()
		return nil
	}
	if tm, ok := w.v.Interface().(encoding.TextMarshaler); ok {
		buf, err := tm.MarshalText()
		w.s = string(buf)
		return err
	}
	switch w.v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		w.s = strconv.FormatInt(w.v.Int(), 10)
		return nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		w.s = strconv.FormatUint(w.v.Uint(), 10)
		return nil
	}
	return &json.UnsupportedTypeError{Type: w.v.Type()}
}

func (sv stringValues) Len() int           { return len(sv) }
func (sv stringValues) Swap(i, j int)      { sv[i], sv[j] = sv[j], sv[i] }
func (sv stringValues) Less(i, j int) bool { return sv[i].s < sv[j].s }

func (encoder *sortKeysMapEncoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *sortKeysMapEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	mapInterface := encoder.mapInterface
	mapInterface.word = ptr
	realInterface := (*interface{})(unsafe.Pointer(&mapInterface))
	realVal := reflect.ValueOf(*realInterface)
	return realVal.Len() == 0
}
