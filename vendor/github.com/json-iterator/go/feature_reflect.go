package jsoniter

import (
	"encoding"
	"encoding/json"
	"fmt"
	"reflect"
	"time"
	"unsafe"
)

// ValDecoder is an internal type registered to cache as needed.
// Don't confuse jsoniter.ValDecoder with json.Decoder.
// For json.Decoder's adapter, refer to jsoniter.AdapterDecoder(todo link).
//
// Reflection on type to create decoders, which is then cached
// Reflection on value is avoided as we can, as the reflect.Value itself will allocate, with following exceptions
// 1. create instance of new value, for example *int will need a int to be allocated
// 2. append to slice, if the existing cap is not enough, allocate will be done using Reflect.New
// 3. assignment to map, both key and value will be reflect.Value
// For a simple struct binding, it will be reflect.Value free and allocation free
type ValDecoder interface {
	Decode(ptr unsafe.Pointer, iter *Iterator)
}

// ValEncoder is an internal type registered to cache as needed.
// Don't confuse jsoniter.ValEncoder with json.Encoder.
// For json.Encoder's adapter, refer to jsoniter.AdapterEncoder(todo godoc link).
type ValEncoder interface {
	IsEmpty(ptr unsafe.Pointer) bool
	Encode(ptr unsafe.Pointer, stream *Stream)
	EncodeInterface(val interface{}, stream *Stream)
}

type checkIsEmpty interface {
	IsEmpty(ptr unsafe.Pointer) bool
}

// WriteToStream the default implementation for TypeEncoder method EncodeInterface
func WriteToStream(val interface{}, stream *Stream, encoder ValEncoder) {
	e := (*emptyInterface)(unsafe.Pointer(&val))
	if e.word == nil {
		stream.WriteNil()
		return
	}
	if reflect.TypeOf(val).Kind() == reflect.Ptr {
		encoder.Encode(unsafe.Pointer(&e.word), stream)
	} else {
		encoder.Encode(e.word, stream)
	}
}

var jsonNumberType reflect.Type
var jsoniterNumberType reflect.Type
var jsonRawMessageType reflect.Type
var jsoniterRawMessageType reflect.Type
var anyType reflect.Type
var marshalerType reflect.Type
var unmarshalerType reflect.Type
var textMarshalerType reflect.Type
var textUnmarshalerType reflect.Type

func init() {
	jsonNumberType = reflect.TypeOf((*json.Number)(nil)).Elem()
	jsoniterNumberType = reflect.TypeOf((*Number)(nil)).Elem()
	jsonRawMessageType = reflect.TypeOf((*json.RawMessage)(nil)).Elem()
	jsoniterRawMessageType = reflect.TypeOf((*RawMessage)(nil)).Elem()
	anyType = reflect.TypeOf((*Any)(nil)).Elem()
	marshalerType = reflect.TypeOf((*json.Marshaler)(nil)).Elem()
	unmarshalerType = reflect.TypeOf((*json.Unmarshaler)(nil)).Elem()
	textMarshalerType = reflect.TypeOf((*encoding.TextMarshaler)(nil)).Elem()
	textUnmarshalerType = reflect.TypeOf((*encoding.TextUnmarshaler)(nil)).Elem()
}

type optionalDecoder struct {
	valueType    reflect.Type
	valueDecoder ValDecoder
}

func (decoder *optionalDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if iter.ReadNil() {
		*((*unsafe.Pointer)(ptr)) = nil
	} else {
		if *((*unsafe.Pointer)(ptr)) == nil {
			//pointer to null, we have to allocate memory to hold the value
			value := reflect.New(decoder.valueType)
			newPtr := extractInterface(value.Interface()).word
			decoder.valueDecoder.Decode(newPtr, iter)
			*((*uintptr)(ptr)) = uintptr(newPtr)
		} else {
			//reuse existing instance
			decoder.valueDecoder.Decode(*((*unsafe.Pointer)(ptr)), iter)
		}
	}
}

type deferenceDecoder struct {
	// only to deference a pointer
	valueType    reflect.Type
	valueDecoder ValDecoder
}

func (decoder *deferenceDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	if *((*unsafe.Pointer)(ptr)) == nil {
		//pointer to null, we have to allocate memory to hold the value
		value := reflect.New(decoder.valueType)
		newPtr := extractInterface(value.Interface()).word
		decoder.valueDecoder.Decode(newPtr, iter)
		*((*uintptr)(ptr)) = uintptr(newPtr)
	} else {
		//reuse existing instance
		decoder.valueDecoder.Decode(*((*unsafe.Pointer)(ptr)), iter)
	}
}

type optionalEncoder struct {
	valueEncoder ValEncoder
}

func (encoder *optionalEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	if *((*unsafe.Pointer)(ptr)) == nil {
		stream.WriteNil()
	} else {
		encoder.valueEncoder.Encode(*((*unsafe.Pointer)(ptr)), stream)
	}
}

func (encoder *optionalEncoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *optionalEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	if *((*unsafe.Pointer)(ptr)) == nil {
		return true
	}
	return false
}

type placeholderEncoder struct {
	cfg      *frozenConfig
	cacheKey reflect.Type
}

func (encoder *placeholderEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	encoder.getRealEncoder().Encode(ptr, stream)
}

func (encoder *placeholderEncoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *placeholderEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return encoder.getRealEncoder().IsEmpty(ptr)
}

func (encoder *placeholderEncoder) getRealEncoder() ValEncoder {
	for i := 0; i < 30; i++ {
		realDecoder := encoder.cfg.getEncoderFromCache(encoder.cacheKey)
		_, isPlaceholder := realDecoder.(*placeholderEncoder)
		if isPlaceholder {
			time.Sleep(time.Second)
		} else {
			return realDecoder
		}
	}
	panic(fmt.Sprintf("real encoder not found for cache key: %v", encoder.cacheKey))
}

type placeholderDecoder struct {
	cfg      *frozenConfig
	cacheKey reflect.Type
}

func (decoder *placeholderDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	for i := 0; i < 30; i++ {
		realDecoder := decoder.cfg.getDecoderFromCache(decoder.cacheKey)
		_, isPlaceholder := realDecoder.(*placeholderDecoder)
		if isPlaceholder {
			time.Sleep(time.Second)
		} else {
			realDecoder.Decode(ptr, iter)
			return
		}
	}
	panic(fmt.Sprintf("real decoder not found for cache key: %v", decoder.cacheKey))
}

// emptyInterface is the header for an interface{} value.
type emptyInterface struct {
	typ  unsafe.Pointer
	word unsafe.Pointer
}

// emptyInterface is the header for an interface with method (not interface{})
type nonEmptyInterface struct {
	// see ../runtime/iface.go:/Itab
	itab *struct {
		ityp   unsafe.Pointer // static interface type
		typ    unsafe.Pointer // dynamic concrete type
		link   unsafe.Pointer
		bad    int32
		unused int32
		fun    [100000]unsafe.Pointer // method table
	}
	word unsafe.Pointer
}

// ReadVal copy the underlying JSON into go interface, same as json.Unmarshal
func (iter *Iterator) ReadVal(obj interface{}) {
	typ := reflect.TypeOf(obj)
	cacheKey := typ.Elem()
	decoder, err := decoderOfType(iter.cfg, cacheKey)
	if err != nil {
		iter.Error = err
		return
	}
	e := (*emptyInterface)(unsafe.Pointer(&obj))
	decoder.Decode(e.word, iter)
}

// WriteVal copy the go interface into underlying JSON, same as json.Marshal
func (stream *Stream) WriteVal(val interface{}) {
	if nil == val {
		stream.WriteNil()
		return
	}
	typ := reflect.TypeOf(val)
	cacheKey := typ
	encoder, err := encoderOfType(stream.cfg, cacheKey)
	if err != nil {
		stream.Error = err
		return
	}
	encoder.EncodeInterface(val, stream)
}

type prefix string

func (p prefix) addToDecoder(decoder ValDecoder, err error) (ValDecoder, error) {
	if err != nil {
		return nil, fmt.Errorf("%s: %s", p, err.Error())
	}
	return decoder, err
}

func (p prefix) addToEncoder(encoder ValEncoder, err error) (ValEncoder, error) {
	if err != nil {
		return nil, fmt.Errorf("%s: %s", p, err.Error())
	}
	return encoder, err
}

func decoderOfType(cfg *frozenConfig, typ reflect.Type) (ValDecoder, error) {
	cacheKey := typ
	decoder := cfg.getDecoderFromCache(cacheKey)
	if decoder != nil {
		return decoder, nil
	}
	decoder = getTypeDecoderFromExtension(typ)
	if decoder != nil {
		cfg.addDecoderToCache(cacheKey, decoder)
		return decoder, nil
	}
	decoder = &placeholderDecoder{cfg: cfg, cacheKey: cacheKey}
	cfg.addDecoderToCache(cacheKey, decoder)
	decoder, err := createDecoderOfType(cfg, typ)
	for _, extension := range extensions {
		decoder = extension.DecorateDecoder(typ, decoder)
	}
	cfg.addDecoderToCache(cacheKey, decoder)
	return decoder, err
}

func createDecoderOfType(cfg *frozenConfig, typ reflect.Type) (ValDecoder, error) {
	typeName := typ.String()
	if typ == jsonRawMessageType {
		return &jsonRawMessageCodec{}, nil
	}
	if typ == jsoniterRawMessageType {
		return &jsoniterRawMessageCodec{}, nil
	}
	if typ.AssignableTo(jsonNumberType) {
		return &jsonNumberCodec{}, nil
	}
	if typ.AssignableTo(jsoniterNumberType) {
		return &jsoniterNumberCodec{}, nil
	}
	if typ.Implements(unmarshalerType) {
		templateInterface := reflect.New(typ).Elem().Interface()
		var decoder ValDecoder = &unmarshalerDecoder{extractInterface(templateInterface)}
		if typ.Kind() == reflect.Ptr {
			decoder = &optionalDecoder{typ.Elem(), decoder}
		}
		return decoder, nil
	}
	if reflect.PtrTo(typ).Implements(unmarshalerType) {
		templateInterface := reflect.New(typ).Interface()
		var decoder ValDecoder = &unmarshalerDecoder{extractInterface(templateInterface)}
		return decoder, nil
	}
	if typ.Implements(textUnmarshalerType) {
		templateInterface := reflect.New(typ).Elem().Interface()
		var decoder ValDecoder = &textUnmarshalerDecoder{extractInterface(templateInterface)}
		if typ.Kind() == reflect.Ptr {
			decoder = &optionalDecoder{typ.Elem(), decoder}
		}
		return decoder, nil
	}
	if reflect.PtrTo(typ).Implements(textUnmarshalerType) {
		templateInterface := reflect.New(typ).Interface()
		var decoder ValDecoder = &textUnmarshalerDecoder{extractInterface(templateInterface)}
		return decoder, nil
	}
	if typ.Kind() == reflect.Slice && typ.Elem().Kind() == reflect.Uint8 {
		sliceDecoder, err := prefix("[slice]").addToDecoder(decoderOfSlice(cfg, typ))
		if err != nil {
			return nil, err
		}
		return &base64Codec{sliceDecoder: sliceDecoder}, nil
	}
	if typ.Implements(anyType) {
		return &anyCodec{}, nil
	}
	switch typ.Kind() {
	case reflect.String:
		if typeName != "string" {
			return decoderOfType(cfg, reflect.TypeOf((*string)(nil)).Elem())
		}
		return &stringCodec{}, nil
	case reflect.Int:
		if typeName != "int" {
			return decoderOfType(cfg, reflect.TypeOf((*int)(nil)).Elem())
		}
		return &intCodec{}, nil
	case reflect.Int8:
		if typeName != "int8" {
			return decoderOfType(cfg, reflect.TypeOf((*int8)(nil)).Elem())
		}
		return &int8Codec{}, nil
	case reflect.Int16:
		if typeName != "int16" {
			return decoderOfType(cfg, reflect.TypeOf((*int16)(nil)).Elem())
		}
		return &int16Codec{}, nil
	case reflect.Int32:
		if typeName != "int32" {
			return decoderOfType(cfg, reflect.TypeOf((*int32)(nil)).Elem())
		}
		return &int32Codec{}, nil
	case reflect.Int64:
		if typeName != "int64" {
			return decoderOfType(cfg, reflect.TypeOf((*int64)(nil)).Elem())
		}
		return &int64Codec{}, nil
	case reflect.Uint:
		if typeName != "uint" {
			return decoderOfType(cfg, reflect.TypeOf((*uint)(nil)).Elem())
		}
		return &uintCodec{}, nil
	case reflect.Uint8:
		if typeName != "uint8" {
			return decoderOfType(cfg, reflect.TypeOf((*uint8)(nil)).Elem())
		}
		return &uint8Codec{}, nil
	case reflect.Uint16:
		if typeName != "uint16" {
			return decoderOfType(cfg, reflect.TypeOf((*uint16)(nil)).Elem())
		}
		return &uint16Codec{}, nil
	case reflect.Uint32:
		if typeName != "uint32" {
			return decoderOfType(cfg, reflect.TypeOf((*uint32)(nil)).Elem())
		}
		return &uint32Codec{}, nil
	case reflect.Uintptr:
		if typeName != "uintptr" {
			return decoderOfType(cfg, reflect.TypeOf((*uintptr)(nil)).Elem())
		}
		return &uintptrCodec{}, nil
	case reflect.Uint64:
		if typeName != "uint64" {
			return decoderOfType(cfg, reflect.TypeOf((*uint64)(nil)).Elem())
		}
		return &uint64Codec{}, nil
	case reflect.Float32:
		if typeName != "float32" {
			return decoderOfType(cfg, reflect.TypeOf((*float32)(nil)).Elem())
		}
		return &float32Codec{}, nil
	case reflect.Float64:
		if typeName != "float64" {
			return decoderOfType(cfg, reflect.TypeOf((*float64)(nil)).Elem())
		}
		return &float64Codec{}, nil
	case reflect.Bool:
		if typeName != "bool" {
			return decoderOfType(cfg, reflect.TypeOf((*bool)(nil)).Elem())
		}
		return &boolCodec{}, nil
	case reflect.Interface:
		if typ.NumMethod() == 0 {
			return &emptyInterfaceCodec{}, nil
		}
		return &nonEmptyInterfaceCodec{}, nil
	case reflect.Struct:
		return prefix(fmt.Sprintf("[%s]", typeName)).addToDecoder(decoderOfStruct(cfg, typ))
	case reflect.Array:
		return prefix("[array]").addToDecoder(decoderOfArray(cfg, typ))
	case reflect.Slice:
		return prefix("[slice]").addToDecoder(decoderOfSlice(cfg, typ))
	case reflect.Map:
		return prefix("[map]").addToDecoder(decoderOfMap(cfg, typ))
	case reflect.Ptr:
		return prefix("[optional]").addToDecoder(decoderOfOptional(cfg, typ))
	default:
		return nil, fmt.Errorf("unsupported type: %v", typ)
	}
}

func encoderOfType(cfg *frozenConfig, typ reflect.Type) (ValEncoder, error) {
	cacheKey := typ
	encoder := cfg.getEncoderFromCache(cacheKey)
	if encoder != nil {
		return encoder, nil
	}
	encoder = getTypeEncoderFromExtension(typ)
	if encoder != nil {
		cfg.addEncoderToCache(cacheKey, encoder)
		return encoder, nil
	}
	encoder = &placeholderEncoder{cfg: cfg, cacheKey: cacheKey}
	cfg.addEncoderToCache(cacheKey, encoder)
	encoder, err := createEncoderOfType(cfg, typ)
	for _, extension := range extensions {
		encoder = extension.DecorateEncoder(typ, encoder)
	}
	cfg.addEncoderToCache(cacheKey, encoder)
	return encoder, err
}

func createEncoderOfType(cfg *frozenConfig, typ reflect.Type) (ValEncoder, error) {
	if typ == jsonRawMessageType {
		return &jsonRawMessageCodec{}, nil
	}
	if typ == jsoniterRawMessageType {
		return &jsoniterRawMessageCodec{}, nil
	}
	if typ.AssignableTo(jsonNumberType) {
		return &jsonNumberCodec{}, nil
	}
	if typ.AssignableTo(jsoniterNumberType) {
		return &jsoniterNumberCodec{}, nil
	}
	if typ.Implements(marshalerType) {
		checkIsEmpty, err := createCheckIsEmpty(typ)
		if err != nil {
			return nil, err
		}
		templateInterface := reflect.New(typ).Elem().Interface()
		var encoder ValEncoder = &marshalerEncoder{
			templateInterface: extractInterface(templateInterface),
			checkIsEmpty:      checkIsEmpty,
		}
		if typ.Kind() == reflect.Ptr {
			encoder = &optionalEncoder{encoder}
		}
		return encoder, nil
	}
	if typ.Implements(textMarshalerType) {
		checkIsEmpty, err := createCheckIsEmpty(typ)
		if err != nil {
			return nil, err
		}
		templateInterface := reflect.New(typ).Elem().Interface()
		var encoder ValEncoder = &textMarshalerEncoder{
			templateInterface: extractInterface(templateInterface),
			checkIsEmpty:      checkIsEmpty,
		}
		if typ.Kind() == reflect.Ptr {
			encoder = &optionalEncoder{encoder}
		}
		return encoder, nil
	}
	if typ.Kind() == reflect.Slice && typ.Elem().Kind() == reflect.Uint8 {
		return &base64Codec{}, nil
	}
	if typ.Implements(anyType) {
		return &anyCodec{}, nil
	}
	return createEncoderOfSimpleType(cfg, typ)
}

func createCheckIsEmpty(typ reflect.Type) (checkIsEmpty, error) {
	kind := typ.Kind()
	switch kind {
	case reflect.String:
		return &stringCodec{}, nil
	case reflect.Int:
		return &intCodec{}, nil
	case reflect.Int8:
		return &int8Codec{}, nil
	case reflect.Int16:
		return &int16Codec{}, nil
	case reflect.Int32:
		return &int32Codec{}, nil
	case reflect.Int64:
		return &int64Codec{}, nil
	case reflect.Uint:
		return &uintCodec{}, nil
	case reflect.Uint8:
		return &uint8Codec{}, nil
	case reflect.Uint16:
		return &uint16Codec{}, nil
	case reflect.Uint32:
		return &uint32Codec{}, nil
	case reflect.Uintptr:
		return &uintptrCodec{}, nil
	case reflect.Uint64:
		return &uint64Codec{}, nil
	case reflect.Float32:
		return &float32Codec{}, nil
	case reflect.Float64:
		return &float64Codec{}, nil
	case reflect.Bool:
		return &boolCodec{}, nil
	case reflect.Interface:
		if typ.NumMethod() == 0 {
			return &emptyInterfaceCodec{}, nil
		}
		return &nonEmptyInterfaceCodec{}, nil
	case reflect.Struct:
		return &structEncoder{}, nil
	case reflect.Array:
		return &arrayEncoder{}, nil
	case reflect.Slice:
		return &sliceEncoder{}, nil
	case reflect.Map:
		return &mapEncoder{}, nil
	case reflect.Ptr:
		return &optionalEncoder{}, nil
	default:
		return nil, fmt.Errorf("unsupported type: %v", typ)
	}
}

func createEncoderOfSimpleType(cfg *frozenConfig, typ reflect.Type) (ValEncoder, error) {
	typeName := typ.String()
	kind := typ.Kind()
	switch kind {
	case reflect.String:
		if typeName != "string" {
			return encoderOfType(cfg, reflect.TypeOf((*string)(nil)).Elem())
		}
		return &stringCodec{}, nil
	case reflect.Int:
		if typeName != "int" {
			return encoderOfType(cfg, reflect.TypeOf((*int)(nil)).Elem())
		}
		return &intCodec{}, nil
	case reflect.Int8:
		if typeName != "int8" {
			return encoderOfType(cfg, reflect.TypeOf((*int8)(nil)).Elem())
		}
		return &int8Codec{}, nil
	case reflect.Int16:
		if typeName != "int16" {
			return encoderOfType(cfg, reflect.TypeOf((*int16)(nil)).Elem())
		}
		return &int16Codec{}, nil
	case reflect.Int32:
		if typeName != "int32" {
			return encoderOfType(cfg, reflect.TypeOf((*int32)(nil)).Elem())
		}
		return &int32Codec{}, nil
	case reflect.Int64:
		if typeName != "int64" {
			return encoderOfType(cfg, reflect.TypeOf((*int64)(nil)).Elem())
		}
		return &int64Codec{}, nil
	case reflect.Uint:
		if typeName != "uint" {
			return encoderOfType(cfg, reflect.TypeOf((*uint)(nil)).Elem())
		}
		return &uintCodec{}, nil
	case reflect.Uint8:
		if typeName != "uint8" {
			return encoderOfType(cfg, reflect.TypeOf((*uint8)(nil)).Elem())
		}
		return &uint8Codec{}, nil
	case reflect.Uint16:
		if typeName != "uint16" {
			return encoderOfType(cfg, reflect.TypeOf((*uint16)(nil)).Elem())
		}
		return &uint16Codec{}, nil
	case reflect.Uint32:
		if typeName != "uint32" {
			return encoderOfType(cfg, reflect.TypeOf((*uint32)(nil)).Elem())
		}
		return &uint32Codec{}, nil
	case reflect.Uintptr:
		if typeName != "uintptr" {
			return encoderOfType(cfg, reflect.TypeOf((*uintptr)(nil)).Elem())
		}
		return &uintptrCodec{}, nil
	case reflect.Uint64:
		if typeName != "uint64" {
			return encoderOfType(cfg, reflect.TypeOf((*uint64)(nil)).Elem())
		}
		return &uint64Codec{}, nil
	case reflect.Float32:
		if typeName != "float32" {
			return encoderOfType(cfg, reflect.TypeOf((*float32)(nil)).Elem())
		}
		return &float32Codec{}, nil
	case reflect.Float64:
		if typeName != "float64" {
			return encoderOfType(cfg, reflect.TypeOf((*float64)(nil)).Elem())
		}
		return &float64Codec{}, nil
	case reflect.Bool:
		if typeName != "bool" {
			return encoderOfType(cfg, reflect.TypeOf((*bool)(nil)).Elem())
		}
		return &boolCodec{}, nil
	case reflect.Interface:
		if typ.NumMethod() == 0 {
			return &emptyInterfaceCodec{}, nil
		}
		return &nonEmptyInterfaceCodec{}, nil
	case reflect.Struct:
		return prefix(fmt.Sprintf("[%s]", typeName)).addToEncoder(encoderOfStruct(cfg, typ))
	case reflect.Array:
		return prefix("[array]").addToEncoder(encoderOfArray(cfg, typ))
	case reflect.Slice:
		return prefix("[slice]").addToEncoder(encoderOfSlice(cfg, typ))
	case reflect.Map:
		return prefix("[map]").addToEncoder(encoderOfMap(cfg, typ))
	case reflect.Ptr:
		return prefix("[optional]").addToEncoder(encoderOfOptional(cfg, typ))
	default:
		return nil, fmt.Errorf("unsupported type: %v", typ)
	}
}

func decoderOfOptional(cfg *frozenConfig, typ reflect.Type) (ValDecoder, error) {
	elemType := typ.Elem()
	decoder, err := decoderOfType(cfg, elemType)
	if err != nil {
		return nil, err
	}
	return &optionalDecoder{elemType, decoder}, nil
}

func encoderOfOptional(cfg *frozenConfig, typ reflect.Type) (ValEncoder, error) {
	elemType := typ.Elem()
	elemEncoder, err := encoderOfType(cfg, elemType)
	if err != nil {
		return nil, err
	}
	encoder := &optionalEncoder{elemEncoder}
	if elemType.Kind() == reflect.Map {
		encoder = &optionalEncoder{encoder}
	}
	return encoder, nil
}

func decoderOfMap(cfg *frozenConfig, typ reflect.Type) (ValDecoder, error) {
	decoder, err := decoderOfType(cfg, typ.Elem())
	if err != nil {
		return nil, err
	}
	mapInterface := reflect.New(typ).Interface()
	return &mapDecoder{typ, typ.Key(), typ.Elem(), decoder, extractInterface(mapInterface)}, nil
}

func extractInterface(val interface{}) emptyInterface {
	return *((*emptyInterface)(unsafe.Pointer(&val)))
}

func encoderOfMap(cfg *frozenConfig, typ reflect.Type) (ValEncoder, error) {
	elemType := typ.Elem()
	encoder, err := encoderOfType(cfg, elemType)
	if err != nil {
		return nil, err
	}
	mapInterface := reflect.New(typ).Elem().Interface()
	if cfg.sortMapKeys {
		return &sortKeysMapEncoder{typ, elemType, encoder, *((*emptyInterface)(unsafe.Pointer(&mapInterface)))}, nil
	}
	return &mapEncoder{typ, elemType, encoder, *((*emptyInterface)(unsafe.Pointer(&mapInterface)))}, nil
}
