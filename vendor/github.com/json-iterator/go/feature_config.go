package jsoniter

import (
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"sync/atomic"
	"unsafe"
)

// Config customize how the API should behave.
// The API is created from Config by Froze.
type Config struct {
	IndentionStep                 int
	MarshalFloatWith6Digits       bool
	EscapeHTML                    bool
	SortMapKeys                   bool
	UseNumber                     bool
	TagKey                        string
	ValidateJsonRawMessage        bool
	ObjectFieldMustBeSimpleString bool
}

type frozenConfig struct {
	configBeforeFrozen            Config
	sortMapKeys                   bool
	indentionStep                 int
	objectFieldMustBeSimpleString bool
	decoderCache                  unsafe.Pointer
	encoderCache                  unsafe.Pointer
	extensions                    []Extension
	streamPool                    chan *Stream
	iteratorPool                  chan *Iterator
}

// API the public interface of this package.
// Primary Marshal and Unmarshal.
type API interface {
	IteratorPool
	StreamPool
	MarshalToString(v interface{}) (string, error)
	Marshal(v interface{}) ([]byte, error)
	MarshalIndent(v interface{}, prefix, indent string) ([]byte, error)
	UnmarshalFromString(str string, v interface{}) error
	Unmarshal(data []byte, v interface{}) error
	Get(data []byte, path ...interface{}) Any
	NewEncoder(writer io.Writer) *Encoder
	NewDecoder(reader io.Reader) *Decoder
	Valid(data []byte) bool
	RegisterExtension(extension Extension)
}

// ConfigDefault the default API
var ConfigDefault = Config{
	EscapeHTML: true,
}.Froze()

// ConfigCompatibleWithStandardLibrary tries to be 100% compatible with standard library behavior
var ConfigCompatibleWithStandardLibrary = Config{
	EscapeHTML:             true,
	SortMapKeys:            true,
	ValidateJsonRawMessage: true,
}.Froze()

// ConfigFastest marshals float with only 6 digits precision
var ConfigFastest = Config{
	EscapeHTML:                    false,
	MarshalFloatWith6Digits:       true, // will lose precession
	ObjectFieldMustBeSimpleString: true, // do not unescape object field
}.Froze()

// Froze forge API from config
func (cfg Config) Froze() API {
	// TODO: cache frozen config
	frozenConfig := &frozenConfig{
		sortMapKeys:                   cfg.SortMapKeys,
		indentionStep:                 cfg.IndentionStep,
		objectFieldMustBeSimpleString: cfg.ObjectFieldMustBeSimpleString,
		streamPool:                    make(chan *Stream, 16),
		iteratorPool:                  make(chan *Iterator, 16),
	}
	atomic.StorePointer(&frozenConfig.decoderCache, unsafe.Pointer(&map[string]ValDecoder{}))
	atomic.StorePointer(&frozenConfig.encoderCache, unsafe.Pointer(&map[string]ValEncoder{}))
	if cfg.MarshalFloatWith6Digits {
		frozenConfig.marshalFloatWith6Digits()
	}
	if cfg.EscapeHTML {
		frozenConfig.escapeHTML()
	}
	if cfg.UseNumber {
		frozenConfig.useNumber()
	}
	if cfg.ValidateJsonRawMessage {
		frozenConfig.validateJsonRawMessage()
	}
	frozenConfig.configBeforeFrozen = cfg
	return frozenConfig
}

func (cfg *frozenConfig) validateJsonRawMessage() {
	encoder := &funcEncoder{func(ptr unsafe.Pointer, stream *Stream) {
		rawMessage := *(*json.RawMessage)(ptr)
		iter := cfg.BorrowIterator([]byte(rawMessage))
		iter.Read()
		if iter.Error != nil {
			stream.WriteRaw("null")
		} else {
			cfg.ReturnIterator(iter)
			stream.WriteRaw(string(rawMessage))
		}
	}, func(ptr unsafe.Pointer) bool {
		return false
	}}
	cfg.addEncoderToCache(reflect.TypeOf((*json.RawMessage)(nil)).Elem(), encoder)
	cfg.addEncoderToCache(reflect.TypeOf((*RawMessage)(nil)).Elem(), encoder)
}

func (cfg *frozenConfig) useNumber() {
	cfg.addDecoderToCache(reflect.TypeOf((*interface{})(nil)).Elem(), &funcDecoder{func(ptr unsafe.Pointer, iter *Iterator) {
		if iter.WhatIsNext() == NumberValue {
			*((*interface{})(ptr)) = json.Number(iter.readNumberAsString())
		} else {
			*((*interface{})(ptr)) = iter.Read()
		}
	}})
}
func (cfg *frozenConfig) getTagKey() string {
	tagKey := cfg.configBeforeFrozen.TagKey
	if tagKey == "" {
		return "json"
	}
	return tagKey
}

func (cfg *frozenConfig) RegisterExtension(extension Extension) {
	cfg.extensions = append(cfg.extensions, extension)
}

type lossyFloat32Encoder struct {
}

func (encoder *lossyFloat32Encoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteFloat32Lossy(*((*float32)(ptr)))
}

func (encoder *lossyFloat32Encoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *lossyFloat32Encoder) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*float32)(ptr)) == 0
}

type lossyFloat64Encoder struct {
}

func (encoder *lossyFloat64Encoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteFloat64Lossy(*((*float64)(ptr)))
}

func (encoder *lossyFloat64Encoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *lossyFloat64Encoder) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*float64)(ptr)) == 0
}

// EnableLossyFloatMarshalling keeps 10**(-6) precision
// for float variables for better performance.
func (cfg *frozenConfig) marshalFloatWith6Digits() {
	// for better performance
	cfg.addEncoderToCache(reflect.TypeOf((*float32)(nil)).Elem(), &lossyFloat32Encoder{})
	cfg.addEncoderToCache(reflect.TypeOf((*float64)(nil)).Elem(), &lossyFloat64Encoder{})
}

type htmlEscapedStringEncoder struct {
}

func (encoder *htmlEscapedStringEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	str := *((*string)(ptr))
	stream.WriteStringWithHTMLEscaped(str)
}

func (encoder *htmlEscapedStringEncoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *htmlEscapedStringEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*string)(ptr)) == ""
}

func (cfg *frozenConfig) escapeHTML() {
	cfg.addEncoderToCache(reflect.TypeOf((*string)(nil)).Elem(), &htmlEscapedStringEncoder{})
}

func (cfg *frozenConfig) addDecoderToCache(cacheKey reflect.Type, decoder ValDecoder) {
	done := false
	for !done {
		ptr := atomic.LoadPointer(&cfg.decoderCache)
		cache := *(*map[reflect.Type]ValDecoder)(ptr)
		copied := map[reflect.Type]ValDecoder{}
		for k, v := range cache {
			copied[k] = v
		}
		copied[cacheKey] = decoder
		done = atomic.CompareAndSwapPointer(&cfg.decoderCache, ptr, unsafe.Pointer(&copied))
	}
}

func (cfg *frozenConfig) addEncoderToCache(cacheKey reflect.Type, encoder ValEncoder) {
	done := false
	for !done {
		ptr := atomic.LoadPointer(&cfg.encoderCache)
		cache := *(*map[reflect.Type]ValEncoder)(ptr)
		copied := map[reflect.Type]ValEncoder{}
		for k, v := range cache {
			copied[k] = v
		}
		copied[cacheKey] = encoder
		done = atomic.CompareAndSwapPointer(&cfg.encoderCache, ptr, unsafe.Pointer(&copied))
	}
}

func (cfg *frozenConfig) getDecoderFromCache(cacheKey reflect.Type) ValDecoder {
	ptr := atomic.LoadPointer(&cfg.decoderCache)
	cache := *(*map[reflect.Type]ValDecoder)(ptr)
	return cache[cacheKey]
}

func (cfg *frozenConfig) getEncoderFromCache(cacheKey reflect.Type) ValEncoder {
	ptr := atomic.LoadPointer(&cfg.encoderCache)
	cache := *(*map[reflect.Type]ValEncoder)(ptr)
	return cache[cacheKey]
}

func (cfg *frozenConfig) cleanDecoders() {
	typeDecoders = map[string]ValDecoder{}
	fieldDecoders = map[string]ValDecoder{}
	*cfg = *(cfg.configBeforeFrozen.Froze().(*frozenConfig))
}

func (cfg *frozenConfig) cleanEncoders() {
	typeEncoders = map[string]ValEncoder{}
	fieldEncoders = map[string]ValEncoder{}
	*cfg = *(cfg.configBeforeFrozen.Froze().(*frozenConfig))
}

func (cfg *frozenConfig) MarshalToString(v interface{}) (string, error) {
	stream := cfg.BorrowStream(nil)
	defer cfg.ReturnStream(stream)
	stream.WriteVal(v)
	if stream.Error != nil {
		return "", stream.Error
	}
	return string(stream.Buffer()), nil
}

func (cfg *frozenConfig) Marshal(v interface{}) ([]byte, error) {
	stream := cfg.BorrowStream(nil)
	defer cfg.ReturnStream(stream)
	stream.WriteVal(v)
	if stream.Error != nil {
		return nil, stream.Error
	}
	result := stream.Buffer()
	copied := make([]byte, len(result))
	copy(copied, result)
	return copied, nil
}

func (cfg *frozenConfig) MarshalIndent(v interface{}, prefix, indent string) ([]byte, error) {
	if prefix != "" {
		panic("prefix is not supported")
	}
	for _, r := range indent {
		if r != ' ' {
			panic("indent can only be space")
		}
	}
	newCfg := cfg.configBeforeFrozen
	newCfg.IndentionStep = len(indent)
	return newCfg.Froze().Marshal(v)
}

func (cfg *frozenConfig) UnmarshalFromString(str string, v interface{}) error {
	data := []byte(str)
	data = data[:lastNotSpacePos(data)]
	iter := cfg.BorrowIterator(data)
	defer cfg.ReturnIterator(iter)
	iter.ReadVal(v)
	if iter.head == iter.tail {
		iter.loadMore()
	}
	if iter.Error == io.EOF {
		return nil
	}
	if iter.Error == nil {
		iter.ReportError("UnmarshalFromString", "there are bytes left after unmarshal")
	}
	return iter.Error
}

func (cfg *frozenConfig) Get(data []byte, path ...interface{}) Any {
	iter := cfg.BorrowIterator(data)
	defer cfg.ReturnIterator(iter)
	return locatePath(iter, path)
}

func (cfg *frozenConfig) Unmarshal(data []byte, v interface{}) error {
	data = data[:lastNotSpacePos(data)]
	iter := cfg.BorrowIterator(data)
	defer cfg.ReturnIterator(iter)
	typ := reflect.TypeOf(v)
	if typ.Kind() != reflect.Ptr {
		// return non-pointer error
		return errors.New("the second param must be ptr type")
	}
	iter.ReadVal(v)
	if iter.head == iter.tail {
		iter.loadMore()
	}
	if iter.Error == io.EOF {
		return nil
	}
	if iter.Error == nil {
		iter.ReportError("Unmarshal", "there are bytes left after unmarshal")
	}
	return iter.Error
}

func (cfg *frozenConfig) NewEncoder(writer io.Writer) *Encoder {
	stream := NewStream(cfg, writer, 512)
	return &Encoder{stream}
}

func (cfg *frozenConfig) NewDecoder(reader io.Reader) *Decoder {
	iter := Parse(cfg, reader, 512)
	return &Decoder{iter}
}

func (cfg *frozenConfig) Valid(data []byte) bool {
	iter := cfg.BorrowIterator(data)
	defer cfg.ReturnIterator(iter)
	iter.Skip()
	return iter.Error == nil
}
