package jsoniter

import (
	"encoding/json"
	"io"
	"reflect"
	"sync"
	"unsafe"

	"github.com/modern-go/concurrent"
	"github.com/modern-go/reflect2"
)

// Config customize how the API should behave.
// The API is created from Config by Froze.
type Config struct {
	IndentionStep                 int
	MarshalFloatWith6Digits       bool
	EscapeHTML                    bool
	SortMapKeys                   bool
	UseNumber                     bool
	DisallowUnknownFields         bool
	TagKey                        string
	OnlyTaggedField               bool
	ValidateJsonRawMessage        bool
	ObjectFieldMustBeSimpleString bool
	CaseSensitive                 bool
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
	DecoderOf(typ reflect2.Type) ValDecoder
	EncoderOf(typ reflect2.Type) ValEncoder
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

type frozenConfig struct {
	configBeforeFrozen            Config
	sortMapKeys                   bool
	indentionStep                 int
	objectFieldMustBeSimpleString bool
	onlyTaggedField               bool
	disallowUnknownFields         bool
	decoderCache                  *concurrent.Map
	encoderCache                  *concurrent.Map
	extensions                    []Extension
	streamPool                    *sync.Pool
	iteratorPool                  *sync.Pool
	caseSensitive                 bool
}

func (cfg *frozenConfig) initCache() {
	cfg.decoderCache = concurrent.NewMap()
	cfg.encoderCache = concurrent.NewMap()
}

func (cfg *frozenConfig) addDecoderToCache(cacheKey uintptr, decoder ValDecoder) {
	cfg.decoderCache.Store(cacheKey, decoder)
}

func (cfg *frozenConfig) addEncoderToCache(cacheKey uintptr, encoder ValEncoder) {
	cfg.encoderCache.Store(cacheKey, encoder)
}

func (cfg *frozenConfig) getDecoderFromCache(cacheKey uintptr) ValDecoder {
	decoder, found := cfg.decoderCache.Load(cacheKey)
	if found {
		return decoder.(ValDecoder)
	}
	return nil
}

func (cfg *frozenConfig) getEncoderFromCache(cacheKey uintptr) ValEncoder {
	encoder, found := cfg.encoderCache.Load(cacheKey)
	if found {
		return encoder.(ValEncoder)
	}
	return nil
}

var cfgCache = concurrent.NewMap()

func getFrozenConfigFromCache(cfg Config) *frozenConfig {
	obj, found := cfgCache.Load(cfg)
	if found {
		return obj.(*frozenConfig)
	}
	return nil
}

func addFrozenConfigToCache(cfg Config, frozenConfig *frozenConfig) {
	cfgCache.Store(cfg, frozenConfig)
}

// Froze forge API from config
func (cfg Config) Froze() API {
	api := &frozenConfig{
		sortMapKeys:                   cfg.SortMapKeys,
		indentionStep:                 cfg.IndentionStep,
		objectFieldMustBeSimpleString: cfg.ObjectFieldMustBeSimpleString,
		onlyTaggedField:               cfg.OnlyTaggedField,
		disallowUnknownFields:         cfg.DisallowUnknownFields,
		caseSensitive:                 cfg.CaseSensitive,
	}
	api.streamPool = &sync.Pool{
		New: func() interface{} {
			return NewStream(api, nil, 512)
		},
	}
	api.iteratorPool = &sync.Pool{
		New: func() interface{} {
			return NewIterator(api)
		},
	}
	api.initCache()
	encoderExtension := EncoderExtension{}
	decoderExtension := DecoderExtension{}
	if cfg.MarshalFloatWith6Digits {
		api.marshalFloatWith6Digits(encoderExtension)
	}
	if cfg.EscapeHTML {
		api.escapeHTML(encoderExtension)
	}
	if cfg.UseNumber {
		api.useNumber(decoderExtension)
	}
	if cfg.ValidateJsonRawMessage {
		api.validateJsonRawMessage(encoderExtension)
	}
	if len(encoderExtension) > 0 {
		api.extensions = append(api.extensions, encoderExtension)
	}
	if len(decoderExtension) > 0 {
		api.extensions = append(api.extensions, decoderExtension)
	}
	api.configBeforeFrozen = cfg
	return api
}

func (cfg Config) frozeWithCacheReuse() *frozenConfig {
	api := getFrozenConfigFromCache(cfg)
	if api != nil {
		return api
	}
	api = cfg.Froze().(*frozenConfig)
	addFrozenConfigToCache(cfg, api)
	return api
}

func (cfg *frozenConfig) validateJsonRawMessage(extension EncoderExtension) {
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
	extension[reflect2.TypeOfPtr((*json.RawMessage)(nil)).Elem()] = encoder
	extension[reflect2.TypeOfPtr((*RawMessage)(nil)).Elem()] = encoder
}

func (cfg *frozenConfig) useNumber(extension DecoderExtension) {
	extension[reflect2.TypeOfPtr((*interface{})(nil)).Elem()] = &funcDecoder{func(ptr unsafe.Pointer, iter *Iterator) {
		exitingValue := *((*interface{})(ptr))
		if exitingValue != nil && reflect.TypeOf(exitingValue).Kind() == reflect.Ptr {
			iter.ReadVal(exitingValue)
			return
		}
		if iter.WhatIsNext() == NumberValue {
			*((*interface{})(ptr)) = json.Number(iter.readNumberAsString())
		} else {
			*((*interface{})(ptr)) = iter.Read()
		}
	}}
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

func (encoder *lossyFloat32Encoder) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*float32)(ptr)) == 0
}

type lossyFloat64Encoder struct {
}

func (encoder *lossyFloat64Encoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteFloat64Lossy(*((*float64)(ptr)))
}

func (encoder *lossyFloat64Encoder) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*float64)(ptr)) == 0
}

// EnableLossyFloatMarshalling keeps 10**(-6) precision
// for float variables for better performance.
func (cfg *frozenConfig) marshalFloatWith6Digits(extension EncoderExtension) {
	// for better performance
	extension[reflect2.TypeOfPtr((*float32)(nil)).Elem()] = &lossyFloat32Encoder{}
	extension[reflect2.TypeOfPtr((*float64)(nil)).Elem()] = &lossyFloat64Encoder{}
}

type htmlEscapedStringEncoder struct {
}

func (encoder *htmlEscapedStringEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	str := *((*string)(ptr))
	stream.WriteStringWithHTMLEscaped(str)
}

func (encoder *htmlEscapedStringEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return *((*string)(ptr)) == ""
}

func (cfg *frozenConfig) escapeHTML(encoderExtension EncoderExtension) {
	encoderExtension[reflect2.TypeOfPtr((*string)(nil)).Elem()] = &htmlEscapedStringEncoder{}
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
	return newCfg.frozeWithCacheReuse().Marshal(v)
}

func (cfg *frozenConfig) UnmarshalFromString(str string, v interface{}) error {
	data := []byte(str)
	iter := cfg.BorrowIterator(data)
	defer cfg.ReturnIterator(iter)
	iter.ReadVal(v)
	c := iter.nextToken()
	if c == 0 {
		if iter.Error == io.EOF {
			return nil
		}
		return iter.Error
	}
	iter.ReportError("Unmarshal", "there are bytes left after unmarshal")
	return iter.Error
}

func (cfg *frozenConfig) Get(data []byte, path ...interface{}) Any {
	iter := cfg.BorrowIterator(data)
	defer cfg.ReturnIterator(iter)
	return locatePath(iter, path)
}

func (cfg *frozenConfig) Unmarshal(data []byte, v interface{}) error {
	iter := cfg.BorrowIterator(data)
	defer cfg.ReturnIterator(iter)
	iter.ReadVal(v)
	c := iter.nextToken()
	if c == 0 {
		if iter.Error == io.EOF {
			return nil
		}
		return iter.Error
	}
	iter.ReportError("Unmarshal", "there are bytes left after unmarshal")
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
