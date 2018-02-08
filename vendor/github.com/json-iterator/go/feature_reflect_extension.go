package jsoniter

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
	"unicode"
	"unsafe"
)

var typeDecoders = map[string]ValDecoder{}
var fieldDecoders = map[string]ValDecoder{}
var typeEncoders = map[string]ValEncoder{}
var fieldEncoders = map[string]ValEncoder{}
var extensions = []Extension{}

// StructDescriptor describe how should we encode/decode the struct
type StructDescriptor struct {
	onePtrEmbedded     bool
	onePtrOptimization bool
	Type               reflect.Type
	Fields             []*Binding
}

// GetField get one field from the descriptor by its name.
// Can not use map here to keep field orders.
func (structDescriptor *StructDescriptor) GetField(fieldName string) *Binding {
	for _, binding := range structDescriptor.Fields {
		if binding.Field.Name == fieldName {
			return binding
		}
	}
	return nil
}

// Binding describe how should we encode/decode the struct field
type Binding struct {
	levels    []int
	Field     *reflect.StructField
	FromNames []string
	ToNames   []string
	Encoder   ValEncoder
	Decoder   ValDecoder
}

// Extension the one for all SPI. Customize encoding/decoding by specifying alternate encoder/decoder.
// Can also rename fields by UpdateStructDescriptor.
type Extension interface {
	UpdateStructDescriptor(structDescriptor *StructDescriptor)
	CreateDecoder(typ reflect.Type) ValDecoder
	CreateEncoder(typ reflect.Type) ValEncoder
	DecorateDecoder(typ reflect.Type, decoder ValDecoder) ValDecoder
	DecorateEncoder(typ reflect.Type, encoder ValEncoder) ValEncoder
}

// DummyExtension embed this type get dummy implementation for all methods of Extension
type DummyExtension struct {
}

// UpdateStructDescriptor No-op
func (extension *DummyExtension) UpdateStructDescriptor(structDescriptor *StructDescriptor) {
}

// CreateDecoder No-op
func (extension *DummyExtension) CreateDecoder(typ reflect.Type) ValDecoder {
	return nil
}

// CreateEncoder No-op
func (extension *DummyExtension) CreateEncoder(typ reflect.Type) ValEncoder {
	return nil
}

// DecorateDecoder No-op
func (extension *DummyExtension) DecorateDecoder(typ reflect.Type, decoder ValDecoder) ValDecoder {
	return decoder
}

// DecorateEncoder No-op
func (extension *DummyExtension) DecorateEncoder(typ reflect.Type, encoder ValEncoder) ValEncoder {
	return encoder
}

type funcDecoder struct {
	fun DecoderFunc
}

func (decoder *funcDecoder) Decode(ptr unsafe.Pointer, iter *Iterator) {
	decoder.fun(ptr, iter)
}

type funcEncoder struct {
	fun         EncoderFunc
	isEmptyFunc func(ptr unsafe.Pointer) bool
}

func (encoder *funcEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	encoder.fun(ptr, stream)
}

func (encoder *funcEncoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *funcEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	if encoder.isEmptyFunc == nil {
		return false
	}
	return encoder.isEmptyFunc(ptr)
}

// DecoderFunc the function form of TypeDecoder
type DecoderFunc func(ptr unsafe.Pointer, iter *Iterator)

// EncoderFunc the function form of TypeEncoder
type EncoderFunc func(ptr unsafe.Pointer, stream *Stream)

// RegisterTypeDecoderFunc register TypeDecoder for a type with function
func RegisterTypeDecoderFunc(typ string, fun DecoderFunc) {
	typeDecoders[typ] = &funcDecoder{fun}
}

// RegisterTypeDecoder register TypeDecoder for a typ
func RegisterTypeDecoder(typ string, decoder ValDecoder) {
	typeDecoders[typ] = decoder
}

// RegisterFieldDecoderFunc register TypeDecoder for a struct field with function
func RegisterFieldDecoderFunc(typ string, field string, fun DecoderFunc) {
	RegisterFieldDecoder(typ, field, &funcDecoder{fun})
}

// RegisterFieldDecoder register TypeDecoder for a struct field
func RegisterFieldDecoder(typ string, field string, decoder ValDecoder) {
	fieldDecoders[fmt.Sprintf("%s/%s", typ, field)] = decoder
}

// RegisterTypeEncoderFunc register TypeEncoder for a type with encode/isEmpty function
func RegisterTypeEncoderFunc(typ string, fun EncoderFunc, isEmptyFunc func(unsafe.Pointer) bool) {
	typeEncoders[typ] = &funcEncoder{fun, isEmptyFunc}
}

// RegisterTypeEncoder register TypeEncoder for a type
func RegisterTypeEncoder(typ string, encoder ValEncoder) {
	typeEncoders[typ] = encoder
}

// RegisterFieldEncoderFunc register TypeEncoder for a struct field with encode/isEmpty function
func RegisterFieldEncoderFunc(typ string, field string, fun EncoderFunc, isEmptyFunc func(unsafe.Pointer) bool) {
	RegisterFieldEncoder(typ, field, &funcEncoder{fun, isEmptyFunc})
}

// RegisterFieldEncoder register TypeEncoder for a struct field
func RegisterFieldEncoder(typ string, field string, encoder ValEncoder) {
	fieldEncoders[fmt.Sprintf("%s/%s", typ, field)] = encoder
}

// RegisterExtension register extension
func RegisterExtension(extension Extension) {
	extensions = append(extensions, extension)
}

func getTypeDecoderFromExtension(cfg *frozenConfig, typ reflect.Type) ValDecoder {
	decoder := _getTypeDecoderFromExtension(cfg, typ)
	if decoder != nil {
		for _, extension := range extensions {
			decoder = extension.DecorateDecoder(typ, decoder)
		}
		for _, extension := range cfg.extensions {
			decoder = extension.DecorateDecoder(typ, decoder)
		}
	}
	return decoder
}
func _getTypeDecoderFromExtension(cfg *frozenConfig, typ reflect.Type) ValDecoder {
	for _, extension := range extensions {
		decoder := extension.CreateDecoder(typ)
		if decoder != nil {
			return decoder
		}
	}
	for _, extension := range cfg.extensions {
		decoder := extension.CreateDecoder(typ)
		if decoder != nil {
			return decoder
		}
	}
	typeName := typ.String()
	decoder := typeDecoders[typeName]
	if decoder != nil {
		return decoder
	}
	if typ.Kind() == reflect.Ptr {
		decoder := typeDecoders[typ.Elem().String()]
		if decoder != nil {
			return &OptionalDecoder{typ.Elem(), decoder}
		}
	}
	return nil
}

func getTypeEncoderFromExtension(cfg *frozenConfig, typ reflect.Type) ValEncoder {
	encoder := _getTypeEncoderFromExtension(cfg, typ)
	if encoder != nil {
		for _, extension := range extensions {
			encoder = extension.DecorateEncoder(typ, encoder)
		}
		for _, extension := range cfg.extensions {
			encoder = extension.DecorateEncoder(typ, encoder)
		}
	}
	return encoder
}

func _getTypeEncoderFromExtension(cfg *frozenConfig, typ reflect.Type) ValEncoder {
	for _, extension := range extensions {
		encoder := extension.CreateEncoder(typ)
		if encoder != nil {
			return encoder
		}
	}
	for _, extension := range cfg.extensions {
		encoder := extension.CreateEncoder(typ)
		if encoder != nil {
			return encoder
		}
	}
	typeName := typ.String()
	encoder := typeEncoders[typeName]
	if encoder != nil {
		return encoder
	}
	if typ.Kind() == reflect.Ptr {
		encoder := typeEncoders[typ.Elem().String()]
		if encoder != nil {
			return &OptionalEncoder{encoder}
		}
	}
	return nil
}

func describeStruct(cfg *frozenConfig, typ reflect.Type) (*StructDescriptor, error) {
	embeddedBindings := []*Binding{}
	bindings := []*Binding{}
	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		tag := field.Tag.Get(cfg.getTagKey())
		tagParts := strings.Split(tag, ",")
		if tag == "-" {
			continue
		}
		if field.Anonymous && (tag == "" || tagParts[0] == "") {
			if field.Type.Kind() == reflect.Struct {
				structDescriptor, err := describeStruct(cfg, field.Type)
				if err != nil {
					return nil, err
				}
				for _, binding := range structDescriptor.Fields {
					binding.levels = append([]int{i}, binding.levels...)
					omitempty := binding.Encoder.(*structFieldEncoder).omitempty
					binding.Encoder = &structFieldEncoder{&field, binding.Encoder, omitempty}
					binding.Decoder = &structFieldDecoder{&field, binding.Decoder}
					embeddedBindings = append(embeddedBindings, binding)
				}
				continue
			} else if field.Type.Kind() == reflect.Ptr && field.Type.Elem().Kind() == reflect.Struct {
				structDescriptor, err := describeStruct(cfg, field.Type.Elem())
				if err != nil {
					return nil, err
				}
				for _, binding := range structDescriptor.Fields {
					binding.levels = append([]int{i}, binding.levels...)
					omitempty := binding.Encoder.(*structFieldEncoder).omitempty
					binding.Encoder = &OptionalEncoder{binding.Encoder}
					binding.Encoder = &structFieldEncoder{&field, binding.Encoder, omitempty}
					binding.Decoder = &deferenceDecoder{field.Type.Elem(), binding.Decoder}
					binding.Decoder = &structFieldDecoder{&field, binding.Decoder}
					embeddedBindings = append(embeddedBindings, binding)
				}
				continue
			}
		}
		fieldNames := calcFieldNames(field.Name, tagParts[0], tag)
		fieldCacheKey := fmt.Sprintf("%s/%s", typ.String(), field.Name)
		decoder := fieldDecoders[fieldCacheKey]
		if decoder == nil {
			var err error
			decoder, err = decoderOfType(cfg, field.Type)
			if len(fieldNames) > 0 && err != nil {
				return nil, err
			}
		}
		encoder := fieldEncoders[fieldCacheKey]
		if encoder == nil {
			var err error
			encoder, err = encoderOfType(cfg, field.Type)
			if len(fieldNames) > 0 && err != nil {
				return nil, err
			}
			// map is stored as pointer in the struct,
			// and treat nil or empty map as empty field
			if encoder != nil && field.Type.Kind() == reflect.Map {
				encoder = &optionalMapEncoder{encoder}
			}
		}
		binding := &Binding{
			Field:     &field,
			FromNames: fieldNames,
			ToNames:   fieldNames,
			Decoder:   decoder,
			Encoder:   encoder,
		}
		binding.levels = []int{i}
		bindings = append(bindings, binding)
	}
	return createStructDescriptor(cfg, typ, bindings, embeddedBindings), nil
}
func createStructDescriptor(cfg *frozenConfig, typ reflect.Type, bindings []*Binding, embeddedBindings []*Binding) *StructDescriptor {
	onePtrEmbedded := false
	onePtrOptimization := false
	if typ.NumField() == 1 {
		firstField := typ.Field(0)
		switch firstField.Type.Kind() {
		case reflect.Ptr:
			if firstField.Anonymous && firstField.Type.Elem().Kind() == reflect.Struct {
				onePtrEmbedded = true
			}
			fallthrough
		case reflect.Map:
			onePtrOptimization = true
		case reflect.Struct:
			onePtrOptimization = isStructOnePtr(firstField.Type)
		}
	}
	structDescriptor := &StructDescriptor{
		onePtrEmbedded:     onePtrEmbedded,
		onePtrOptimization: onePtrOptimization,
		Type:               typ,
		Fields:             bindings,
	}
	for _, extension := range extensions {
		extension.UpdateStructDescriptor(structDescriptor)
	}
	for _, extension := range cfg.extensions {
		extension.UpdateStructDescriptor(structDescriptor)
	}
	processTags(structDescriptor, cfg)
	// merge normal & embedded bindings & sort with original order
	allBindings := sortableBindings(append(embeddedBindings, structDescriptor.Fields...))
	sort.Sort(allBindings)
	structDescriptor.Fields = allBindings
	return structDescriptor
}

func isStructOnePtr(typ reflect.Type) bool {
	if typ.NumField() == 1 {
		firstField := typ.Field(0)
		switch firstField.Type.Kind() {
		case reflect.Ptr:
			return true
		case reflect.Map:
			return true
		case reflect.Struct:
			return isStructOnePtr(firstField.Type)
		}
	}
	return false
}

type sortableBindings []*Binding

func (bindings sortableBindings) Len() int {
	return len(bindings)
}

func (bindings sortableBindings) Less(i, j int) bool {
	left := bindings[i].levels
	right := bindings[j].levels
	k := 0
	for {
		if left[k] < right[k] {
			return true
		} else if left[k] > right[k] {
			return false
		}
		k++
	}
}

func (bindings sortableBindings) Swap(i, j int) {
	bindings[i], bindings[j] = bindings[j], bindings[i]
}

func processTags(structDescriptor *StructDescriptor, cfg *frozenConfig) {
	for _, binding := range structDescriptor.Fields {
		shouldOmitEmpty := false
		tagParts := strings.Split(binding.Field.Tag.Get(cfg.getTagKey()), ",")
		for _, tagPart := range tagParts[1:] {
			if tagPart == "omitempty" {
				shouldOmitEmpty = true
			} else if tagPart == "string" {
				if binding.Field.Type.Kind() == reflect.String {
					binding.Decoder = &stringModeStringDecoder{binding.Decoder, cfg}
					binding.Encoder = &stringModeStringEncoder{binding.Encoder, cfg}
				} else {
					binding.Decoder = &stringModeNumberDecoder{binding.Decoder}
					binding.Encoder = &stringModeNumberEncoder{binding.Encoder}
				}
			}
		}
		binding.Decoder = &structFieldDecoder{binding.Field, binding.Decoder}
		binding.Encoder = &structFieldEncoder{binding.Field, binding.Encoder, shouldOmitEmpty}
	}
}

func calcFieldNames(originalFieldName string, tagProvidedFieldName string, wholeTag string) []string {
	// ignore?
	if wholeTag == "-" {
		return []string{}
	}
	// rename?
	var fieldNames []string
	if tagProvidedFieldName == "" {
		fieldNames = []string{originalFieldName}
	} else {
		fieldNames = []string{tagProvidedFieldName}
	}
	// private?
	isNotExported := unicode.IsLower(rune(originalFieldName[0]))
	if isNotExported {
		fieldNames = []string{}
	}
	return fieldNames
}
