package jsoniter

import (
	"fmt"
	"io"
	"reflect"
	"strings"
	"unsafe"
)

func encoderOfStruct(cfg *frozenConfig, typ reflect.Type) (ValEncoder, error) {
	type bindingTo struct {
		binding *Binding
		toName  string
		ignored bool
	}
	orderedBindings := []*bindingTo{}
	structDescriptor, err := describeStruct(cfg, typ)
	if err != nil {
		return nil, err
	}
	for _, binding := range structDescriptor.Fields {
		for _, toName := range binding.ToNames {
			new := &bindingTo{
				binding: binding,
				toName:  toName,
			}
			for _, old := range orderedBindings {
				if old.toName != toName {
					continue
				}
				old.ignored, new.ignored = resolveConflictBinding(cfg, old.binding, new.binding)
			}
			orderedBindings = append(orderedBindings, new)
		}
	}
	if len(orderedBindings) == 0 {
		return &emptyStructEncoder{}, nil
	}
	finalOrderedFields := []structFieldTo{}
	for _, bindingTo := range orderedBindings {
		if !bindingTo.ignored {
			finalOrderedFields = append(finalOrderedFields, structFieldTo{
				encoder: bindingTo.binding.Encoder.(*structFieldEncoder),
				toName:  bindingTo.toName,
			})
		}
	}
	return &structEncoder{structDescriptor.onePtrEmbedded, structDescriptor.onePtrOptimization, finalOrderedFields}, nil
}

func resolveConflictBinding(cfg *frozenConfig, old, new *Binding) (ignoreOld, ignoreNew bool) {
	newTagged := new.Field.Tag.Get(cfg.getTagKey()) != ""
	oldTagged := old.Field.Tag.Get(cfg.getTagKey()) != ""
	if newTagged {
		if oldTagged {
			if len(old.levels) > len(new.levels) {
				return true, false
			} else if len(new.levels) > len(old.levels) {
				return false, true
			} else {
				return true, true
			}
		} else {
			return true, false
		}
	} else {
		if oldTagged {
			return true, false
		}
		if len(old.levels) > len(new.levels) {
			return true, false
		} else if len(new.levels) > len(old.levels) {
			return false, true
		} else {
			return true, true
		}
	}
}

func decoderOfStruct(cfg *frozenConfig, typ reflect.Type) (ValDecoder, error) {
	bindings := map[string]*Binding{}
	structDescriptor, err := describeStruct(cfg, typ)
	if err != nil {
		return nil, err
	}
	for _, binding := range structDescriptor.Fields {
		for _, fromName := range binding.FromNames {
			old := bindings[fromName]
			if old == nil {
				bindings[fromName] = binding
				continue
			}
			ignoreOld, ignoreNew := resolveConflictBinding(cfg, old, binding)
			if ignoreOld {
				delete(bindings, fromName)
			}
			if !ignoreNew {
				bindings[fromName] = binding
			}
		}
	}
	fields := map[string]*structFieldDecoder{}
	for k, binding := range bindings {
		fields[strings.ToLower(k)] = binding.Decoder.(*structFieldDecoder)
	}
	return createStructDecoder(typ, fields)
}

type structFieldEncoder struct {
	field        *reflect.StructField
	fieldEncoder ValEncoder
	omitempty    bool
}

func (encoder *structFieldEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	fieldPtr := unsafe.Pointer(uintptr(ptr) + encoder.field.Offset)
	encoder.fieldEncoder.Encode(fieldPtr, stream)
	if stream.Error != nil && stream.Error != io.EOF {
		stream.Error = fmt.Errorf("%s: %s", encoder.field.Name, stream.Error.Error())
	}
}

func (encoder *structFieldEncoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *structFieldEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	fieldPtr := unsafe.Pointer(uintptr(ptr) + encoder.field.Offset)
	return encoder.fieldEncoder.IsEmpty(fieldPtr)
}

type structEncoder struct {
	onePtrEmbedded     bool
	onePtrOptimization bool
	fields             []structFieldTo
}

type structFieldTo struct {
	encoder *structFieldEncoder
	toName  string
}

func (encoder *structEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteObjectStart()
	isNotFirst := false
	for _, field := range encoder.fields {
		if field.encoder.omitempty && field.encoder.IsEmpty(ptr) {
			continue
		}
		if isNotFirst {
			stream.WriteMore()
		}
		stream.WriteObjectField(field.toName)
		field.encoder.Encode(ptr, stream)
		isNotFirst = true
	}
	stream.WriteObjectEnd()
}

func (encoder *structEncoder) EncodeInterface(val interface{}, stream *Stream) {
	e := (*emptyInterface)(unsafe.Pointer(&val))
	if encoder.onePtrOptimization {
		if e.word == nil && encoder.onePtrEmbedded {
			stream.WriteObjectStart()
			stream.WriteObjectEnd()
			return
		}
		ptr := uintptr(e.word)
		e.word = unsafe.Pointer(&ptr)
	}
	if reflect.TypeOf(val).Kind() == reflect.Ptr {
		encoder.Encode(unsafe.Pointer(&e.word), stream)
	} else {
		encoder.Encode(e.word, stream)
	}
}

func (encoder *structEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return false
}

type emptyStructEncoder struct {
}

func (encoder *emptyStructEncoder) Encode(ptr unsafe.Pointer, stream *Stream) {
	stream.WriteEmptyObject()
}

func (encoder *emptyStructEncoder) EncodeInterface(val interface{}, stream *Stream) {
	WriteToStream(val, stream, encoder)
}

func (encoder *emptyStructEncoder) IsEmpty(ptr unsafe.Pointer) bool {
	return false
}
