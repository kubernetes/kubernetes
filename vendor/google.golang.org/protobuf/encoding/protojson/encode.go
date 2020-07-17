// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protojson

import (
	"encoding/base64"
	"fmt"
	"sort"

	"google.golang.org/protobuf/internal/encoding/json"
	"google.golang.org/protobuf/internal/encoding/messageset"
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/internal/flags"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/proto"
	pref "google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

const defaultIndent = "  "

// Format formats the message as a multiline string.
// This function is only intended for human consumption and ignores errors.
// Do not depend on the output being stable. It may change over time across
// different versions of the program.
func Format(m proto.Message) string {
	return MarshalOptions{Multiline: true}.Format(m)
}

// Marshal writes the given proto.Message in JSON format using default options.
// Do not depend on the output being stable. It may change over time across
// different versions of the program.
func Marshal(m proto.Message) ([]byte, error) {
	return MarshalOptions{}.Marshal(m)
}

// MarshalOptions is a configurable JSON format marshaler.
type MarshalOptions struct {
	pragma.NoUnkeyedLiterals

	// Multiline specifies whether the marshaler should format the output in
	// indented-form with every textual element on a new line.
	// If Indent is an empty string, then an arbitrary indent is chosen.
	Multiline bool

	// Indent specifies the set of indentation characters to use in a multiline
	// formatted output such that every entry is preceded by Indent and
	// terminated by a newline. If non-empty, then Multiline is treated as true.
	// Indent can only be composed of space or tab characters.
	Indent string

	// AllowPartial allows messages that have missing required fields to marshal
	// without returning an error. If AllowPartial is false (the default),
	// Marshal will return error if there are any missing required fields.
	AllowPartial bool

	// UseProtoNames uses proto field name instead of lowerCamelCase name in JSON
	// field names.
	UseProtoNames bool

	// UseEnumNumbers emits enum values as numbers.
	UseEnumNumbers bool

	// EmitUnpopulated specifies whether to emit unpopulated fields. It does not
	// emit unpopulated oneof fields or unpopulated extension fields.
	// The JSON value emitted for unpopulated fields are as follows:
	//  ╔═══════╤════════════════════════════╗
	//  ║ JSON  │ Protobuf field             ║
	//  ╠═══════╪════════════════════════════╣
	//  ║ false │ proto3 boolean fields      ║
	//  ║ 0     │ proto3 numeric fields      ║
	//  ║ ""    │ proto3 string/bytes fields ║
	//  ║ null  │ proto2 scalar fields       ║
	//  ║ null  │ message fields             ║
	//  ║ []    │ list fields                ║
	//  ║ {}    │ map fields                 ║
	//  ╚═══════╧════════════════════════════╝
	EmitUnpopulated bool

	// Resolver is used for looking up types when expanding google.protobuf.Any
	// messages. If nil, this defaults to using protoregistry.GlobalTypes.
	Resolver interface {
		protoregistry.ExtensionTypeResolver
		protoregistry.MessageTypeResolver
	}
}

// Format formats the message as a string.
// This method is only intended for human consumption and ignores errors.
// Do not depend on the output being stable. It may change over time across
// different versions of the program.
func (o MarshalOptions) Format(m proto.Message) string {
	if m == nil || !m.ProtoReflect().IsValid() {
		return "<nil>" // invalid syntax, but okay since this is for debugging
	}
	o.AllowPartial = true
	b, _ := o.Marshal(m)
	return string(b)
}

// Marshal marshals the given proto.Message in the JSON format using options in
// MarshalOptions. Do not depend on the output being stable. It may change over
// time across different versions of the program.
func (o MarshalOptions) Marshal(m proto.Message) ([]byte, error) {
	return o.marshal(m)
}

// marshal is a centralized function that all marshal operations go through.
// For profiling purposes, avoid changing the name of this function or
// introducing other code paths for marshal that do not go through this.
func (o MarshalOptions) marshal(m proto.Message) ([]byte, error) {
	if o.Multiline && o.Indent == "" {
		o.Indent = defaultIndent
	}
	if o.Resolver == nil {
		o.Resolver = protoregistry.GlobalTypes
	}

	internalEnc, err := json.NewEncoder(o.Indent)
	if err != nil {
		return nil, err
	}

	// Treat nil message interface as an empty message,
	// in which case the output in an empty JSON object.
	if m == nil {
		return []byte("{}"), nil
	}

	enc := encoder{internalEnc, o}
	if err := enc.marshalMessage(m.ProtoReflect()); err != nil {
		return nil, err
	}
	if o.AllowPartial {
		return enc.Bytes(), nil
	}
	return enc.Bytes(), proto.CheckInitialized(m)
}

type encoder struct {
	*json.Encoder
	opts MarshalOptions
}

// marshalMessage marshals the given protoreflect.Message.
func (e encoder) marshalMessage(m pref.Message) error {
	if isCustomType(m.Descriptor().FullName()) {
		return e.marshalCustomType(m)
	}

	e.StartObject()
	defer e.EndObject()
	if err := e.marshalFields(m); err != nil {
		return err
	}

	return nil
}

// marshalFields marshals the fields in the given protoreflect.Message.
func (e encoder) marshalFields(m pref.Message) error {
	messageDesc := m.Descriptor()
	if !flags.ProtoLegacy && messageset.IsMessageSet(messageDesc) {
		return errors.New("no support for proto1 MessageSets")
	}

	// Marshal out known fields.
	fieldDescs := messageDesc.Fields()
	for i := 0; i < fieldDescs.Len(); {
		fd := fieldDescs.Get(i)
		if od := fd.ContainingOneof(); od != nil {
			fd = m.WhichOneof(od)
			i += od.Fields().Len()
			if fd == nil {
				continue // unpopulated oneofs are not affected by EmitUnpopulated
			}
		} else {
			i++
		}

		val := m.Get(fd)
		if !m.Has(fd) {
			if !e.opts.EmitUnpopulated {
				continue
			}
			isProto2Scalar := fd.Syntax() == pref.Proto2 && fd.Default().IsValid()
			isSingularMessage := fd.Cardinality() != pref.Repeated && fd.Message() != nil
			if isProto2Scalar || isSingularMessage {
				// Use invalid value to emit null.
				val = pref.Value{}
			}
		}

		name := fd.JSONName()
		if e.opts.UseProtoNames {
			name = string(fd.Name())
			// Use type name for group field name.
			if fd.Kind() == pref.GroupKind {
				name = string(fd.Message().Name())
			}
		}
		if err := e.WriteName(name); err != nil {
			return err
		}
		if err := e.marshalValue(val, fd); err != nil {
			return err
		}
	}

	// Marshal out extensions.
	if err := e.marshalExtensions(m); err != nil {
		return err
	}
	return nil
}

// marshalValue marshals the given protoreflect.Value.
func (e encoder) marshalValue(val pref.Value, fd pref.FieldDescriptor) error {
	switch {
	case fd.IsList():
		return e.marshalList(val.List(), fd)
	case fd.IsMap():
		return e.marshalMap(val.Map(), fd)
	default:
		return e.marshalSingular(val, fd)
	}
}

// marshalSingular marshals the given non-repeated field value. This includes
// all scalar types, enums, messages, and groups.
func (e encoder) marshalSingular(val pref.Value, fd pref.FieldDescriptor) error {
	if !val.IsValid() {
		e.WriteNull()
		return nil
	}

	switch kind := fd.Kind(); kind {
	case pref.BoolKind:
		e.WriteBool(val.Bool())

	case pref.StringKind:
		if e.WriteString(val.String()) != nil {
			return errors.InvalidUTF8(string(fd.FullName()))
		}

	case pref.Int32Kind, pref.Sint32Kind, pref.Sfixed32Kind:
		e.WriteInt(val.Int())

	case pref.Uint32Kind, pref.Fixed32Kind:
		e.WriteUint(val.Uint())

	case pref.Int64Kind, pref.Sint64Kind, pref.Uint64Kind,
		pref.Sfixed64Kind, pref.Fixed64Kind:
		// 64-bit integers are written out as JSON string.
		e.WriteString(val.String())

	case pref.FloatKind:
		// Encoder.WriteFloat handles the special numbers NaN and infinites.
		e.WriteFloat(val.Float(), 32)

	case pref.DoubleKind:
		// Encoder.WriteFloat handles the special numbers NaN and infinites.
		e.WriteFloat(val.Float(), 64)

	case pref.BytesKind:
		e.WriteString(base64.StdEncoding.EncodeToString(val.Bytes()))

	case pref.EnumKind:
		if fd.Enum().FullName() == "google.protobuf.NullValue" {
			e.WriteNull()
		} else {
			desc := fd.Enum().Values().ByNumber(val.Enum())
			if e.opts.UseEnumNumbers || desc == nil {
				e.WriteInt(int64(val.Enum()))
			} else {
				e.WriteString(string(desc.Name()))
			}
		}

	case pref.MessageKind, pref.GroupKind:
		if err := e.marshalMessage(val.Message()); err != nil {
			return err
		}

	default:
		panic(fmt.Sprintf("%v has unknown kind: %v", fd.FullName(), kind))
	}
	return nil
}

// marshalList marshals the given protoreflect.List.
func (e encoder) marshalList(list pref.List, fd pref.FieldDescriptor) error {
	e.StartArray()
	defer e.EndArray()

	for i := 0; i < list.Len(); i++ {
		item := list.Get(i)
		if err := e.marshalSingular(item, fd); err != nil {
			return err
		}
	}
	return nil
}

type mapEntry struct {
	key   pref.MapKey
	value pref.Value
}

// marshalMap marshals given protoreflect.Map.
func (e encoder) marshalMap(mmap pref.Map, fd pref.FieldDescriptor) error {
	e.StartObject()
	defer e.EndObject()

	// Get a sorted list based on keyType first.
	entries := make([]mapEntry, 0, mmap.Len())
	mmap.Range(func(key pref.MapKey, val pref.Value) bool {
		entries = append(entries, mapEntry{key: key, value: val})
		return true
	})
	sortMap(fd.MapKey().Kind(), entries)

	// Write out sorted list.
	for _, entry := range entries {
		if err := e.WriteName(entry.key.String()); err != nil {
			return err
		}
		if err := e.marshalSingular(entry.value, fd.MapValue()); err != nil {
			return err
		}
	}
	return nil
}

// sortMap orders list based on value of key field for deterministic ordering.
func sortMap(keyKind pref.Kind, values []mapEntry) {
	sort.Slice(values, func(i, j int) bool {
		switch keyKind {
		case pref.Int32Kind, pref.Sint32Kind, pref.Sfixed32Kind,
			pref.Int64Kind, pref.Sint64Kind, pref.Sfixed64Kind:
			return values[i].key.Int() < values[j].key.Int()

		case pref.Uint32Kind, pref.Fixed32Kind,
			pref.Uint64Kind, pref.Fixed64Kind:
			return values[i].key.Uint() < values[j].key.Uint()
		}
		return values[i].key.String() < values[j].key.String()
	})
}

// marshalExtensions marshals extension fields.
func (e encoder) marshalExtensions(m pref.Message) error {
	type entry struct {
		key   string
		value pref.Value
		desc  pref.FieldDescriptor
	}

	// Get a sorted list based on field key first.
	var entries []entry
	m.Range(func(fd pref.FieldDescriptor, v pref.Value) bool {
		if !fd.IsExtension() {
			return true
		}

		// For MessageSet extensions, the name used is the parent message.
		name := fd.FullName()
		if messageset.IsMessageSetExtension(fd) {
			name = name.Parent()
		}

		// Use [name] format for JSON field name.
		entries = append(entries, entry{
			key:   string(name),
			value: v,
			desc:  fd,
		})
		return true
	})

	// Sort extensions lexicographically.
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].key < entries[j].key
	})

	// Write out sorted list.
	for _, entry := range entries {
		// JSON field name is the proto field name enclosed in [], similar to
		// textproto. This is consistent with Go v1 lib. C++ lib v3.7.0 does not
		// marshal out extension fields.
		if err := e.WriteName("[" + entry.key + "]"); err != nil {
			return err
		}
		if err := e.marshalValue(entry.value, entry.desc); err != nil {
			return err
		}
	}
	return nil
}
