// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protojson

import (
	"encoding/base64"
	"fmt"
	"math"
	"strconv"
	"strings"

	"google.golang.org/protobuf/internal/encoding/json"
	"google.golang.org/protobuf/internal/encoding/messageset"
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/internal/flags"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/internal/set"
	"google.golang.org/protobuf/proto"
	pref "google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

// Unmarshal reads the given []byte into the given proto.Message.
func Unmarshal(b []byte, m proto.Message) error {
	return UnmarshalOptions{}.Unmarshal(b, m)
}

// UnmarshalOptions is a configurable JSON format parser.
type UnmarshalOptions struct {
	pragma.NoUnkeyedLiterals

	// If AllowPartial is set, input for messages that will result in missing
	// required fields will not return an error.
	AllowPartial bool

	// If DiscardUnknown is set, unknown fields are ignored.
	DiscardUnknown bool

	// Resolver is used for looking up types when unmarshaling
	// google.protobuf.Any messages or extension fields.
	// If nil, this defaults to using protoregistry.GlobalTypes.
	Resolver interface {
		protoregistry.MessageTypeResolver
		protoregistry.ExtensionTypeResolver
	}
}

// Unmarshal reads the given []byte and populates the given proto.Message using
// options in UnmarshalOptions object. It will clear the message first before
// setting the fields. If it returns an error, the given message may be
// partially set.
func (o UnmarshalOptions) Unmarshal(b []byte, m proto.Message) error {
	return o.unmarshal(b, m)
}

// unmarshal is a centralized function that all unmarshal operations go through.
// For profiling purposes, avoid changing the name of this function or
// introducing other code paths for unmarshal that do not go through this.
func (o UnmarshalOptions) unmarshal(b []byte, m proto.Message) error {
	proto.Reset(m)

	if o.Resolver == nil {
		o.Resolver = protoregistry.GlobalTypes
	}

	dec := decoder{json.NewDecoder(b), o}
	if err := dec.unmarshalMessage(m.ProtoReflect(), false); err != nil {
		return err
	}

	// Check for EOF.
	tok, err := dec.Read()
	if err != nil {
		return err
	}
	if tok.Kind() != json.EOF {
		return dec.unexpectedTokenError(tok)
	}

	if o.AllowPartial {
		return nil
	}
	return proto.CheckInitialized(m)
}

type decoder struct {
	*json.Decoder
	opts UnmarshalOptions
}

// newError returns an error object with position info.
func (d decoder) newError(pos int, f string, x ...interface{}) error {
	line, column := d.Position(pos)
	head := fmt.Sprintf("(line %d:%d): ", line, column)
	return errors.New(head+f, x...)
}

// unexpectedTokenError returns a syntax error for the given unexpected token.
func (d decoder) unexpectedTokenError(tok json.Token) error {
	return d.syntaxError(tok.Pos(), "unexpected token %s", tok.RawString())
}

// syntaxError returns a syntax error for given position.
func (d decoder) syntaxError(pos int, f string, x ...interface{}) error {
	line, column := d.Position(pos)
	head := fmt.Sprintf("syntax error (line %d:%d): ", line, column)
	return errors.New(head+f, x...)
}

// unmarshalMessage unmarshals a message into the given protoreflect.Message.
func (d decoder) unmarshalMessage(m pref.Message, skipTypeURL bool) error {
	if isCustomType(m.Descriptor().FullName()) {
		return d.unmarshalCustomType(m)
	}

	tok, err := d.Read()
	if err != nil {
		return err
	}
	if tok.Kind() != json.ObjectOpen {
		return d.unexpectedTokenError(tok)
	}

	if err := d.unmarshalFields(m, skipTypeURL); err != nil {
		return err
	}

	return nil
}

// unmarshalFields unmarshals the fields into the given protoreflect.Message.
func (d decoder) unmarshalFields(m pref.Message, skipTypeURL bool) error {
	messageDesc := m.Descriptor()
	if !flags.ProtoLegacy && messageset.IsMessageSet(messageDesc) {
		return errors.New("no support for proto1 MessageSets")
	}

	var seenNums set.Ints
	var seenOneofs set.Ints
	fieldDescs := messageDesc.Fields()
	for {
		// Read field name.
		tok, err := d.Read()
		if err != nil {
			return err
		}
		switch tok.Kind() {
		default:
			return d.unexpectedTokenError(tok)
		case json.ObjectClose:
			return nil
		case json.Name:
			// Continue below.
		}

		name := tok.Name()
		// Unmarshaling a non-custom embedded message in Any will contain the
		// JSON field "@type" which should be skipped because it is not a field
		// of the embedded message, but simply an artifact of the Any format.
		if skipTypeURL && name == "@type" {
			d.Read()
			continue
		}

		// Get the FieldDescriptor.
		var fd pref.FieldDescriptor
		if strings.HasPrefix(name, "[") && strings.HasSuffix(name, "]") {
			// Only extension names are in [name] format.
			extName := pref.FullName(name[1 : len(name)-1])
			extType, err := d.findExtension(extName)
			if err != nil && err != protoregistry.NotFound {
				return d.newError(tok.Pos(), "unable to resolve %s: %v", tok.RawString(), err)
			}
			if extType != nil {
				fd = extType.TypeDescriptor()
				if !messageDesc.ExtensionRanges().Has(fd.Number()) || fd.ContainingMessage().FullName() != messageDesc.FullName() {
					return d.newError(tok.Pos(), "message %v cannot be extended by %v", messageDesc.FullName(), fd.FullName())
				}
			}
		} else {
			// The name can either be the JSON name or the proto field name.
			fd = fieldDescs.ByJSONName(name)
			if fd == nil {
				fd = fieldDescs.ByName(pref.Name(name))
				if fd == nil {
					// The proto name of a group field is in all lowercase,
					// while the textual field name is the group message name.
					gd := fieldDescs.ByName(pref.Name(strings.ToLower(name)))
					if gd != nil && gd.Kind() == pref.GroupKind && gd.Message().Name() == pref.Name(name) {
						fd = gd
					}
				} else if fd.Kind() == pref.GroupKind && fd.Message().Name() != pref.Name(name) {
					fd = nil // reset since field name is actually the message name
				}
			}
		}
		if flags.ProtoLegacy {
			if fd != nil && fd.IsWeak() && fd.Message().IsPlaceholder() {
				fd = nil // reset since the weak reference is not linked in
			}
		}

		if fd == nil {
			// Field is unknown.
			if d.opts.DiscardUnknown {
				if err := d.skipJSONValue(); err != nil {
					return err
				}
				continue
			}
			return d.newError(tok.Pos(), "unknown field %v", tok.RawString())
		}

		// Do not allow duplicate fields.
		num := uint64(fd.Number())
		if seenNums.Has(num) {
			return d.newError(tok.Pos(), "duplicate field %v", tok.RawString())
		}
		seenNums.Set(num)

		// No need to set values for JSON null unless the field type is
		// google.protobuf.Value or google.protobuf.NullValue.
		if tok, _ := d.Peek(); tok.Kind() == json.Null && !isKnownValue(fd) && !isNullValue(fd) {
			d.Read()
			continue
		}

		switch {
		case fd.IsList():
			list := m.Mutable(fd).List()
			if err := d.unmarshalList(list, fd); err != nil {
				return err
			}
		case fd.IsMap():
			mmap := m.Mutable(fd).Map()
			if err := d.unmarshalMap(mmap, fd); err != nil {
				return err
			}
		default:
			// If field is a oneof, check if it has already been set.
			if od := fd.ContainingOneof(); od != nil {
				idx := uint64(od.Index())
				if seenOneofs.Has(idx) {
					return d.newError(tok.Pos(), "error parsing %s, oneof %v is already set", tok.RawString(), od.FullName())
				}
				seenOneofs.Set(idx)
			}

			// Required or optional fields.
			if err := d.unmarshalSingular(m, fd); err != nil {
				return err
			}
		}
	}
}

// findExtension returns protoreflect.ExtensionType from the resolver if found.
func (d decoder) findExtension(xtName pref.FullName) (pref.ExtensionType, error) {
	xt, err := d.opts.Resolver.FindExtensionByName(xtName)
	if err == nil {
		return xt, nil
	}
	return messageset.FindMessageSetExtension(d.opts.Resolver, xtName)
}

func isKnownValue(fd pref.FieldDescriptor) bool {
	md := fd.Message()
	return md != nil && md.FullName() == "google.protobuf.Value"
}

func isNullValue(fd pref.FieldDescriptor) bool {
	ed := fd.Enum()
	return ed != nil && ed.FullName() == "google.protobuf.NullValue"
}

// unmarshalSingular unmarshals to the non-repeated field specified
// by the given FieldDescriptor.
func (d decoder) unmarshalSingular(m pref.Message, fd pref.FieldDescriptor) error {
	var val pref.Value
	var err error
	switch fd.Kind() {
	case pref.MessageKind, pref.GroupKind:
		val = m.NewField(fd)
		err = d.unmarshalMessage(val.Message(), false)
	default:
		val, err = d.unmarshalScalar(fd)
	}

	if err != nil {
		return err
	}
	m.Set(fd, val)
	return nil
}

// unmarshalScalar unmarshals to a scalar/enum protoreflect.Value specified by
// the given FieldDescriptor.
func (d decoder) unmarshalScalar(fd pref.FieldDescriptor) (pref.Value, error) {
	const b32 int = 32
	const b64 int = 64

	tok, err := d.Read()
	if err != nil {
		return pref.Value{}, err
	}

	kind := fd.Kind()
	switch kind {
	case pref.BoolKind:
		if tok.Kind() == json.Bool {
			return pref.ValueOfBool(tok.Bool()), nil
		}

	case pref.Int32Kind, pref.Sint32Kind, pref.Sfixed32Kind:
		if v, ok := unmarshalInt(tok, b32); ok {
			return v, nil
		}

	case pref.Int64Kind, pref.Sint64Kind, pref.Sfixed64Kind:
		if v, ok := unmarshalInt(tok, b64); ok {
			return v, nil
		}

	case pref.Uint32Kind, pref.Fixed32Kind:
		if v, ok := unmarshalUint(tok, b32); ok {
			return v, nil
		}

	case pref.Uint64Kind, pref.Fixed64Kind:
		if v, ok := unmarshalUint(tok, b64); ok {
			return v, nil
		}

	case pref.FloatKind:
		if v, ok := unmarshalFloat(tok, b32); ok {
			return v, nil
		}

	case pref.DoubleKind:
		if v, ok := unmarshalFloat(tok, b64); ok {
			return v, nil
		}

	case pref.StringKind:
		if tok.Kind() == json.String {
			return pref.ValueOfString(tok.ParsedString()), nil
		}

	case pref.BytesKind:
		if v, ok := unmarshalBytes(tok); ok {
			return v, nil
		}

	case pref.EnumKind:
		if v, ok := unmarshalEnum(tok, fd); ok {
			return v, nil
		}

	default:
		panic(fmt.Sprintf("unmarshalScalar: invalid scalar kind %v", kind))
	}

	return pref.Value{}, d.newError(tok.Pos(), "invalid value for %v type: %v", kind, tok.RawString())
}

func unmarshalInt(tok json.Token, bitSize int) (pref.Value, bool) {
	switch tok.Kind() {
	case json.Number:
		return getInt(tok, bitSize)

	case json.String:
		// Decode number from string.
		s := strings.TrimSpace(tok.ParsedString())
		if len(s) != len(tok.ParsedString()) {
			return pref.Value{}, false
		}
		dec := json.NewDecoder([]byte(s))
		tok, err := dec.Read()
		if err != nil {
			return pref.Value{}, false
		}
		return getInt(tok, bitSize)
	}
	return pref.Value{}, false
}

func getInt(tok json.Token, bitSize int) (pref.Value, bool) {
	n, ok := tok.Int(bitSize)
	if !ok {
		return pref.Value{}, false
	}
	if bitSize == 32 {
		return pref.ValueOfInt32(int32(n)), true
	}
	return pref.ValueOfInt64(n), true
}

func unmarshalUint(tok json.Token, bitSize int) (pref.Value, bool) {
	switch tok.Kind() {
	case json.Number:
		return getUint(tok, bitSize)

	case json.String:
		// Decode number from string.
		s := strings.TrimSpace(tok.ParsedString())
		if len(s) != len(tok.ParsedString()) {
			return pref.Value{}, false
		}
		dec := json.NewDecoder([]byte(s))
		tok, err := dec.Read()
		if err != nil {
			return pref.Value{}, false
		}
		return getUint(tok, bitSize)
	}
	return pref.Value{}, false
}

func getUint(tok json.Token, bitSize int) (pref.Value, bool) {
	n, ok := tok.Uint(bitSize)
	if !ok {
		return pref.Value{}, false
	}
	if bitSize == 32 {
		return pref.ValueOfUint32(uint32(n)), true
	}
	return pref.ValueOfUint64(n), true
}

func unmarshalFloat(tok json.Token, bitSize int) (pref.Value, bool) {
	switch tok.Kind() {
	case json.Number:
		return getFloat(tok, bitSize)

	case json.String:
		s := tok.ParsedString()
		switch s {
		case "NaN":
			if bitSize == 32 {
				return pref.ValueOfFloat32(float32(math.NaN())), true
			}
			return pref.ValueOfFloat64(math.NaN()), true
		case "Infinity":
			if bitSize == 32 {
				return pref.ValueOfFloat32(float32(math.Inf(+1))), true
			}
			return pref.ValueOfFloat64(math.Inf(+1)), true
		case "-Infinity":
			if bitSize == 32 {
				return pref.ValueOfFloat32(float32(math.Inf(-1))), true
			}
			return pref.ValueOfFloat64(math.Inf(-1)), true
		}

		// Decode number from string.
		if len(s) != len(strings.TrimSpace(s)) {
			return pref.Value{}, false
		}
		dec := json.NewDecoder([]byte(s))
		tok, err := dec.Read()
		if err != nil {
			return pref.Value{}, false
		}
		return getFloat(tok, bitSize)
	}
	return pref.Value{}, false
}

func getFloat(tok json.Token, bitSize int) (pref.Value, bool) {
	n, ok := tok.Float(bitSize)
	if !ok {
		return pref.Value{}, false
	}
	if bitSize == 32 {
		return pref.ValueOfFloat32(float32(n)), true
	}
	return pref.ValueOfFloat64(n), true
}

func unmarshalBytes(tok json.Token) (pref.Value, bool) {
	if tok.Kind() != json.String {
		return pref.Value{}, false
	}

	s := tok.ParsedString()
	enc := base64.StdEncoding
	if strings.ContainsAny(s, "-_") {
		enc = base64.URLEncoding
	}
	if len(s)%4 != 0 {
		enc = enc.WithPadding(base64.NoPadding)
	}
	b, err := enc.DecodeString(s)
	if err != nil {
		return pref.Value{}, false
	}
	return pref.ValueOfBytes(b), true
}

func unmarshalEnum(tok json.Token, fd pref.FieldDescriptor) (pref.Value, bool) {
	switch tok.Kind() {
	case json.String:
		// Lookup EnumNumber based on name.
		s := tok.ParsedString()
		if enumVal := fd.Enum().Values().ByName(pref.Name(s)); enumVal != nil {
			return pref.ValueOfEnum(enumVal.Number()), true
		}

	case json.Number:
		if n, ok := tok.Int(32); ok {
			return pref.ValueOfEnum(pref.EnumNumber(n)), true
		}

	case json.Null:
		// This is only valid for google.protobuf.NullValue.
		if isNullValue(fd) {
			return pref.ValueOfEnum(0), true
		}
	}

	return pref.Value{}, false
}

func (d decoder) unmarshalList(list pref.List, fd pref.FieldDescriptor) error {
	tok, err := d.Read()
	if err != nil {
		return err
	}
	if tok.Kind() != json.ArrayOpen {
		return d.unexpectedTokenError(tok)
	}

	switch fd.Kind() {
	case pref.MessageKind, pref.GroupKind:
		for {
			tok, err := d.Peek()
			if err != nil {
				return err
			}

			if tok.Kind() == json.ArrayClose {
				d.Read()
				return nil
			}

			val := list.NewElement()
			if err := d.unmarshalMessage(val.Message(), false); err != nil {
				return err
			}
			list.Append(val)
		}
	default:
		for {
			tok, err := d.Peek()
			if err != nil {
				return err
			}

			if tok.Kind() == json.ArrayClose {
				d.Read()
				return nil
			}

			val, err := d.unmarshalScalar(fd)
			if err != nil {
				return err
			}
			list.Append(val)
		}
	}

	return nil
}

func (d decoder) unmarshalMap(mmap pref.Map, fd pref.FieldDescriptor) error {
	tok, err := d.Read()
	if err != nil {
		return err
	}
	if tok.Kind() != json.ObjectOpen {
		return d.unexpectedTokenError(tok)
	}

	// Determine ahead whether map entry is a scalar type or a message type in
	// order to call the appropriate unmarshalMapValue func inside the for loop
	// below.
	var unmarshalMapValue func() (pref.Value, error)
	switch fd.MapValue().Kind() {
	case pref.MessageKind, pref.GroupKind:
		unmarshalMapValue = func() (pref.Value, error) {
			val := mmap.NewValue()
			if err := d.unmarshalMessage(val.Message(), false); err != nil {
				return pref.Value{}, err
			}
			return val, nil
		}
	default:
		unmarshalMapValue = func() (pref.Value, error) {
			return d.unmarshalScalar(fd.MapValue())
		}
	}

Loop:
	for {
		// Read field name.
		tok, err := d.Read()
		if err != nil {
			return err
		}
		switch tok.Kind() {
		default:
			return d.unexpectedTokenError(tok)
		case json.ObjectClose:
			break Loop
		case json.Name:
			// Continue.
		}

		// Unmarshal field name.
		pkey, err := d.unmarshalMapKey(tok, fd.MapKey())
		if err != nil {
			return err
		}

		// Check for duplicate field name.
		if mmap.Has(pkey) {
			return d.newError(tok.Pos(), "duplicate map key %v", tok.RawString())
		}

		// Read and unmarshal field value.
		pval, err := unmarshalMapValue()
		if err != nil {
			return err
		}

		mmap.Set(pkey, pval)
	}

	return nil
}

// unmarshalMapKey converts given token of Name kind into a protoreflect.MapKey.
// A map key type is any integral or string type.
func (d decoder) unmarshalMapKey(tok json.Token, fd pref.FieldDescriptor) (pref.MapKey, error) {
	const b32 = 32
	const b64 = 64
	const base10 = 10

	name := tok.Name()
	kind := fd.Kind()
	switch kind {
	case pref.StringKind:
		return pref.ValueOfString(name).MapKey(), nil

	case pref.BoolKind:
		switch name {
		case "true":
			return pref.ValueOfBool(true).MapKey(), nil
		case "false":
			return pref.ValueOfBool(false).MapKey(), nil
		}

	case pref.Int32Kind, pref.Sint32Kind, pref.Sfixed32Kind:
		if n, err := strconv.ParseInt(name, base10, b32); err == nil {
			return pref.ValueOfInt32(int32(n)).MapKey(), nil
		}

	case pref.Int64Kind, pref.Sint64Kind, pref.Sfixed64Kind:
		if n, err := strconv.ParseInt(name, base10, b64); err == nil {
			return pref.ValueOfInt64(int64(n)).MapKey(), nil
		}

	case pref.Uint32Kind, pref.Fixed32Kind:
		if n, err := strconv.ParseUint(name, base10, b32); err == nil {
			return pref.ValueOfUint32(uint32(n)).MapKey(), nil
		}

	case pref.Uint64Kind, pref.Fixed64Kind:
		if n, err := strconv.ParseUint(name, base10, b64); err == nil {
			return pref.ValueOfUint64(uint64(n)).MapKey(), nil
		}

	default:
		panic(fmt.Sprintf("invalid kind for map key: %v", kind))
	}

	return pref.MapKey{}, d.newError(tok.Pos(), "invalid value for %v key: %s", kind, tok.RawString())
}
