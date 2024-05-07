// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonpb

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"
	"google.golang.org/protobuf/encoding/protojson"
	protoV2 "google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

const wrapJSONUnmarshalV2 = false

// UnmarshalNext unmarshals the next JSON object from d into m.
func UnmarshalNext(d *json.Decoder, m proto.Message) error {
	return new(Unmarshaler).UnmarshalNext(d, m)
}

// Unmarshal unmarshals a JSON object from r into m.
func Unmarshal(r io.Reader, m proto.Message) error {
	return new(Unmarshaler).Unmarshal(r, m)
}

// UnmarshalString unmarshals a JSON object from s into m.
func UnmarshalString(s string, m proto.Message) error {
	return new(Unmarshaler).Unmarshal(strings.NewReader(s), m)
}

// Unmarshaler is a configurable object for converting from a JSON
// representation to a protocol buffer object.
type Unmarshaler struct {
	// AllowUnknownFields specifies whether to allow messages to contain
	// unknown JSON fields, as opposed to failing to unmarshal.
	AllowUnknownFields bool

	// AnyResolver is used to resolve the google.protobuf.Any well-known type.
	// If unset, the global registry is used by default.
	AnyResolver AnyResolver
}

// JSONPBUnmarshaler is implemented by protobuf messages that customize the way
// they are unmarshaled from JSON. Messages that implement this should also
// implement JSONPBMarshaler so that the custom format can be produced.
//
// The JSON unmarshaling must follow the JSON to proto specification:
//	https://developers.google.com/protocol-buffers/docs/proto3#json
//
// Deprecated: Custom types should implement protobuf reflection instead.
type JSONPBUnmarshaler interface {
	UnmarshalJSONPB(*Unmarshaler, []byte) error
}

// Unmarshal unmarshals a JSON object from r into m.
func (u *Unmarshaler) Unmarshal(r io.Reader, m proto.Message) error {
	return u.UnmarshalNext(json.NewDecoder(r), m)
}

// UnmarshalNext unmarshals the next JSON object from d into m.
func (u *Unmarshaler) UnmarshalNext(d *json.Decoder, m proto.Message) error {
	if m == nil {
		return errors.New("invalid nil message")
	}

	// Parse the next JSON object from the stream.
	raw := json.RawMessage{}
	if err := d.Decode(&raw); err != nil {
		return err
	}

	// Check for custom unmarshalers first since they may not properly
	// implement protobuf reflection that the logic below relies on.
	if jsu, ok := m.(JSONPBUnmarshaler); ok {
		return jsu.UnmarshalJSONPB(u, raw)
	}

	mr := proto.MessageReflect(m)

	// NOTE: For historical reasons, a top-level null is treated as a noop.
	// This is incorrect, but kept for compatibility.
	if string(raw) == "null" && mr.Descriptor().FullName() != "google.protobuf.Value" {
		return nil
	}

	if wrapJSONUnmarshalV2 {
		// NOTE: If input message is non-empty, we need to preserve merge semantics
		// of the old jsonpb implementation. These semantics are not supported by
		// the protobuf JSON specification.
		isEmpty := true
		mr.Range(func(protoreflect.FieldDescriptor, protoreflect.Value) bool {
			isEmpty = false // at least one iteration implies non-empty
			return false
		})
		if !isEmpty {
			// Perform unmarshaling into a newly allocated, empty message.
			mr = mr.New()

			// Use a defer to copy all unmarshaled fields into the original message.
			dst := proto.MessageReflect(m)
			defer mr.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
				dst.Set(fd, v)
				return true
			})
		}

		// Unmarshal using the v2 JSON unmarshaler.
		opts := protojson.UnmarshalOptions{
			DiscardUnknown: u.AllowUnknownFields,
		}
		if u.AnyResolver != nil {
			opts.Resolver = anyResolver{u.AnyResolver}
		}
		return opts.Unmarshal(raw, mr.Interface())
	} else {
		if err := u.unmarshalMessage(mr, raw); err != nil {
			return err
		}
		return protoV2.CheckInitialized(mr.Interface())
	}
}

func (u *Unmarshaler) unmarshalMessage(m protoreflect.Message, in []byte) error {
	md := m.Descriptor()
	fds := md.Fields()

	if jsu, ok := proto.MessageV1(m.Interface()).(JSONPBUnmarshaler); ok {
		return jsu.UnmarshalJSONPB(u, in)
	}

	if string(in) == "null" && md.FullName() != "google.protobuf.Value" {
		return nil
	}

	switch wellKnownType(md.FullName()) {
	case "Any":
		var jsonObject map[string]json.RawMessage
		if err := json.Unmarshal(in, &jsonObject); err != nil {
			return err
		}

		rawTypeURL, ok := jsonObject["@type"]
		if !ok {
			return errors.New("Any JSON doesn't have '@type'")
		}
		typeURL, err := unquoteString(string(rawTypeURL))
		if err != nil {
			return fmt.Errorf("can't unmarshal Any's '@type': %q", rawTypeURL)
		}
		m.Set(fds.ByNumber(1), protoreflect.ValueOfString(typeURL))

		var m2 protoreflect.Message
		if u.AnyResolver != nil {
			mi, err := u.AnyResolver.Resolve(typeURL)
			if err != nil {
				return err
			}
			m2 = proto.MessageReflect(mi)
		} else {
			mt, err := protoregistry.GlobalTypes.FindMessageByURL(typeURL)
			if err != nil {
				if err == protoregistry.NotFound {
					return fmt.Errorf("could not resolve Any message type: %v", typeURL)
				}
				return err
			}
			m2 = mt.New()
		}

		if wellKnownType(m2.Descriptor().FullName()) != "" {
			rawValue, ok := jsonObject["value"]
			if !ok {
				return errors.New("Any JSON doesn't have 'value'")
			}
			if err := u.unmarshalMessage(m2, rawValue); err != nil {
				return fmt.Errorf("can't unmarshal Any nested proto %v: %v", typeURL, err)
			}
		} else {
			delete(jsonObject, "@type")
			rawJSON, err := json.Marshal(jsonObject)
			if err != nil {
				return fmt.Errorf("can't generate JSON for Any's nested proto to be unmarshaled: %v", err)
			}
			if err = u.unmarshalMessage(m2, rawJSON); err != nil {
				return fmt.Errorf("can't unmarshal Any nested proto %v: %v", typeURL, err)
			}
		}

		rawWire, err := protoV2.Marshal(m2.Interface())
		if err != nil {
			return fmt.Errorf("can't marshal proto %v into Any.Value: %v", typeURL, err)
		}
		m.Set(fds.ByNumber(2), protoreflect.ValueOfBytes(rawWire))
		return nil
	case "BoolValue", "BytesValue", "StringValue",
		"Int32Value", "UInt32Value", "FloatValue",
		"Int64Value", "UInt64Value", "DoubleValue":
		fd := fds.ByNumber(1)
		v, err := u.unmarshalValue(m.NewField(fd), in, fd)
		if err != nil {
			return err
		}
		m.Set(fd, v)
		return nil
	case "Duration":
		v, err := unquoteString(string(in))
		if err != nil {
			return err
		}
		d, err := time.ParseDuration(v)
		if err != nil {
			return fmt.Errorf("bad Duration: %v", err)
		}

		sec := d.Nanoseconds() / 1e9
		nsec := d.Nanoseconds() % 1e9
		m.Set(fds.ByNumber(1), protoreflect.ValueOfInt64(int64(sec)))
		m.Set(fds.ByNumber(2), protoreflect.ValueOfInt32(int32(nsec)))
		return nil
	case "Timestamp":
		v, err := unquoteString(string(in))
		if err != nil {
			return err
		}
		t, err := time.Parse(time.RFC3339Nano, v)
		if err != nil {
			return fmt.Errorf("bad Timestamp: %v", err)
		}

		sec := t.Unix()
		nsec := t.Nanosecond()
		m.Set(fds.ByNumber(1), protoreflect.ValueOfInt64(int64(sec)))
		m.Set(fds.ByNumber(2), protoreflect.ValueOfInt32(int32(nsec)))
		return nil
	case "Value":
		switch {
		case string(in) == "null":
			m.Set(fds.ByNumber(1), protoreflect.ValueOfEnum(0))
		case string(in) == "true":
			m.Set(fds.ByNumber(4), protoreflect.ValueOfBool(true))
		case string(in) == "false":
			m.Set(fds.ByNumber(4), protoreflect.ValueOfBool(false))
		case hasPrefixAndSuffix('"', in, '"'):
			s, err := unquoteString(string(in))
			if err != nil {
				return fmt.Errorf("unrecognized type for Value %q", in)
			}
			m.Set(fds.ByNumber(3), protoreflect.ValueOfString(s))
		case hasPrefixAndSuffix('[', in, ']'):
			v := m.Mutable(fds.ByNumber(6))
			return u.unmarshalMessage(v.Message(), in)
		case hasPrefixAndSuffix('{', in, '}'):
			v := m.Mutable(fds.ByNumber(5))
			return u.unmarshalMessage(v.Message(), in)
		default:
			f, err := strconv.ParseFloat(string(in), 0)
			if err != nil {
				return fmt.Errorf("unrecognized type for Value %q", in)
			}
			m.Set(fds.ByNumber(2), protoreflect.ValueOfFloat64(f))
		}
		return nil
	case "ListValue":
		var jsonArray []json.RawMessage
		if err := json.Unmarshal(in, &jsonArray); err != nil {
			return fmt.Errorf("bad ListValue: %v", err)
		}

		lv := m.Mutable(fds.ByNumber(1)).List()
		for _, raw := range jsonArray {
			ve := lv.NewElement()
			if err := u.unmarshalMessage(ve.Message(), raw); err != nil {
				return err
			}
			lv.Append(ve)
		}
		return nil
	case "Struct":
		var jsonObject map[string]json.RawMessage
		if err := json.Unmarshal(in, &jsonObject); err != nil {
			return fmt.Errorf("bad StructValue: %v", err)
		}

		mv := m.Mutable(fds.ByNumber(1)).Map()
		for key, raw := range jsonObject {
			kv := protoreflect.ValueOf(key).MapKey()
			vv := mv.NewValue()
			if err := u.unmarshalMessage(vv.Message(), raw); err != nil {
				return fmt.Errorf("bad value in StructValue for key %q: %v", key, err)
			}
			mv.Set(kv, vv)
		}
		return nil
	}

	var jsonObject map[string]json.RawMessage
	if err := json.Unmarshal(in, &jsonObject); err != nil {
		return err
	}

	// Handle known fields.
	for i := 0; i < fds.Len(); i++ {
		fd := fds.Get(i)
		if fd.IsWeak() && fd.Message().IsPlaceholder() {
			continue //  weak reference is not linked in
		}

		// Search for any raw JSON value associated with this field.
		var raw json.RawMessage
		name := string(fd.Name())
		if fd.Kind() == protoreflect.GroupKind {
			name = string(fd.Message().Name())
		}
		if v, ok := jsonObject[name]; ok {
			delete(jsonObject, name)
			raw = v
		}
		name = string(fd.JSONName())
		if v, ok := jsonObject[name]; ok {
			delete(jsonObject, name)
			raw = v
		}

		field := m.NewField(fd)
		// Unmarshal the field value.
		if raw == nil || (string(raw) == "null" && !isSingularWellKnownValue(fd) && !isSingularJSONPBUnmarshaler(field, fd)) {
			continue
		}
		v, err := u.unmarshalValue(field, raw, fd)
		if err != nil {
			return err
		}
		m.Set(fd, v)
	}

	// Handle extension fields.
	for name, raw := range jsonObject {
		if !strings.HasPrefix(name, "[") || !strings.HasSuffix(name, "]") {
			continue
		}

		// Resolve the extension field by name.
		xname := protoreflect.FullName(name[len("[") : len(name)-len("]")])
		xt, _ := protoregistry.GlobalTypes.FindExtensionByName(xname)
		if xt == nil && isMessageSet(md) {
			xt, _ = protoregistry.GlobalTypes.FindExtensionByName(xname.Append("message_set_extension"))
		}
		if xt == nil {
			continue
		}
		delete(jsonObject, name)
		fd := xt.TypeDescriptor()
		if fd.ContainingMessage().FullName() != m.Descriptor().FullName() {
			return fmt.Errorf("extension field %q does not extend message %q", xname, m.Descriptor().FullName())
		}

		field := m.NewField(fd)
		// Unmarshal the field value.
		if raw == nil || (string(raw) == "null" && !isSingularWellKnownValue(fd) && !isSingularJSONPBUnmarshaler(field, fd)) {
			continue
		}
		v, err := u.unmarshalValue(field, raw, fd)
		if err != nil {
			return err
		}
		m.Set(fd, v)
	}

	if !u.AllowUnknownFields && len(jsonObject) > 0 {
		for name := range jsonObject {
			return fmt.Errorf("unknown field %q in %v", name, md.FullName())
		}
	}
	return nil
}

func isSingularWellKnownValue(fd protoreflect.FieldDescriptor) bool {
	if fd.Cardinality() == protoreflect.Repeated {
		return false
	}
	if md := fd.Message(); md != nil {
		return md.FullName() == "google.protobuf.Value"
	}
	if ed := fd.Enum(); ed != nil {
		return ed.FullName() == "google.protobuf.NullValue"
	}
	return false
}

func isSingularJSONPBUnmarshaler(v protoreflect.Value, fd protoreflect.FieldDescriptor) bool {
	if fd.Message() != nil && fd.Cardinality() != protoreflect.Repeated {
		_, ok := proto.MessageV1(v.Interface()).(JSONPBUnmarshaler)
		return ok
	}
	return false
}

func (u *Unmarshaler) unmarshalValue(v protoreflect.Value, in []byte, fd protoreflect.FieldDescriptor) (protoreflect.Value, error) {
	switch {
	case fd.IsList():
		var jsonArray []json.RawMessage
		if err := json.Unmarshal(in, &jsonArray); err != nil {
			return v, err
		}
		lv := v.List()
		for _, raw := range jsonArray {
			ve, err := u.unmarshalSingularValue(lv.NewElement(), raw, fd)
			if err != nil {
				return v, err
			}
			lv.Append(ve)
		}
		return v, nil
	case fd.IsMap():
		var jsonObject map[string]json.RawMessage
		if err := json.Unmarshal(in, &jsonObject); err != nil {
			return v, err
		}
		kfd := fd.MapKey()
		vfd := fd.MapValue()
		mv := v.Map()
		for key, raw := range jsonObject {
			var kv protoreflect.MapKey
			if kfd.Kind() == protoreflect.StringKind {
				kv = protoreflect.ValueOf(key).MapKey()
			} else {
				v, err := u.unmarshalSingularValue(kfd.Default(), []byte(key), kfd)
				if err != nil {
					return v, err
				}
				kv = v.MapKey()
			}

			vv, err := u.unmarshalSingularValue(mv.NewValue(), raw, vfd)
			if err != nil {
				return v, err
			}
			mv.Set(kv, vv)
		}
		return v, nil
	default:
		return u.unmarshalSingularValue(v, in, fd)
	}
}

var nonFinite = map[string]float64{
	`"NaN"`:       math.NaN(),
	`"Infinity"`:  math.Inf(+1),
	`"-Infinity"`: math.Inf(-1),
}

func (u *Unmarshaler) unmarshalSingularValue(v protoreflect.Value, in []byte, fd protoreflect.FieldDescriptor) (protoreflect.Value, error) {
	switch fd.Kind() {
	case protoreflect.BoolKind:
		return unmarshalValue(in, new(bool))
	case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Sfixed32Kind:
		return unmarshalValue(trimQuote(in), new(int32))
	case protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind:
		return unmarshalValue(trimQuote(in), new(int64))
	case protoreflect.Uint32Kind, protoreflect.Fixed32Kind:
		return unmarshalValue(trimQuote(in), new(uint32))
	case protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
		return unmarshalValue(trimQuote(in), new(uint64))
	case protoreflect.FloatKind:
		if f, ok := nonFinite[string(in)]; ok {
			return protoreflect.ValueOfFloat32(float32(f)), nil
		}
		return unmarshalValue(trimQuote(in), new(float32))
	case protoreflect.DoubleKind:
		if f, ok := nonFinite[string(in)]; ok {
			return protoreflect.ValueOfFloat64(float64(f)), nil
		}
		return unmarshalValue(trimQuote(in), new(float64))
	case protoreflect.StringKind:
		return unmarshalValue(in, new(string))
	case protoreflect.BytesKind:
		return unmarshalValue(in, new([]byte))
	case protoreflect.EnumKind:
		if hasPrefixAndSuffix('"', in, '"') {
			vd := fd.Enum().Values().ByName(protoreflect.Name(trimQuote(in)))
			if vd == nil {
				return v, fmt.Errorf("unknown value %q for enum %s", in, fd.Enum().FullName())
			}
			return protoreflect.ValueOfEnum(vd.Number()), nil
		}
		return unmarshalValue(in, new(protoreflect.EnumNumber))
	case protoreflect.MessageKind, protoreflect.GroupKind:
		err := u.unmarshalMessage(v.Message(), in)
		return v, err
	default:
		panic(fmt.Sprintf("invalid kind %v", fd.Kind()))
	}
}

func unmarshalValue(in []byte, v interface{}) (protoreflect.Value, error) {
	err := json.Unmarshal(in, v)
	return protoreflect.ValueOf(reflect.ValueOf(v).Elem().Interface()), err
}

func unquoteString(in string) (out string, err error) {
	err = json.Unmarshal([]byte(in), &out)
	return out, err
}

func hasPrefixAndSuffix(prefix byte, in []byte, suffix byte) bool {
	if len(in) >= 2 && in[0] == prefix && in[len(in)-1] == suffix {
		return true
	}
	return false
}

// trimQuote is like unquoteString but simply strips surrounding quotes.
// This is incorrect, but is behavior done by the legacy implementation.
func trimQuote(in []byte) []byte {
	if len(in) >= 2 && in[0] == '"' && in[len(in)-1] == '"' {
		in = in[1 : len(in)-1]
	}
	return in
}
