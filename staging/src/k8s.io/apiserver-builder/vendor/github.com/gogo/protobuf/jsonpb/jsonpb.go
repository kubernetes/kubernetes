// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2015 The Go Authors.  All rights reserved.
// https://github.com/golang/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/*
Package jsonpb provides marshaling and unmarshaling between protocol buffers and JSON.
It follows the specification at https://developers.google.com/protocol-buffers/docs/proto3#json.

This package produces a different output than the standard "encoding/json" package,
which does not operate correctly on protocol buffers.
*/
package jsonpb

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/gogo/protobuf/proto"
)

// Marshaler is a configurable object for converting between
// protocol buffer objects and a JSON representation for them.
type Marshaler struct {
	// Whether to render enum values as integers, as opposed to string values.
	EnumsAsInts bool

	// Whether to render fields with zero values.
	EmitDefaults bool

	// A string to indent each level by. The presence of this field will
	// also cause a space to appear between the field separator and
	// value, and for newlines to be appear between fields and array
	// elements.
	Indent string

	// Whether to use the original (.proto) name for fields.
	OrigName bool
}

// Marshal marshals a protocol buffer into JSON.
func (m *Marshaler) Marshal(out io.Writer, pb proto.Message) error {
	writer := &errWriter{writer: out}
	return m.marshalObject(writer, pb, "")
}

// MarshalToString converts a protocol buffer object to JSON string.
func (m *Marshaler) MarshalToString(pb proto.Message) (string, error) {
	var buf bytes.Buffer
	if err := m.Marshal(&buf, pb); err != nil {
		return "", err
	}
	return buf.String(), nil
}

type int32Slice []int32

// For sorting extensions ids to ensure stable output.
func (s int32Slice) Len() int           { return len(s) }
func (s int32Slice) Less(i, j int) bool { return s[i] < s[j] }
func (s int32Slice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// marshalObject writes a struct to the Writer.
func (m *Marshaler) marshalObject(out *errWriter, v proto.Message, indent string) error {
	out.write("{")
	if m.Indent != "" {
		out.write("\n")
	}

	s := reflect.ValueOf(v).Elem()
	firstField := true
	for i := 0; i < s.NumField(); i++ {
		value := s.Field(i)
		valueField := s.Type().Field(i)
		if strings.HasPrefix(valueField.Name, "XXX_") {
			continue
		}

		// IsNil will panic on most value kinds.
		switch value.Kind() {
		case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
			if value.IsNil() {
				continue
			}
		}

		if !m.EmitDefaults {
			switch value.Kind() {
			case reflect.Bool:
				if !value.Bool() {
					continue
				}
			case reflect.Int32, reflect.Int64:
				if value.Int() == 0 {
					continue
				}
			case reflect.Uint32, reflect.Uint64:
				if value.Uint() == 0 {
					continue
				}
			case reflect.Float32, reflect.Float64:
				if value.Float() == 0 {
					continue
				}
			case reflect.String:
				if value.Len() == 0 {
					continue
				}
			}
		}

		// Oneof fields need special handling.
		if valueField.Tag.Get("protobuf_oneof") != "" {
			// value is an interface containing &T{real_value}.
			sv := value.Elem().Elem() // interface -> *T -> T
			value = sv.Field(0)
			valueField = sv.Type().Field(0)
		}
		prop := jsonProperties(valueField, m.OrigName)
		if !firstField {
			m.writeSep(out)
		}
		// If the map value is a cast type, it may not implement proto.Message, therefore
		// allow the struct tag to declare the underlying message type. Instead of changing
		// the signatures of the child types (and because prop.mvalue is not public), use
		// CustomType as a passer.
		if value.Kind() == reflect.Map {
			if tag := valueField.Tag.Get("protobuf"); tag != "" {
				for _, v := range strings.Split(tag, ",") {
					if !strings.HasPrefix(v, "castvaluetype=") {
						continue
					}
					v = strings.TrimPrefix(v, "castvaluetype=")
					prop.CustomType = v
					break
				}
			}
		}
		if err := m.marshalField(out, prop, value, indent); err != nil {
			return err
		}
		firstField = false
	}

	// Handle proto2 extensions.
	if ep, ok := v.(extendableProto); ok {
		extensions := proto.RegisteredExtensions(v)
		extensionMap := ep.ExtensionMap()
		// Sort extensions for stable output.
		ids := make([]int32, 0, len(extensionMap))
		for id := range extensionMap {
			ids = append(ids, id)
		}
		sort.Sort(int32Slice(ids))
		for _, id := range ids {
			desc := extensions[id]
			if desc == nil {
				// unknown extension
				continue
			}
			ext, extErr := proto.GetExtension(ep, desc)
			if extErr != nil {
				return extErr
			}
			value := reflect.ValueOf(ext)
			var prop proto.Properties
			prop.Parse(desc.Tag)
			prop.JSONName = fmt.Sprintf("[%s]", desc.Name)
			if !firstField {
				m.writeSep(out)
			}
			if err := m.marshalField(out, &prop, value, indent); err != nil {
				return err
			}
			firstField = false
		}

	}

	if m.Indent != "" {
		out.write("\n")
		out.write(indent)
	}
	out.write("}")
	return out.err
}

func (m *Marshaler) writeSep(out *errWriter) {
	if m.Indent != "" {
		out.write(",\n")
	} else {
		out.write(",")
	}
}

// marshalField writes field description and value to the Writer.
func (m *Marshaler) marshalField(out *errWriter, prop *proto.Properties, v reflect.Value, indent string) error {
	if m.Indent != "" {
		out.write(indent)
		out.write(m.Indent)
	}
	out.write(`"`)
	out.write(prop.JSONName)
	out.write(`":`)
	if m.Indent != "" {
		out.write(" ")
	}
	if err := m.marshalValue(out, prop, v, indent); err != nil {
		return err
	}
	return nil
}

// marshalValue writes the value to the Writer.
func (m *Marshaler) marshalValue(out *errWriter, prop *proto.Properties, v reflect.Value, indent string) error {

	v = reflect.Indirect(v)

	// Handle repeated elements.
	if v.Kind() == reflect.Slice && v.Type().Elem().Kind() != reflect.Uint8 {
		out.write("[")
		comma := ""
		for i := 0; i < v.Len(); i++ {
			sliceVal := v.Index(i)
			out.write(comma)
			if m.Indent != "" {
				out.write("\n")
				out.write(indent)
				out.write(m.Indent)
				out.write(m.Indent)
			}
			m.marshalValue(out, prop, sliceVal, indent+m.Indent)
			comma = ","
		}
		if m.Indent != "" {
			out.write("\n")
			out.write(indent)
			out.write(m.Indent)
		}
		out.write("]")
		return out.err
	}

	// Handle enumerations.
	if !m.EnumsAsInts && prop.Enum != "" {
		// Unknown enum values will are stringified by the proto library as their
		// value. Such values should _not_ be quoted or they will be interpreted
		// as an enum string instead of their value.
		enumStr := v.Interface().(fmt.Stringer).String()
		var valStr string
		if v.Kind() == reflect.Ptr {
			valStr = strconv.Itoa(int(v.Elem().Int()))
		} else {
			valStr = strconv.Itoa(int(v.Int()))
		}

		if m, ok := v.Interface().(interface {
			MarshalJSON() ([]byte, error)
		}); ok {
			data, err := m.MarshalJSON()
			if err != nil {
				return err
			}
			enumStr = string(data)
			enumStr, err = strconv.Unquote(enumStr)
			if err != nil {
				return err
			}
		}

		isKnownEnum := enumStr != valStr

		if isKnownEnum {
			out.write(`"`)
		}
		out.write(enumStr)
		if isKnownEnum {
			out.write(`"`)
		}
		return out.err
	}

	// Handle nested messages.
	if v.Kind() == reflect.Struct {
		i := v
		if v.CanAddr() {
			i = v.Addr()
		} else {
			i = reflect.New(v.Type())
			i.Elem().Set(v)
		}
		iface := i.Interface()
		if iface == nil {
			out.write(`null`)
			return out.err
		}
		pm, ok := iface.(proto.Message)
		if !ok {
			if prop.CustomType == "" {
				return fmt.Errorf("%v does not implement proto.Message", v.Type())
			}
			t := proto.MessageType(prop.CustomType)
			if t == nil || !i.Type().ConvertibleTo(t) {
				return fmt.Errorf("%v declared custom type %s but it is not convertible to %v", v.Type(), prop.CustomType, t)
			}
			pm = i.Convert(t).Interface().(proto.Message)
		}
		return m.marshalObject(out, pm, indent+m.Indent)
	}

	// Handle maps.
	// Since Go randomizes map iteration, we sort keys for stable output.
	if v.Kind() == reflect.Map {
		out.write(`{`)
		keys := v.MapKeys()
		sort.Sort(mapKeys(keys))
		for i, k := range keys {
			if i > 0 {
				out.write(`,`)
			}
			if m.Indent != "" {
				out.write("\n")
				out.write(indent)
				out.write(m.Indent)
				out.write(m.Indent)
			}

			b, err := json.Marshal(k.Interface())
			if err != nil {
				return err
			}
			s := string(b)

			// If the JSON is not a string value, encode it again to make it one.
			if !strings.HasPrefix(s, `"`) {
				b, err := json.Marshal(s)
				if err != nil {
					return err
				}
				s = string(b)
			}

			out.write(s)
			out.write(`:`)
			if m.Indent != "" {
				out.write(` `)
			}

			if err := m.marshalValue(out, prop, v.MapIndex(k), indent+m.Indent); err != nil {
				return err
			}
		}
		if m.Indent != "" {
			out.write("\n")
			out.write(indent)
			out.write(m.Indent)
		}
		out.write(`}`)
		return out.err
	}

	// Default handling defers to the encoding/json library.
	b, err := json.Marshal(v.Interface())
	if err != nil {
		return err
	}
	needToQuote := string(b[0]) != `"` && (v.Kind() == reflect.Int64 || v.Kind() == reflect.Uint64)
	if needToQuote {
		out.write(`"`)
	}
	out.write(string(b))
	if needToQuote {
		out.write(`"`)
	}
	return out.err
}

// UnmarshalNext unmarshals the next protocol buffer from a JSON object stream.
// This function is lenient and will decode any options permutations of the
// related Marshaler.
func UnmarshalNext(dec *json.Decoder, pb proto.Message) error {
	inputValue := json.RawMessage{}
	if err := dec.Decode(&inputValue); err != nil {
		return err
	}
	return unmarshalValue(reflect.ValueOf(pb).Elem(), inputValue)
}

// Unmarshal unmarshals a JSON object stream into a protocol
// buffer. This function is lenient and will decode any options
// permutations of the related Marshaler.
func Unmarshal(r io.Reader, pb proto.Message) error {
	dec := json.NewDecoder(r)
	return UnmarshalNext(dec, pb)
}

// UnmarshalString will populate the fields of a protocol buffer based
// on a JSON string. This function is lenient and will decode any options
// permutations of the related Marshaler.
func UnmarshalString(str string, pb proto.Message) error {
	return Unmarshal(strings.NewReader(str), pb)
}

// unmarshalValue converts/copies a value into the target.
func unmarshalValue(target reflect.Value, inputValue json.RawMessage) error {
	targetType := target.Type()

	// Allocate memory for pointer fields.
	if targetType.Kind() == reflect.Ptr {
		target.Set(reflect.New(targetType.Elem()))
		return unmarshalValue(target.Elem(), inputValue)
	}

	// Handle nested messages.
	if targetType.Kind() == reflect.Struct {
		var jsonFields map[string]json.RawMessage
		if err := json.Unmarshal(inputValue, &jsonFields); err != nil {
			return err
		}

		consumeField := func(prop *proto.Properties) (json.RawMessage, bool) {
			// Be liberal in what names we accept; both orig_name and camelName are okay.
			fieldNames := acceptedJSONFieldNames(prop)

			vOrig, okOrig := jsonFields[fieldNames.orig]
			vCamel, okCamel := jsonFields[fieldNames.camel]
			if !okOrig && !okCamel {
				return nil, false
			}
			// If, for some reason, both are present in the data, favour the camelName.
			var raw json.RawMessage
			if okOrig {
				raw = vOrig
				delete(jsonFields, fieldNames.orig)
			}
			if okCamel {
				raw = vCamel
				delete(jsonFields, fieldNames.camel)
			}
			return raw, true
		}

		sprops := proto.GetProperties(targetType)
		for i := 0; i < target.NumField(); i++ {
			ft := target.Type().Field(i)
			if strings.HasPrefix(ft.Name, "XXX_") {
				continue
			}
			valueForField, ok := consumeField(sprops.Prop[i])
			if !ok {
				continue
			}

			// Handle enums, which have an underlying type of int32,
			// and may appear as strings. We do this while handling
			// the struct so we have access to the enum info.
			// The case of an enum appearing as a number is handled
			// by the recursive call to unmarshalValue.
			if enum := sprops.Prop[i].Enum; valueForField[0] == '"' && enum != "" {
				vmap := proto.EnumValueMap(enum)
				// Don't need to do unquoting; valid enum names
				// are from a limited character set.
				s := valueForField[1 : len(valueForField)-1]
				n, ok := vmap[string(s)]
				if !ok {
					return fmt.Errorf("unknown value %q for enum %s", s, enum)
				}
				f := target.Field(i)
				if f.Kind() == reflect.Ptr { // proto2
					f.Set(reflect.New(f.Type().Elem()))
					f = f.Elem()
				}
				f.SetInt(int64(n))
				continue
			}
			if err := unmarshalValue(target.Field(i), valueForField); err != nil {
				return err
			}
		}
		// Check for any oneof fields.
		if len(jsonFields) > 0 {
			for _, oop := range sprops.OneofTypes {
				raw, ok := consumeField(oop.Prop)
				if !ok {
					continue
				}
				nv := reflect.New(oop.Type.Elem())
				target.Field(oop.Field).Set(nv)
				if err := unmarshalValue(nv.Elem().Field(0), raw); err != nil {
					return err
				}
			}
		}
		if len(jsonFields) > 0 {
			// Pick any field to be the scapegoat.
			var f string
			for fname := range jsonFields {
				f = fname
				break
			}
			return fmt.Errorf("unknown field %q in %v", f, targetType)
		}
		return nil
	}

	// Handle arrays
	if targetType.Kind() == reflect.Slice {
		// Special case for encoded bytes. Pre-go1.5 doesn't support unmarshalling
		// strings into aliased []byte types.
		// https://github.com/golang/go/commit/4302fd0409da5e4f1d71471a6770dacdc3301197
		// https://github.com/golang/go/commit/c60707b14d6be26bf4213114d13070bff00d0b0a
		if targetType.Elem().Kind() == reflect.Uint8 {
			var out []byte
			if err := json.Unmarshal(inputValue, &out); err != nil {
				return err
			}
			target.SetBytes(out)
			return nil
		}

		var slc []json.RawMessage
		if err := json.Unmarshal(inputValue, &slc); err != nil {
			return err
		}
		len := len(slc)
		target.Set(reflect.MakeSlice(targetType, len, len))
		for i := 0; i < len; i++ {
			if err := unmarshalValue(target.Index(i), slc[i]); err != nil {
				return err
			}
		}
		return nil
	}

	// Handle maps (whose keys are always strings)
	if targetType.Kind() == reflect.Map {
		var mp map[string]json.RawMessage
		if err := json.Unmarshal(inputValue, &mp); err != nil {
			return err
		}
		target.Set(reflect.MakeMap(targetType))
		for ks, raw := range mp {
			// Unmarshal map key. The core json library already decoded the key into a
			// string, so we handle that specially. Other types were quoted post-serialization.
			var k reflect.Value
			if targetType.Key().Kind() == reflect.String {
				k = reflect.ValueOf(ks)
			} else {
				k = reflect.New(targetType.Key()).Elem()
				if err := unmarshalValue(k, json.RawMessage(ks)); err != nil {
					return err
				}
			}

			if !k.Type().AssignableTo(targetType.Key()) {
				k = k.Convert(targetType.Key())
			}

			// Unmarshal map value.
			v := reflect.New(targetType.Elem()).Elem()
			if err := unmarshalValue(v, raw); err != nil {
				return err
			}
			target.SetMapIndex(k, v)
		}
		return nil
	}

	// 64-bit integers can be encoded as strings. In this case we drop
	// the quotes and proceed as normal.
	isNum := targetType.Kind() == reflect.Int64 || targetType.Kind() == reflect.Uint64
	if isNum && strings.HasPrefix(string(inputValue), `"`) {
		inputValue = inputValue[1 : len(inputValue)-1]
	}

	// Use the encoding/json for parsing other value types.
	return json.Unmarshal(inputValue, target.Addr().Interface())
}

// jsonProperties returns parsed proto.Properties for the field and corrects JSONName attribute.
func jsonProperties(f reflect.StructField, origName bool) *proto.Properties {
	var prop proto.Properties
	prop.Init(f.Type, f.Name, f.Tag.Get("protobuf"), &f)
	if origName || prop.JSONName == "" {
		prop.JSONName = prop.OrigName
	}
	return &prop
}

type fieldNames struct {
	orig, camel string
}

func acceptedJSONFieldNames(prop *proto.Properties) fieldNames {
	opts := fieldNames{orig: prop.OrigName, camel: prop.OrigName}
	if prop.JSONName != "" {
		opts.camel = prop.JSONName
	}
	return opts
}

// extendableProto is an interface implemented by any protocol buffer that may be extended.
type extendableProto interface {
	proto.Message
	ExtensionRangeArray() []proto.ExtensionRange
	ExtensionMap() map[int32]proto.Extension
}

// Writer wrapper inspired by https://blog.golang.org/errors-are-values
type errWriter struct {
	writer io.Writer
	err    error
}

func (w *errWriter) write(str string) {
	if w.err != nil {
		return
	}
	_, w.err = w.writer.Write([]byte(str))
}

// Map fields may have key types of non-float scalars, strings and enums.
// The easiest way to sort them in some deterministic order is to use fmt.
// If this turns out to be inefficient we can always consider other options,
// such as doing a Schwartzian transform.
type mapKeys []reflect.Value

func (s mapKeys) Len() int      { return len(s) }
func (s mapKeys) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s mapKeys) Less(i, j int) bool {
	return fmt.Sprint(s[i].Interface()) < fmt.Sprint(s[j].Interface())
}
