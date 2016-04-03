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

	"github.com/golang/protobuf/proto"
)

var (
	byteArrayType = reflect.TypeOf([]byte{})
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
		prop := jsonProperties(valueField)
		if !firstField {
			m.writeSep(out)
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
			prop.OrigName = fmt.Sprintf("[%s]", desc.Name)
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
	out.write(prop.OrigName)
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

	var err error
	v = reflect.Indirect(v)

	// Handle repeated elements.
	if v.Type() != byteArrayType && v.Kind() == reflect.Slice {
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
		return m.marshalObject(out, v.Addr().Interface().(proto.Message), indent+m.Indent)
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

// Unmarshal unmarshals a JSON object stream into a protocol
// buffer. This function is lenient and will decode any options
// permutations of the related Marshaler.
func Unmarshal(r io.Reader, pb proto.Message) error {
	inputValue := json.RawMessage{}
	if err := json.NewDecoder(r).Decode(&inputValue); err != nil {
		return err
	}
	return unmarshalValue(reflect.ValueOf(pb).Elem(), inputValue)
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

		sprops := proto.GetProperties(targetType)
		for i := 0; i < target.NumField(); i++ {
			ft := target.Type().Field(i)
			if strings.HasPrefix(ft.Name, "XXX_") {
				continue
			}
			fieldName := jsonProperties(ft).OrigName

			valueForField, ok := jsonFields[fieldName]
			if !ok {
				continue
			}
			delete(jsonFields, fieldName)

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
		for fname, raw := range jsonFields {
			if oop, ok := sprops.OneofTypes[fname]; ok {
				nv := reflect.New(oop.Type.Elem())
				target.Field(oop.Field).Set(nv)
				if err := unmarshalValue(nv.Elem().Field(0), raw); err != nil {
					return err
				}
				delete(jsonFields, fname)
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

	// Handle arrays (which aren't encoded bytes)
	if targetType != byteArrayType && targetType.Kind() == reflect.Slice {
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

// jsonProperties returns parsed proto.Properties for the field.
func jsonProperties(f reflect.StructField) *proto.Properties {
	var prop proto.Properties
	prop.Init(f.Type, f.Name, f.Tag.Get("protobuf"), &f)
	return &prop
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
