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
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"
	"google.golang.org/protobuf/encoding/protojson"
	protoV2 "google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

const wrapJSONMarshalV2 = false

// Marshaler is a configurable object for marshaling protocol buffer messages
// to the specified JSON representation.
type Marshaler struct {
	// OrigName specifies whether to use the original protobuf name for fields.
	OrigName bool

	// EnumsAsInts specifies whether to render enum values as integers,
	// as opposed to string values.
	EnumsAsInts bool

	// EmitDefaults specifies whether to render fields with zero values.
	EmitDefaults bool

	// Indent controls whether the output is compact or not.
	// If empty, the output is compact JSON. Otherwise, every JSON object
	// entry and JSON array value will be on its own line.
	// Each line will be preceded by repeated copies of Indent, where the
	// number of copies is the current indentation depth.
	Indent string

	// AnyResolver is used to resolve the google.protobuf.Any well-known type.
	// If unset, the global registry is used by default.
	AnyResolver AnyResolver
}

// JSONPBMarshaler is implemented by protobuf messages that customize the
// way they are marshaled to JSON. Messages that implement this should also
// implement JSONPBUnmarshaler so that the custom format can be parsed.
//
// The JSON marshaling must follow the proto to JSON specification:
//	https://developers.google.com/protocol-buffers/docs/proto3#json
//
// Deprecated: Custom types should implement protobuf reflection instead.
type JSONPBMarshaler interface {
	MarshalJSONPB(*Marshaler) ([]byte, error)
}

// Marshal serializes a protobuf message as JSON into w.
func (jm *Marshaler) Marshal(w io.Writer, m proto.Message) error {
	b, err := jm.marshal(m)
	if len(b) > 0 {
		if _, err := w.Write(b); err != nil {
			return err
		}
	}
	return err
}

// MarshalToString serializes a protobuf message as JSON in string form.
func (jm *Marshaler) MarshalToString(m proto.Message) (string, error) {
	b, err := jm.marshal(m)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func (jm *Marshaler) marshal(m proto.Message) ([]byte, error) {
	v := reflect.ValueOf(m)
	if m == nil || (v.Kind() == reflect.Ptr && v.IsNil()) {
		return nil, errors.New("Marshal called with nil")
	}

	// Check for custom marshalers first since they may not properly
	// implement protobuf reflection that the logic below relies on.
	if jsm, ok := m.(JSONPBMarshaler); ok {
		return jsm.MarshalJSONPB(jm)
	}

	if wrapJSONMarshalV2 {
		opts := protojson.MarshalOptions{
			UseProtoNames:   jm.OrigName,
			UseEnumNumbers:  jm.EnumsAsInts,
			EmitUnpopulated: jm.EmitDefaults,
			Indent:          jm.Indent,
		}
		if jm.AnyResolver != nil {
			opts.Resolver = anyResolver{jm.AnyResolver}
		}
		return opts.Marshal(proto.MessageReflect(m).Interface())
	} else {
		// Check for unpopulated required fields first.
		m2 := proto.MessageReflect(m)
		if err := protoV2.CheckInitialized(m2.Interface()); err != nil {
			return nil, err
		}

		w := jsonWriter{Marshaler: jm}
		err := w.marshalMessage(m2, "", "")
		return w.buf, err
	}
}

type jsonWriter struct {
	*Marshaler
	buf []byte
}

func (w *jsonWriter) write(s string) {
	w.buf = append(w.buf, s...)
}

func (w *jsonWriter) marshalMessage(m protoreflect.Message, indent, typeURL string) error {
	if jsm, ok := proto.MessageV1(m.Interface()).(JSONPBMarshaler); ok {
		b, err := jsm.MarshalJSONPB(w.Marshaler)
		if err != nil {
			return err
		}
		if typeURL != "" {
			// we are marshaling this object to an Any type
			var js map[string]*json.RawMessage
			if err = json.Unmarshal(b, &js); err != nil {
				return fmt.Errorf("type %T produced invalid JSON: %v", m.Interface(), err)
			}
			turl, err := json.Marshal(typeURL)
			if err != nil {
				return fmt.Errorf("failed to marshal type URL %q to JSON: %v", typeURL, err)
			}
			js["@type"] = (*json.RawMessage)(&turl)
			if b, err = json.Marshal(js); err != nil {
				return err
			}
		}
		w.write(string(b))
		return nil
	}

	md := m.Descriptor()
	fds := md.Fields()

	// Handle well-known types.
	const secondInNanos = int64(time.Second / time.Nanosecond)
	switch wellKnownType(md.FullName()) {
	case "Any":
		return w.marshalAny(m, indent)
	case "BoolValue", "BytesValue", "StringValue",
		"Int32Value", "UInt32Value", "FloatValue",
		"Int64Value", "UInt64Value", "DoubleValue":
		fd := fds.ByNumber(1)
		return w.marshalValue(fd, m.Get(fd), indent)
	case "Duration":
		// "Generated output always contains 0, 3, 6, or 9 fractional digits,
		//  depending on required precision."
		s := m.Get(fds.ByNumber(1)).Int()
		ns := m.Get(fds.ByNumber(2)).Int()
		if ns <= -secondInNanos || ns >= secondInNanos {
			return fmt.Errorf("ns out of range (%v, %v)", -secondInNanos, secondInNanos)
		}
		if (s > 0 && ns < 0) || (s < 0 && ns > 0) {
			return errors.New("signs of seconds and nanos do not match")
		}
		if s < 0 {
			ns = -ns
		}
		x := fmt.Sprintf("%d.%09d", s, ns)
		x = strings.TrimSuffix(x, "000")
		x = strings.TrimSuffix(x, "000")
		x = strings.TrimSuffix(x, ".000")
		w.write(fmt.Sprintf(`"%vs"`, x))
		return nil
	case "Timestamp":
		// "RFC 3339, where generated output will always be Z-normalized
		//  and uses 0, 3, 6 or 9 fractional digits."
		s := m.Get(fds.ByNumber(1)).Int()
		ns := m.Get(fds.ByNumber(2)).Int()
		if ns < 0 || ns >= secondInNanos {
			return fmt.Errorf("ns out of range [0, %v)", secondInNanos)
		}
		t := time.Unix(s, ns).UTC()
		// time.RFC3339Nano isn't exactly right (we need to get 3/6/9 fractional digits).
		x := t.Format("2006-01-02T15:04:05.000000000")
		x = strings.TrimSuffix(x, "000")
		x = strings.TrimSuffix(x, "000")
		x = strings.TrimSuffix(x, ".000")
		w.write(fmt.Sprintf(`"%vZ"`, x))
		return nil
	case "Value":
		// JSON value; which is a null, number, string, bool, object, or array.
		od := md.Oneofs().Get(0)
		fd := m.WhichOneof(od)
		if fd == nil {
			return errors.New("nil Value")
		}
		return w.marshalValue(fd, m.Get(fd), indent)
	case "Struct", "ListValue":
		// JSON object or array.
		fd := fds.ByNumber(1)
		return w.marshalValue(fd, m.Get(fd), indent)
	}

	w.write("{")
	if w.Indent != "" {
		w.write("\n")
	}

	firstField := true
	if typeURL != "" {
		if err := w.marshalTypeURL(indent, typeURL); err != nil {
			return err
		}
		firstField = false
	}

	for i := 0; i < fds.Len(); {
		fd := fds.Get(i)
		if od := fd.ContainingOneof(); od != nil {
			fd = m.WhichOneof(od)
			i += od.Fields().Len()
			if fd == nil {
				continue
			}
		} else {
			i++
		}

		v := m.Get(fd)

		if !m.Has(fd) {
			if !w.EmitDefaults || fd.ContainingOneof() != nil {
				continue
			}
			if fd.Cardinality() != protoreflect.Repeated && (fd.Message() != nil || fd.Syntax() == protoreflect.Proto2) {
				v = protoreflect.Value{} // use "null" for singular messages or proto2 scalars
			}
		}

		if !firstField {
			w.writeComma()
		}
		if err := w.marshalField(fd, v, indent); err != nil {
			return err
		}
		firstField = false
	}

	// Handle proto2 extensions.
	if md.ExtensionRanges().Len() > 0 {
		// Collect a sorted list of all extension descriptor and values.
		type ext struct {
			desc protoreflect.FieldDescriptor
			val  protoreflect.Value
		}
		var exts []ext
		m.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
			if fd.IsExtension() {
				exts = append(exts, ext{fd, v})
			}
			return true
		})
		sort.Slice(exts, func(i, j int) bool {
			return exts[i].desc.Number() < exts[j].desc.Number()
		})

		for _, ext := range exts {
			if !firstField {
				w.writeComma()
			}
			if err := w.marshalField(ext.desc, ext.val, indent); err != nil {
				return err
			}
			firstField = false
		}
	}

	if w.Indent != "" {
		w.write("\n")
		w.write(indent)
	}
	w.write("}")
	return nil
}

func (w *jsonWriter) writeComma() {
	if w.Indent != "" {
		w.write(",\n")
	} else {
		w.write(",")
	}
}

func (w *jsonWriter) marshalAny(m protoreflect.Message, indent string) error {
	// "If the Any contains a value that has a special JSON mapping,
	//  it will be converted as follows: {"@type": xxx, "value": yyy}.
	//  Otherwise, the value will be converted into a JSON object,
	//  and the "@type" field will be inserted to indicate the actual data type."
	md := m.Descriptor()
	typeURL := m.Get(md.Fields().ByNumber(1)).String()
	rawVal := m.Get(md.Fields().ByNumber(2)).Bytes()

	var m2 protoreflect.Message
	if w.AnyResolver != nil {
		mi, err := w.AnyResolver.Resolve(typeURL)
		if err != nil {
			return err
		}
		m2 = proto.MessageReflect(mi)
	} else {
		mt, err := protoregistry.GlobalTypes.FindMessageByURL(typeURL)
		if err != nil {
			return err
		}
		m2 = mt.New()
	}

	if err := protoV2.Unmarshal(rawVal, m2.Interface()); err != nil {
		return err
	}

	if wellKnownType(m2.Descriptor().FullName()) == "" {
		return w.marshalMessage(m2, indent, typeURL)
	}

	w.write("{")
	if w.Indent != "" {
		w.write("\n")
	}
	if err := w.marshalTypeURL(indent, typeURL); err != nil {
		return err
	}
	w.writeComma()
	if w.Indent != "" {
		w.write(indent)
		w.write(w.Indent)
		w.write(`"value": `)
	} else {
		w.write(`"value":`)
	}
	if err := w.marshalMessage(m2, indent+w.Indent, ""); err != nil {
		return err
	}
	if w.Indent != "" {
		w.write("\n")
		w.write(indent)
	}
	w.write("}")
	return nil
}

func (w *jsonWriter) marshalTypeURL(indent, typeURL string) error {
	if w.Indent != "" {
		w.write(indent)
		w.write(w.Indent)
	}
	w.write(`"@type":`)
	if w.Indent != "" {
		w.write(" ")
	}
	b, err := json.Marshal(typeURL)
	if err != nil {
		return err
	}
	w.write(string(b))
	return nil
}

// marshalField writes field description and value to the Writer.
func (w *jsonWriter) marshalField(fd protoreflect.FieldDescriptor, v protoreflect.Value, indent string) error {
	if w.Indent != "" {
		w.write(indent)
		w.write(w.Indent)
	}
	w.write(`"`)
	switch {
	case fd.IsExtension():
		// For message set, use the fname of the message as the extension name.
		name := string(fd.FullName())
		if isMessageSet(fd.ContainingMessage()) {
			name = strings.TrimSuffix(name, ".message_set_extension")
		}

		w.write("[" + name + "]")
	case w.OrigName:
		name := string(fd.Name())
		if fd.Kind() == protoreflect.GroupKind {
			name = string(fd.Message().Name())
		}
		w.write(name)
	default:
		w.write(string(fd.JSONName()))
	}
	w.write(`":`)
	if w.Indent != "" {
		w.write(" ")
	}
	return w.marshalValue(fd, v, indent)
}

func (w *jsonWriter) marshalValue(fd protoreflect.FieldDescriptor, v protoreflect.Value, indent string) error {
	switch {
	case fd.IsList():
		w.write("[")
		comma := ""
		lv := v.List()
		for i := 0; i < lv.Len(); i++ {
			w.write(comma)
			if w.Indent != "" {
				w.write("\n")
				w.write(indent)
				w.write(w.Indent)
				w.write(w.Indent)
			}
			if err := w.marshalSingularValue(fd, lv.Get(i), indent+w.Indent); err != nil {
				return err
			}
			comma = ","
		}
		if w.Indent != "" {
			w.write("\n")
			w.write(indent)
			w.write(w.Indent)
		}
		w.write("]")
		return nil
	case fd.IsMap():
		kfd := fd.MapKey()
		vfd := fd.MapValue()
		mv := v.Map()

		// Collect a sorted list of all map keys and values.
		type entry struct{ key, val protoreflect.Value }
		var entries []entry
		mv.Range(func(k protoreflect.MapKey, v protoreflect.Value) bool {
			entries = append(entries, entry{k.Value(), v})
			return true
		})
		sort.Slice(entries, func(i, j int) bool {
			switch kfd.Kind() {
			case protoreflect.BoolKind:
				return !entries[i].key.Bool() && entries[j].key.Bool()
			case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Sfixed32Kind, protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind:
				return entries[i].key.Int() < entries[j].key.Int()
			case protoreflect.Uint32Kind, protoreflect.Fixed32Kind, protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
				return entries[i].key.Uint() < entries[j].key.Uint()
			case protoreflect.StringKind:
				return entries[i].key.String() < entries[j].key.String()
			default:
				panic("invalid kind")
			}
		})

		w.write(`{`)
		comma := ""
		for _, entry := range entries {
			w.write(comma)
			if w.Indent != "" {
				w.write("\n")
				w.write(indent)
				w.write(w.Indent)
				w.write(w.Indent)
			}

			s := fmt.Sprint(entry.key.Interface())
			b, err := json.Marshal(s)
			if err != nil {
				return err
			}
			w.write(string(b))

			w.write(`:`)
			if w.Indent != "" {
				w.write(` `)
			}

			if err := w.marshalSingularValue(vfd, entry.val, indent+w.Indent); err != nil {
				return err
			}
			comma = ","
		}
		if w.Indent != "" {
			w.write("\n")
			w.write(indent)
			w.write(w.Indent)
		}
		w.write(`}`)
		return nil
	default:
		return w.marshalSingularValue(fd, v, indent)
	}
}

func (w *jsonWriter) marshalSingularValue(fd protoreflect.FieldDescriptor, v protoreflect.Value, indent string) error {
	switch {
	case !v.IsValid():
		w.write("null")
		return nil
	case fd.Message() != nil:
		return w.marshalMessage(v.Message(), indent+w.Indent, "")
	case fd.Enum() != nil:
		if fd.Enum().FullName() == "google.protobuf.NullValue" {
			w.write("null")
			return nil
		}

		vd := fd.Enum().Values().ByNumber(v.Enum())
		if vd == nil || w.EnumsAsInts {
			w.write(strconv.Itoa(int(v.Enum())))
		} else {
			w.write(`"` + string(vd.Name()) + `"`)
		}
		return nil
	default:
		switch v.Interface().(type) {
		case float32, float64:
			switch {
			case math.IsInf(v.Float(), +1):
				w.write(`"Infinity"`)
				return nil
			case math.IsInf(v.Float(), -1):
				w.write(`"-Infinity"`)
				return nil
			case math.IsNaN(v.Float()):
				w.write(`"NaN"`)
				return nil
			}
		case int64, uint64:
			w.write(fmt.Sprintf(`"%d"`, v.Interface()))
			return nil
		}

		b, err := json.Marshal(v.Interface())
		if err != nil {
			return err
		}
		w.write(string(b))
		return nil
	}
}
