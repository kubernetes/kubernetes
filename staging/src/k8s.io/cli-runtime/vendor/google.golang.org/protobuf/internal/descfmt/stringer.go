// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package descfmt provides functionality to format descriptors.
package descfmt

import (
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"

	"google.golang.org/protobuf/internal/detrand"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/reflect/protoreflect"
)

type list interface {
	Len() int
	pragma.DoNotImplement
}

func FormatList(s fmt.State, r rune, vs list) {
	io.WriteString(s, formatListOpt(vs, true, r == 'v' && (s.Flag('+') || s.Flag('#'))))
}
func formatListOpt(vs list, isRoot, allowMulti bool) string {
	start, end := "[", "]"
	if isRoot {
		var name string
		switch vs.(type) {
		case protoreflect.Names:
			name = "Names"
		case protoreflect.FieldNumbers:
			name = "FieldNumbers"
		case protoreflect.FieldRanges:
			name = "FieldRanges"
		case protoreflect.EnumRanges:
			name = "EnumRanges"
		case protoreflect.FileImports:
			name = "FileImports"
		case protoreflect.Descriptor:
			name = reflect.ValueOf(vs).MethodByName("Get").Type().Out(0).Name() + "s"
		default:
			name = reflect.ValueOf(vs).Elem().Type().Name()
		}
		start, end = name+"{", "}"
	}

	var ss []string
	switch vs := vs.(type) {
	case protoreflect.Names:
		for i := 0; i < vs.Len(); i++ {
			ss = append(ss, fmt.Sprint(vs.Get(i)))
		}
		return start + joinStrings(ss, false) + end
	case protoreflect.FieldNumbers:
		for i := 0; i < vs.Len(); i++ {
			ss = append(ss, fmt.Sprint(vs.Get(i)))
		}
		return start + joinStrings(ss, false) + end
	case protoreflect.FieldRanges:
		for i := 0; i < vs.Len(); i++ {
			r := vs.Get(i)
			if r[0]+1 == r[1] {
				ss = append(ss, fmt.Sprintf("%d", r[0]))
			} else {
				ss = append(ss, fmt.Sprintf("%d:%d", r[0], r[1])) // enum ranges are end exclusive
			}
		}
		return start + joinStrings(ss, false) + end
	case protoreflect.EnumRanges:
		for i := 0; i < vs.Len(); i++ {
			r := vs.Get(i)
			if r[0] == r[1] {
				ss = append(ss, fmt.Sprintf("%d", r[0]))
			} else {
				ss = append(ss, fmt.Sprintf("%d:%d", r[0], int64(r[1])+1)) // enum ranges are end inclusive
			}
		}
		return start + joinStrings(ss, false) + end
	case protoreflect.FileImports:
		for i := 0; i < vs.Len(); i++ {
			var rs records
			rs.Append(reflect.ValueOf(vs.Get(i)), "Path", "Package", "IsPublic", "IsWeak")
			ss = append(ss, "{"+rs.Join()+"}")
		}
		return start + joinStrings(ss, allowMulti) + end
	default:
		_, isEnumValue := vs.(protoreflect.EnumValueDescriptors)
		for i := 0; i < vs.Len(); i++ {
			m := reflect.ValueOf(vs).MethodByName("Get")
			v := m.Call([]reflect.Value{reflect.ValueOf(i)})[0].Interface()
			ss = append(ss, formatDescOpt(v.(protoreflect.Descriptor), false, allowMulti && !isEnumValue))
		}
		return start + joinStrings(ss, allowMulti && isEnumValue) + end
	}
}

// descriptorAccessors is a list of accessors to print for each descriptor.
//
// Do not print all accessors since some contain redundant information,
// while others are pointers that we do not want to follow since the descriptor
// is actually a cyclic graph.
//
// Using a list allows us to print the accessors in a sensible order.
var descriptorAccessors = map[reflect.Type][]string{
	reflect.TypeOf((*protoreflect.FileDescriptor)(nil)).Elem():      {"Path", "Package", "Imports", "Messages", "Enums", "Extensions", "Services"},
	reflect.TypeOf((*protoreflect.MessageDescriptor)(nil)).Elem():   {"IsMapEntry", "Fields", "Oneofs", "ReservedNames", "ReservedRanges", "RequiredNumbers", "ExtensionRanges", "Messages", "Enums", "Extensions"},
	reflect.TypeOf((*protoreflect.FieldDescriptor)(nil)).Elem():     {"Number", "Cardinality", "Kind", "HasJSONName", "JSONName", "HasPresence", "IsExtension", "IsPacked", "IsWeak", "IsList", "IsMap", "MapKey", "MapValue", "HasDefault", "Default", "ContainingOneof", "ContainingMessage", "Message", "Enum"},
	reflect.TypeOf((*protoreflect.OneofDescriptor)(nil)).Elem():     {"Fields"}, // not directly used; must keep in sync with formatDescOpt
	reflect.TypeOf((*protoreflect.EnumDescriptor)(nil)).Elem():      {"Values", "ReservedNames", "ReservedRanges"},
	reflect.TypeOf((*protoreflect.EnumValueDescriptor)(nil)).Elem(): {"Number"},
	reflect.TypeOf((*protoreflect.ServiceDescriptor)(nil)).Elem():   {"Methods"},
	reflect.TypeOf((*protoreflect.MethodDescriptor)(nil)).Elem():    {"Input", "Output", "IsStreamingClient", "IsStreamingServer"},
}

func FormatDesc(s fmt.State, r rune, t protoreflect.Descriptor) {
	io.WriteString(s, formatDescOpt(t, true, r == 'v' && (s.Flag('+') || s.Flag('#'))))
}
func formatDescOpt(t protoreflect.Descriptor, isRoot, allowMulti bool) string {
	rv := reflect.ValueOf(t)
	rt := rv.MethodByName("ProtoType").Type().In(0)

	start, end := "{", "}"
	if isRoot {
		start = rt.Name() + "{"
	}

	_, isFile := t.(protoreflect.FileDescriptor)
	rs := records{allowMulti: allowMulti}
	if t.IsPlaceholder() {
		if isFile {
			rs.Append(rv, "Path", "Package", "IsPlaceholder")
		} else {
			rs.Append(rv, "FullName", "IsPlaceholder")
		}
	} else {
		switch {
		case isFile:
			rs.Append(rv, "Syntax")
		case isRoot:
			rs.Append(rv, "Syntax", "FullName")
		default:
			rs.Append(rv, "Name")
		}
		switch t := t.(type) {
		case protoreflect.FieldDescriptor:
			for _, s := range descriptorAccessors[rt] {
				switch s {
				case "MapKey":
					if k := t.MapKey(); k != nil {
						rs.recs = append(rs.recs, [2]string{"MapKey", k.Kind().String()})
					}
				case "MapValue":
					if v := t.MapValue(); v != nil {
						switch v.Kind() {
						case protoreflect.EnumKind:
							rs.recs = append(rs.recs, [2]string{"MapValue", string(v.Enum().FullName())})
						case protoreflect.MessageKind, protoreflect.GroupKind:
							rs.recs = append(rs.recs, [2]string{"MapValue", string(v.Message().FullName())})
						default:
							rs.recs = append(rs.recs, [2]string{"MapValue", v.Kind().String()})
						}
					}
				case "ContainingOneof":
					if od := t.ContainingOneof(); od != nil {
						rs.recs = append(rs.recs, [2]string{"Oneof", string(od.Name())})
					}
				case "ContainingMessage":
					if t.IsExtension() {
						rs.recs = append(rs.recs, [2]string{"Extendee", string(t.ContainingMessage().FullName())})
					}
				case "Message":
					if !t.IsMap() {
						rs.Append(rv, s)
					}
				default:
					rs.Append(rv, s)
				}
			}
		case protoreflect.OneofDescriptor:
			var ss []string
			fs := t.Fields()
			for i := 0; i < fs.Len(); i++ {
				ss = append(ss, string(fs.Get(i).Name()))
			}
			if len(ss) > 0 {
				rs.recs = append(rs.recs, [2]string{"Fields", "[" + joinStrings(ss, false) + "]"})
			}
		default:
			rs.Append(rv, descriptorAccessors[rt]...)
		}
		if rv.MethodByName("GoType").IsValid() {
			rs.Append(rv, "GoType")
		}
	}
	return start + rs.Join() + end
}

type records struct {
	recs       [][2]string
	allowMulti bool
}

func (rs *records) Append(v reflect.Value, accessors ...string) {
	for _, a := range accessors {
		var rv reflect.Value
		if m := v.MethodByName(a); m.IsValid() {
			rv = m.Call(nil)[0]
		}
		if v.Kind() == reflect.Struct && !rv.IsValid() {
			rv = v.FieldByName(a)
		}
		if !rv.IsValid() {
			panic(fmt.Sprintf("unknown accessor: %v.%s", v.Type(), a))
		}
		if _, ok := rv.Interface().(protoreflect.Value); ok {
			rv = rv.MethodByName("Interface").Call(nil)[0]
			if !rv.IsNil() {
				rv = rv.Elem()
			}
		}

		// Ignore zero values.
		var isZero bool
		switch rv.Kind() {
		case reflect.Interface, reflect.Slice:
			isZero = rv.IsNil()
		case reflect.Bool:
			isZero = rv.Bool() == false
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			isZero = rv.Int() == 0
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			isZero = rv.Uint() == 0
		case reflect.String:
			isZero = rv.String() == ""
		}
		if n, ok := rv.Interface().(list); ok {
			isZero = n.Len() == 0
		}
		if isZero {
			continue
		}

		// Format the value.
		var s string
		v := rv.Interface()
		switch v := v.(type) {
		case list:
			s = formatListOpt(v, false, rs.allowMulti)
		case protoreflect.FieldDescriptor, protoreflect.OneofDescriptor, protoreflect.EnumValueDescriptor, protoreflect.MethodDescriptor:
			s = string(v.(protoreflect.Descriptor).Name())
		case protoreflect.Descriptor:
			s = string(v.FullName())
		case string:
			s = strconv.Quote(v)
		case []byte:
			s = fmt.Sprintf("%q", v)
		default:
			s = fmt.Sprint(v)
		}
		rs.recs = append(rs.recs, [2]string{a, s})
	}
}

func (rs *records) Join() string {
	var ss []string

	// In single line mode, simply join all records with commas.
	if !rs.allowMulti {
		for _, r := range rs.recs {
			ss = append(ss, r[0]+formatColon(0)+r[1])
		}
		return joinStrings(ss, false)
	}

	// In allowMulti line mode, align single line records for more readable output.
	var maxLen int
	flush := func(i int) {
		for _, r := range rs.recs[len(ss):i] {
			ss = append(ss, r[0]+formatColon(maxLen-len(r[0]))+r[1])
		}
		maxLen = 0
	}
	for i, r := range rs.recs {
		if isMulti := strings.Contains(r[1], "\n"); isMulti {
			flush(i)
			ss = append(ss, r[0]+formatColon(0)+strings.Join(strings.Split(r[1], "\n"), "\n\t"))
		} else if maxLen < len(r[0]) {
			maxLen = len(r[0])
		}
	}
	flush(len(rs.recs))
	return joinStrings(ss, true)
}

func formatColon(padding int) string {
	// Deliberately introduce instability into the debug output to
	// discourage users from performing string comparisons.
	// This provides us flexibility to change the output in the future.
	if detrand.Bool() {
		return ":" + strings.Repeat(" ", 1+padding) // use non-breaking spaces (U+00a0)
	} else {
		return ":" + strings.Repeat(" ", 1+padding) // use regular spaces (U+0020)
	}
}

func joinStrings(ss []string, isMulti bool) string {
	if len(ss) == 0 {
		return ""
	}
	if isMulti {
		return "\n\t" + strings.Join(ss, "\n\t") + "\n"
	}
	return strings.Join(ss, ", ")
}
