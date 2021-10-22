// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package protobuild constructs messages.
//
// This package is used to construct multiple types of message with a similar shape
// from a common template.
package protobuild

import (
	"fmt"
	"math"
	"reflect"

	pref "google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

// A Value is a value assignable to a field.
// A Value may be a value accepted by protoreflect.ValueOf. In addition:
//
// • An int may be assigned to any numeric field.
//
// • A float64 may be assigned to a double field.
//
// • Either a string or []byte may be assigned to a string or bytes field.
//
// • A string containing the value name may be assigned to an enum field.
//
// • A slice may be assigned to a list, and a map may be assigned to a map.
type Value interface{}

// A Message is a template to apply to a message. Keys are field names, including
// extension names.
type Message map[pref.Name]Value

// Unknown is a key associated with the unknown fields of a message.
// The value should be a []byte.
const Unknown = "@unknown"

// Build applies the template to a message.
func (template Message) Build(m pref.Message) {
	md := m.Descriptor()
	fields := md.Fields()
	exts := make(map[pref.Name]pref.FieldDescriptor)
	protoregistry.GlobalTypes.RangeExtensionsByMessage(md.FullName(), func(xt pref.ExtensionType) bool {
		xd := xt.TypeDescriptor()
		exts[xd.Name()] = xd
		return true
	})
	for k, v := range template {
		if k == Unknown {
			m.SetUnknown(pref.RawFields(v.([]byte)))
			continue
		}
		fd := fields.ByName(k)
		if fd == nil {
			fd = exts[k]
		}
		if fd == nil {
			panic(fmt.Sprintf("%v.%v: not found", md.FullName(), k))
		}
		switch {
		case fd.IsList():
			list := m.Mutable(fd).List()
			s := reflect.ValueOf(v)
			for i := 0; i < s.Len(); i++ {
				if fd.Message() == nil {
					list.Append(fieldValue(fd, s.Index(i).Interface()))
				} else {
					e := list.NewElement()
					s.Index(i).Interface().(Message).Build(e.Message())
					list.Append(e)
				}
			}
		case fd.IsMap():
			mapv := m.Mutable(fd).Map()
			rm := reflect.ValueOf(v)
			for _, k := range rm.MapKeys() {
				mk := fieldValue(fd.MapKey(), k.Interface()).MapKey()
				if fd.MapValue().Message() == nil {
					mv := fieldValue(fd.MapValue(), rm.MapIndex(k).Interface())
					mapv.Set(mk, mv)
				} else if mapv.Has(mk) {
					mv := mapv.Get(mk).Message()
					rm.MapIndex(k).Interface().(Message).Build(mv)
				} else {
					mv := mapv.NewValue()
					rm.MapIndex(k).Interface().(Message).Build(mv.Message())
					mapv.Set(mk, mv)
				}
			}
		default:
			if fd.Message() == nil {
				m.Set(fd, fieldValue(fd, v))
			} else {
				v.(Message).Build(m.Mutable(fd).Message())
			}
		}
	}
}

func fieldValue(fd pref.FieldDescriptor, v interface{}) pref.Value {
	switch o := v.(type) {
	case int:
		switch fd.Kind() {
		case pref.Int32Kind, pref.Sint32Kind, pref.Sfixed32Kind:
			if o < math.MinInt32 || math.MaxInt32 < o {
				panic(fmt.Sprintf("%v: value %v out of range [%v, %v]", fd.FullName(), o, int32(math.MinInt32), int32(math.MaxInt32)))
			}
			v = int32(o)
		case pref.Uint32Kind, pref.Fixed32Kind:
			if o < 0 || math.MaxUint32 < 0 {
				panic(fmt.Sprintf("%v: value %v out of range [%v, %v]", fd.FullName(), o, uint32(0), uint32(math.MaxUint32)))
			}
			v = uint32(o)
		case pref.Int64Kind, pref.Sint64Kind, pref.Sfixed64Kind:
			v = int64(o)
		case pref.Uint64Kind, pref.Fixed64Kind:
			if o < 0 {
				panic(fmt.Sprintf("%v: value %v out of range [%v, %v]", fd.FullName(), o, uint64(0), uint64(math.MaxUint64)))
			}
			v = uint64(o)
		case pref.FloatKind:
			v = float32(o)
		case pref.DoubleKind:
			v = float64(o)
		case pref.EnumKind:
			v = pref.EnumNumber(o)
		default:
			panic(fmt.Sprintf("%v: invalid value type int", fd.FullName()))
		}
	case float64:
		switch fd.Kind() {
		case pref.FloatKind:
			v = float32(o)
		}
	case string:
		switch fd.Kind() {
		case pref.BytesKind:
			v = []byte(o)
		case pref.EnumKind:
			v = fd.Enum().Values().ByName(pref.Name(o)).Number()
		}
	case []byte:
		return pref.ValueOf(append([]byte{}, o...))
	}
	return pref.ValueOf(v)
}
