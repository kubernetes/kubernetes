// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"fmt"
	"reflect"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/strs"
	pref "google.golang.org/protobuf/reflect/protoreflect"
)

// pointerCoderFuncs is a set of pointer encoding functions.
type pointerCoderFuncs struct {
	mi        *MessageInfo
	size      func(p pointer, f *coderFieldInfo, opts marshalOptions) int
	marshal   func(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error)
	unmarshal func(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (unmarshalOutput, error)
	isInit    func(p pointer, f *coderFieldInfo) error
	merge     func(dst, src pointer, f *coderFieldInfo, opts mergeOptions)
}

// valueCoderFuncs is a set of protoreflect.Value encoding functions.
type valueCoderFuncs struct {
	size      func(v pref.Value, tagsize int, opts marshalOptions) int
	marshal   func(b []byte, v pref.Value, wiretag uint64, opts marshalOptions) ([]byte, error)
	unmarshal func(b []byte, v pref.Value, num protowire.Number, wtyp protowire.Type, opts unmarshalOptions) (pref.Value, unmarshalOutput, error)
	isInit    func(v pref.Value) error
	merge     func(dst, src pref.Value, opts mergeOptions) pref.Value
}

// fieldCoder returns pointer functions for a field, used for operating on
// struct fields.
func fieldCoder(fd pref.FieldDescriptor, ft reflect.Type) (*MessageInfo, pointerCoderFuncs) {
	switch {
	case fd.IsMap():
		return encoderFuncsForMap(fd, ft)
	case fd.Cardinality() == pref.Repeated && !fd.IsPacked():
		// Repeated fields (not packed).
		if ft.Kind() != reflect.Slice {
			break
		}
		ft := ft.Elem()
		switch fd.Kind() {
		case pref.BoolKind:
			if ft.Kind() == reflect.Bool {
				return nil, coderBoolSlice
			}
		case pref.EnumKind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderEnumSlice
			}
		case pref.Int32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderInt32Slice
			}
		case pref.Sint32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSint32Slice
			}
		case pref.Uint32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderUint32Slice
			}
		case pref.Int64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderInt64Slice
			}
		case pref.Sint64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSint64Slice
			}
		case pref.Uint64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderUint64Slice
			}
		case pref.Sfixed32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSfixed32Slice
			}
		case pref.Fixed32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderFixed32Slice
			}
		case pref.FloatKind:
			if ft.Kind() == reflect.Float32 {
				return nil, coderFloatSlice
			}
		case pref.Sfixed64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSfixed64Slice
			}
		case pref.Fixed64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderFixed64Slice
			}
		case pref.DoubleKind:
			if ft.Kind() == reflect.Float64 {
				return nil, coderDoubleSlice
			}
		case pref.StringKind:
			if ft.Kind() == reflect.String && strs.EnforceUTF8(fd) {
				return nil, coderStringSliceValidateUTF8
			}
			if ft.Kind() == reflect.String {
				return nil, coderStringSlice
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 && strs.EnforceUTF8(fd) {
				return nil, coderBytesSliceValidateUTF8
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 {
				return nil, coderBytesSlice
			}
		case pref.BytesKind:
			if ft.Kind() == reflect.String {
				return nil, coderStringSlice
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 {
				return nil, coderBytesSlice
			}
		case pref.MessageKind:
			return getMessageInfo(ft), makeMessageSliceFieldCoder(fd, ft)
		case pref.GroupKind:
			return getMessageInfo(ft), makeGroupSliceFieldCoder(fd, ft)
		}
	case fd.Cardinality() == pref.Repeated && fd.IsPacked():
		// Packed repeated fields.
		//
		// Only repeated fields of primitive numeric types
		// (Varint, Fixed32, or Fixed64 wire type) can be packed.
		if ft.Kind() != reflect.Slice {
			break
		}
		ft := ft.Elem()
		switch fd.Kind() {
		case pref.BoolKind:
			if ft.Kind() == reflect.Bool {
				return nil, coderBoolPackedSlice
			}
		case pref.EnumKind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderEnumPackedSlice
			}
		case pref.Int32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderInt32PackedSlice
			}
		case pref.Sint32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSint32PackedSlice
			}
		case pref.Uint32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderUint32PackedSlice
			}
		case pref.Int64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderInt64PackedSlice
			}
		case pref.Sint64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSint64PackedSlice
			}
		case pref.Uint64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderUint64PackedSlice
			}
		case pref.Sfixed32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSfixed32PackedSlice
			}
		case pref.Fixed32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderFixed32PackedSlice
			}
		case pref.FloatKind:
			if ft.Kind() == reflect.Float32 {
				return nil, coderFloatPackedSlice
			}
		case pref.Sfixed64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSfixed64PackedSlice
			}
		case pref.Fixed64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderFixed64PackedSlice
			}
		case pref.DoubleKind:
			if ft.Kind() == reflect.Float64 {
				return nil, coderDoublePackedSlice
			}
		}
	case fd.Kind() == pref.MessageKind:
		return getMessageInfo(ft), makeMessageFieldCoder(fd, ft)
	case fd.Kind() == pref.GroupKind:
		return getMessageInfo(ft), makeGroupFieldCoder(fd, ft)
	case fd.Syntax() == pref.Proto3 && fd.ContainingOneof() == nil:
		// Populated oneof fields always encode even if set to the zero value,
		// which normally are not encoded in proto3.
		switch fd.Kind() {
		case pref.BoolKind:
			if ft.Kind() == reflect.Bool {
				return nil, coderBoolNoZero
			}
		case pref.EnumKind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderEnumNoZero
			}
		case pref.Int32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderInt32NoZero
			}
		case pref.Sint32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSint32NoZero
			}
		case pref.Uint32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderUint32NoZero
			}
		case pref.Int64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderInt64NoZero
			}
		case pref.Sint64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSint64NoZero
			}
		case pref.Uint64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderUint64NoZero
			}
		case pref.Sfixed32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSfixed32NoZero
			}
		case pref.Fixed32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderFixed32NoZero
			}
		case pref.FloatKind:
			if ft.Kind() == reflect.Float32 {
				return nil, coderFloatNoZero
			}
		case pref.Sfixed64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSfixed64NoZero
			}
		case pref.Fixed64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderFixed64NoZero
			}
		case pref.DoubleKind:
			if ft.Kind() == reflect.Float64 {
				return nil, coderDoubleNoZero
			}
		case pref.StringKind:
			if ft.Kind() == reflect.String && strs.EnforceUTF8(fd) {
				return nil, coderStringNoZeroValidateUTF8
			}
			if ft.Kind() == reflect.String {
				return nil, coderStringNoZero
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 && strs.EnforceUTF8(fd) {
				return nil, coderBytesNoZeroValidateUTF8
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 {
				return nil, coderBytesNoZero
			}
		case pref.BytesKind:
			if ft.Kind() == reflect.String {
				return nil, coderStringNoZero
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 {
				return nil, coderBytesNoZero
			}
		}
	case ft.Kind() == reflect.Ptr:
		ft := ft.Elem()
		switch fd.Kind() {
		case pref.BoolKind:
			if ft.Kind() == reflect.Bool {
				return nil, coderBoolPtr
			}
		case pref.EnumKind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderEnumPtr
			}
		case pref.Int32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderInt32Ptr
			}
		case pref.Sint32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSint32Ptr
			}
		case pref.Uint32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderUint32Ptr
			}
		case pref.Int64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderInt64Ptr
			}
		case pref.Sint64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSint64Ptr
			}
		case pref.Uint64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderUint64Ptr
			}
		case pref.Sfixed32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSfixed32Ptr
			}
		case pref.Fixed32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderFixed32Ptr
			}
		case pref.FloatKind:
			if ft.Kind() == reflect.Float32 {
				return nil, coderFloatPtr
			}
		case pref.Sfixed64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSfixed64Ptr
			}
		case pref.Fixed64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderFixed64Ptr
			}
		case pref.DoubleKind:
			if ft.Kind() == reflect.Float64 {
				return nil, coderDoublePtr
			}
		case pref.StringKind:
			if ft.Kind() == reflect.String && strs.EnforceUTF8(fd) {
				return nil, coderStringPtrValidateUTF8
			}
			if ft.Kind() == reflect.String {
				return nil, coderStringPtr
			}
		case pref.BytesKind:
			if ft.Kind() == reflect.String {
				return nil, coderStringPtr
			}
		}
	default:
		switch fd.Kind() {
		case pref.BoolKind:
			if ft.Kind() == reflect.Bool {
				return nil, coderBool
			}
		case pref.EnumKind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderEnum
			}
		case pref.Int32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderInt32
			}
		case pref.Sint32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSint32
			}
		case pref.Uint32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderUint32
			}
		case pref.Int64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderInt64
			}
		case pref.Sint64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSint64
			}
		case pref.Uint64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderUint64
			}
		case pref.Sfixed32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSfixed32
			}
		case pref.Fixed32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderFixed32
			}
		case pref.FloatKind:
			if ft.Kind() == reflect.Float32 {
				return nil, coderFloat
			}
		case pref.Sfixed64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSfixed64
			}
		case pref.Fixed64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderFixed64
			}
		case pref.DoubleKind:
			if ft.Kind() == reflect.Float64 {
				return nil, coderDouble
			}
		case pref.StringKind:
			if ft.Kind() == reflect.String && strs.EnforceUTF8(fd) {
				return nil, coderStringValidateUTF8
			}
			if ft.Kind() == reflect.String {
				return nil, coderString
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 && strs.EnforceUTF8(fd) {
				return nil, coderBytesValidateUTF8
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 {
				return nil, coderBytes
			}
		case pref.BytesKind:
			if ft.Kind() == reflect.String {
				return nil, coderString
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 {
				return nil, coderBytes
			}
		}
	}
	panic(fmt.Sprintf("invalid type: no encoder for %v %v %v/%v", fd.FullName(), fd.Cardinality(), fd.Kind(), ft))
}

// encoderFuncsForValue returns value functions for a field, used for
// extension values and map encoding.
func encoderFuncsForValue(fd pref.FieldDescriptor) valueCoderFuncs {
	switch {
	case fd.Cardinality() == pref.Repeated && !fd.IsPacked():
		switch fd.Kind() {
		case pref.BoolKind:
			return coderBoolSliceValue
		case pref.EnumKind:
			return coderEnumSliceValue
		case pref.Int32Kind:
			return coderInt32SliceValue
		case pref.Sint32Kind:
			return coderSint32SliceValue
		case pref.Uint32Kind:
			return coderUint32SliceValue
		case pref.Int64Kind:
			return coderInt64SliceValue
		case pref.Sint64Kind:
			return coderSint64SliceValue
		case pref.Uint64Kind:
			return coderUint64SliceValue
		case pref.Sfixed32Kind:
			return coderSfixed32SliceValue
		case pref.Fixed32Kind:
			return coderFixed32SliceValue
		case pref.FloatKind:
			return coderFloatSliceValue
		case pref.Sfixed64Kind:
			return coderSfixed64SliceValue
		case pref.Fixed64Kind:
			return coderFixed64SliceValue
		case pref.DoubleKind:
			return coderDoubleSliceValue
		case pref.StringKind:
			// We don't have a UTF-8 validating coder for repeated string fields.
			// Value coders are used for extensions and maps.
			// Extensions are never proto3, and maps never contain lists.
			return coderStringSliceValue
		case pref.BytesKind:
			return coderBytesSliceValue
		case pref.MessageKind:
			return coderMessageSliceValue
		case pref.GroupKind:
			return coderGroupSliceValue
		}
	case fd.Cardinality() == pref.Repeated && fd.IsPacked():
		switch fd.Kind() {
		case pref.BoolKind:
			return coderBoolPackedSliceValue
		case pref.EnumKind:
			return coderEnumPackedSliceValue
		case pref.Int32Kind:
			return coderInt32PackedSliceValue
		case pref.Sint32Kind:
			return coderSint32PackedSliceValue
		case pref.Uint32Kind:
			return coderUint32PackedSliceValue
		case pref.Int64Kind:
			return coderInt64PackedSliceValue
		case pref.Sint64Kind:
			return coderSint64PackedSliceValue
		case pref.Uint64Kind:
			return coderUint64PackedSliceValue
		case pref.Sfixed32Kind:
			return coderSfixed32PackedSliceValue
		case pref.Fixed32Kind:
			return coderFixed32PackedSliceValue
		case pref.FloatKind:
			return coderFloatPackedSliceValue
		case pref.Sfixed64Kind:
			return coderSfixed64PackedSliceValue
		case pref.Fixed64Kind:
			return coderFixed64PackedSliceValue
		case pref.DoubleKind:
			return coderDoublePackedSliceValue
		}
	default:
		switch fd.Kind() {
		default:
		case pref.BoolKind:
			return coderBoolValue
		case pref.EnumKind:
			return coderEnumValue
		case pref.Int32Kind:
			return coderInt32Value
		case pref.Sint32Kind:
			return coderSint32Value
		case pref.Uint32Kind:
			return coderUint32Value
		case pref.Int64Kind:
			return coderInt64Value
		case pref.Sint64Kind:
			return coderSint64Value
		case pref.Uint64Kind:
			return coderUint64Value
		case pref.Sfixed32Kind:
			return coderSfixed32Value
		case pref.Fixed32Kind:
			return coderFixed32Value
		case pref.FloatKind:
			return coderFloatValue
		case pref.Sfixed64Kind:
			return coderSfixed64Value
		case pref.Fixed64Kind:
			return coderFixed64Value
		case pref.DoubleKind:
			return coderDoubleValue
		case pref.StringKind:
			if strs.EnforceUTF8(fd) {
				return coderStringValueValidateUTF8
			}
			return coderStringValue
		case pref.BytesKind:
			return coderBytesValue
		case pref.MessageKind:
			return coderMessageValue
		case pref.GroupKind:
			return coderGroupValue
		}
	}
	panic(fmt.Sprintf("invalid field: no encoder for %v %v %v", fd.FullName(), fd.Cardinality(), fd.Kind()))
}
