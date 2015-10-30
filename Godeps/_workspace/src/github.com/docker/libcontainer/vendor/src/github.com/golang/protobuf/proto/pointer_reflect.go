// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2012 The Go Authors.  All rights reserved.
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

// +build appengine

// This file contains an implementation of proto field accesses using package reflect.
// It is slower than the code in pointer_unsafe.go but it avoids package unsafe and can
// be used on App Engine.

package proto

import (
	"math"
	"reflect"
)

// A structPointer is a pointer to a struct.
type structPointer struct {
	v reflect.Value
}

// toStructPointer returns a structPointer equivalent to the given reflect value.
// The reflect value must itself be a pointer to a struct.
func toStructPointer(v reflect.Value) structPointer {
	return structPointer{v}
}

// IsNil reports whether p is nil.
func structPointer_IsNil(p structPointer) bool {
	return p.v.IsNil()
}

// Interface returns the struct pointer as an interface value.
func structPointer_Interface(p structPointer, _ reflect.Type) interface{} {
	return p.v.Interface()
}

// A field identifies a field in a struct, accessible from a structPointer.
// In this implementation, a field is identified by the sequence of field indices
// passed to reflect's FieldByIndex.
type field []int

// toField returns a field equivalent to the given reflect field.
func toField(f *reflect.StructField) field {
	return f.Index
}

// invalidField is an invalid field identifier.
var invalidField = field(nil)

// IsValid reports whether the field identifier is valid.
func (f field) IsValid() bool { return f != nil }

// field returns the given field in the struct as a reflect value.
func structPointer_field(p structPointer, f field) reflect.Value {
	// Special case: an extension map entry with a value of type T
	// passes a *T to the struct-handling code with a zero field,
	// expecting that it will be treated as equivalent to *struct{ X T },
	// which has the same memory layout. We have to handle that case
	// specially, because reflect will panic if we call FieldByIndex on a
	// non-struct.
	if f == nil {
		return p.v.Elem()
	}

	return p.v.Elem().FieldByIndex(f)
}

// ifield returns the given field in the struct as an interface value.
func structPointer_ifield(p structPointer, f field) interface{} {
	return structPointer_field(p, f).Addr().Interface()
}

// Bytes returns the address of a []byte field in the struct.
func structPointer_Bytes(p structPointer, f field) *[]byte {
	return structPointer_ifield(p, f).(*[]byte)
}

// BytesSlice returns the address of a [][]byte field in the struct.
func structPointer_BytesSlice(p structPointer, f field) *[][]byte {
	return structPointer_ifield(p, f).(*[][]byte)
}

// Bool returns the address of a *bool field in the struct.
func structPointer_Bool(p structPointer, f field) **bool {
	return structPointer_ifield(p, f).(**bool)
}

// BoolVal returns the address of a bool field in the struct.
func structPointer_BoolVal(p structPointer, f field) *bool {
	return structPointer_ifield(p, f).(*bool)
}

// BoolSlice returns the address of a []bool field in the struct.
func structPointer_BoolSlice(p structPointer, f field) *[]bool {
	return structPointer_ifield(p, f).(*[]bool)
}

// String returns the address of a *string field in the struct.
func structPointer_String(p structPointer, f field) **string {
	return structPointer_ifield(p, f).(**string)
}

// StringVal returns the address of a string field in the struct.
func structPointer_StringVal(p structPointer, f field) *string {
	return structPointer_ifield(p, f).(*string)
}

// StringSlice returns the address of a []string field in the struct.
func structPointer_StringSlice(p structPointer, f field) *[]string {
	return structPointer_ifield(p, f).(*[]string)
}

// ExtMap returns the address of an extension map field in the struct.
func structPointer_ExtMap(p structPointer, f field) *map[int32]Extension {
	return structPointer_ifield(p, f).(*map[int32]Extension)
}

// Map returns the reflect.Value for the address of a map field in the struct.
func structPointer_Map(p structPointer, f field, typ reflect.Type) reflect.Value {
	return structPointer_field(p, f).Addr()
}

// SetStructPointer writes a *struct field in the struct.
func structPointer_SetStructPointer(p structPointer, f field, q structPointer) {
	structPointer_field(p, f).Set(q.v)
}

// GetStructPointer reads a *struct field in the struct.
func structPointer_GetStructPointer(p structPointer, f field) structPointer {
	return structPointer{structPointer_field(p, f)}
}

// StructPointerSlice the address of a []*struct field in the struct.
func structPointer_StructPointerSlice(p structPointer, f field) structPointerSlice {
	return structPointerSlice{structPointer_field(p, f)}
}

// A structPointerSlice represents the address of a slice of pointers to structs
// (themselves messages or groups). That is, v.Type() is *[]*struct{...}.
type structPointerSlice struct {
	v reflect.Value
}

func (p structPointerSlice) Len() int                  { return p.v.Len() }
func (p structPointerSlice) Index(i int) structPointer { return structPointer{p.v.Index(i)} }
func (p structPointerSlice) Append(q structPointer) {
	p.v.Set(reflect.Append(p.v, q.v))
}

var (
	int32Type   = reflect.TypeOf(int32(0))
	uint32Type  = reflect.TypeOf(uint32(0))
	float32Type = reflect.TypeOf(float32(0))
	int64Type   = reflect.TypeOf(int64(0))
	uint64Type  = reflect.TypeOf(uint64(0))
	float64Type = reflect.TypeOf(float64(0))
)

// A word32 represents a field of type *int32, *uint32, *float32, or *enum.
// That is, v.Type() is *int32, *uint32, *float32, or *enum and v is assignable.
type word32 struct {
	v reflect.Value
}

// IsNil reports whether p is nil.
func word32_IsNil(p word32) bool {
	return p.v.IsNil()
}

// Set sets p to point at a newly allocated word with bits set to x.
func word32_Set(p word32, o *Buffer, x uint32) {
	t := p.v.Type().Elem()
	switch t {
	case int32Type:
		if len(o.int32s) == 0 {
			o.int32s = make([]int32, uint32PoolSize)
		}
		o.int32s[0] = int32(x)
		p.v.Set(reflect.ValueOf(&o.int32s[0]))
		o.int32s = o.int32s[1:]
		return
	case uint32Type:
		if len(o.uint32s) == 0 {
			o.uint32s = make([]uint32, uint32PoolSize)
		}
		o.uint32s[0] = x
		p.v.Set(reflect.ValueOf(&o.uint32s[0]))
		o.uint32s = o.uint32s[1:]
		return
	case float32Type:
		if len(o.float32s) == 0 {
			o.float32s = make([]float32, uint32PoolSize)
		}
		o.float32s[0] = math.Float32frombits(x)
		p.v.Set(reflect.ValueOf(&o.float32s[0]))
		o.float32s = o.float32s[1:]
		return
	}

	// must be enum
	p.v.Set(reflect.New(t))
	p.v.Elem().SetInt(int64(int32(x)))
}

// Get gets the bits pointed at by p, as a uint32.
func word32_Get(p word32) uint32 {
	elem := p.v.Elem()
	switch elem.Kind() {
	case reflect.Int32:
		return uint32(elem.Int())
	case reflect.Uint32:
		return uint32(elem.Uint())
	case reflect.Float32:
		return math.Float32bits(float32(elem.Float()))
	}
	panic("unreachable")
}

// Word32 returns a reference to a *int32, *uint32, *float32, or *enum field in the struct.
func structPointer_Word32(p structPointer, f field) word32 {
	return word32{structPointer_field(p, f)}
}

// A word32Val represents a field of type int32, uint32, float32, or enum.
// That is, v.Type() is int32, uint32, float32, or enum and v is assignable.
type word32Val struct {
	v reflect.Value
}

// Set sets *p to x.
func word32Val_Set(p word32Val, x uint32) {
	switch p.v.Type() {
	case int32Type:
		p.v.SetInt(int64(x))
		return
	case uint32Type:
		p.v.SetUint(uint64(x))
		return
	case float32Type:
		p.v.SetFloat(float64(math.Float32frombits(x)))
		return
	}

	// must be enum
	p.v.SetInt(int64(int32(x)))
}

// Get gets the bits pointed at by p, as a uint32.
func word32Val_Get(p word32Val) uint32 {
	elem := p.v
	switch elem.Kind() {
	case reflect.Int32:
		return uint32(elem.Int())
	case reflect.Uint32:
		return uint32(elem.Uint())
	case reflect.Float32:
		return math.Float32bits(float32(elem.Float()))
	}
	panic("unreachable")
}

// Word32Val returns a reference to a int32, uint32, float32, or enum field in the struct.
func structPointer_Word32Val(p structPointer, f field) word32Val {
	return word32Val{structPointer_field(p, f)}
}

// A word32Slice is a slice of 32-bit values.
// That is, v.Type() is []int32, []uint32, []float32, or []enum.
type word32Slice struct {
	v reflect.Value
}

func (p word32Slice) Append(x uint32) {
	n, m := p.v.Len(), p.v.Cap()
	if n < m {
		p.v.SetLen(n + 1)
	} else {
		t := p.v.Type().Elem()
		p.v.Set(reflect.Append(p.v, reflect.Zero(t)))
	}
	elem := p.v.Index(n)
	switch elem.Kind() {
	case reflect.Int32:
		elem.SetInt(int64(int32(x)))
	case reflect.Uint32:
		elem.SetUint(uint64(x))
	case reflect.Float32:
		elem.SetFloat(float64(math.Float32frombits(x)))
	}
}

func (p word32Slice) Len() int {
	return p.v.Len()
}

func (p word32Slice) Index(i int) uint32 {
	elem := p.v.Index(i)
	switch elem.Kind() {
	case reflect.Int32:
		return uint32(elem.Int())
	case reflect.Uint32:
		return uint32(elem.Uint())
	case reflect.Float32:
		return math.Float32bits(float32(elem.Float()))
	}
	panic("unreachable")
}

// Word32Slice returns a reference to a []int32, []uint32, []float32, or []enum field in the struct.
func structPointer_Word32Slice(p structPointer, f field) word32Slice {
	return word32Slice{structPointer_field(p, f)}
}

// word64 is like word32 but for 64-bit values.
type word64 struct {
	v reflect.Value
}

func word64_Set(p word64, o *Buffer, x uint64) {
	t := p.v.Type().Elem()
	switch t {
	case int64Type:
		if len(o.int64s) == 0 {
			o.int64s = make([]int64, uint64PoolSize)
		}
		o.int64s[0] = int64(x)
		p.v.Set(reflect.ValueOf(&o.int64s[0]))
		o.int64s = o.int64s[1:]
		return
	case uint64Type:
		if len(o.uint64s) == 0 {
			o.uint64s = make([]uint64, uint64PoolSize)
		}
		o.uint64s[0] = x
		p.v.Set(reflect.ValueOf(&o.uint64s[0]))
		o.uint64s = o.uint64s[1:]
		return
	case float64Type:
		if len(o.float64s) == 0 {
			o.float64s = make([]float64, uint64PoolSize)
		}
		o.float64s[0] = math.Float64frombits(x)
		p.v.Set(reflect.ValueOf(&o.float64s[0]))
		o.float64s = o.float64s[1:]
		return
	}
	panic("unreachable")
}

func word64_IsNil(p word64) bool {
	return p.v.IsNil()
}

func word64_Get(p word64) uint64 {
	elem := p.v.Elem()
	switch elem.Kind() {
	case reflect.Int64:
		return uint64(elem.Int())
	case reflect.Uint64:
		return elem.Uint()
	case reflect.Float64:
		return math.Float64bits(elem.Float())
	}
	panic("unreachable")
}

func structPointer_Word64(p structPointer, f field) word64 {
	return word64{structPointer_field(p, f)}
}

// word64Val is like word32Val but for 64-bit values.
type word64Val struct {
	v reflect.Value
}

func word64Val_Set(p word64Val, o *Buffer, x uint64) {
	switch p.v.Type() {
	case int64Type:
		p.v.SetInt(int64(x))
		return
	case uint64Type:
		p.v.SetUint(x)
		return
	case float64Type:
		p.v.SetFloat(math.Float64frombits(x))
		return
	}
	panic("unreachable")
}

func word64Val_Get(p word64Val) uint64 {
	elem := p.v
	switch elem.Kind() {
	case reflect.Int64:
		return uint64(elem.Int())
	case reflect.Uint64:
		return elem.Uint()
	case reflect.Float64:
		return math.Float64bits(elem.Float())
	}
	panic("unreachable")
}

func structPointer_Word64Val(p structPointer, f field) word64Val {
	return word64Val{structPointer_field(p, f)}
}

type word64Slice struct {
	v reflect.Value
}

func (p word64Slice) Append(x uint64) {
	n, m := p.v.Len(), p.v.Cap()
	if n < m {
		p.v.SetLen(n + 1)
	} else {
		t := p.v.Type().Elem()
		p.v.Set(reflect.Append(p.v, reflect.Zero(t)))
	}
	elem := p.v.Index(n)
	switch elem.Kind() {
	case reflect.Int64:
		elem.SetInt(int64(int64(x)))
	case reflect.Uint64:
		elem.SetUint(uint64(x))
	case reflect.Float64:
		elem.SetFloat(float64(math.Float64frombits(x)))
	}
}

func (p word64Slice) Len() int {
	return p.v.Len()
}

func (p word64Slice) Index(i int) uint64 {
	elem := p.v.Index(i)
	switch elem.Kind() {
	case reflect.Int64:
		return uint64(elem.Int())
	case reflect.Uint64:
		return uint64(elem.Uint())
	case reflect.Float64:
		return math.Float64bits(float64(elem.Float()))
	}
	panic("unreachable")
}

func structPointer_Word64Slice(p structPointer, f field) word64Slice {
	return word64Slice{structPointer_field(p, f)}
}
