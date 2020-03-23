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

// +build purego appengine js

// This file contains an implementation of proto field accesses using package reflect.
// It is slower than the code in pointer_unsafe.go but it avoids package unsafe and can
// be used on App Engine.

package proto

import (
	"reflect"
	"sync"
)

const unsafeAllowed = false

// A field identifies a field in a struct, accessible from a pointer.
// In this implementation, a field is identified by the sequence of field indices
// passed to reflect's FieldByIndex.
type field []int

// toField returns a field equivalent to the given reflect field.
func toField(f *reflect.StructField) field {
	return f.Index
}

// invalidField is an invalid field identifier.
var invalidField = field(nil)

// zeroField is a noop when calling pointer.offset.
var zeroField = field([]int{})

// IsValid reports whether the field identifier is valid.
func (f field) IsValid() bool { return f != nil }

// The pointer type is for the table-driven decoder.
// The implementation here uses a reflect.Value of pointer type to
// create a generic pointer. In pointer_unsafe.go we use unsafe
// instead of reflect to implement the same (but faster) interface.
type pointer struct {
	v reflect.Value
}

// toPointer converts an interface of pointer type to a pointer
// that points to the same target.
func toPointer(i *Message) pointer {
	return pointer{v: reflect.ValueOf(*i)}
}

// toAddrPointer converts an interface to a pointer that points to
// the interface data.
func toAddrPointer(i *interface{}, isptr, deref bool) pointer {
	v := reflect.ValueOf(*i)
	u := reflect.New(v.Type())
	u.Elem().Set(v)
	if deref {
		u = u.Elem()
	}
	return pointer{v: u}
}

// valToPointer converts v to a pointer.  v must be of pointer type.
func valToPointer(v reflect.Value) pointer {
	return pointer{v: v}
}

// offset converts from a pointer to a structure to a pointer to
// one of its fields.
func (p pointer) offset(f field) pointer {
	return pointer{v: p.v.Elem().FieldByIndex(f).Addr()}
}

func (p pointer) isNil() bool {
	return p.v.IsNil()
}

// grow updates the slice s in place to make it one element longer.
// s must be addressable.
// Returns the (addressable) new element.
func grow(s reflect.Value) reflect.Value {
	n, m := s.Len(), s.Cap()
	if n < m {
		s.SetLen(n + 1)
	} else {
		s.Set(reflect.Append(s, reflect.Zero(s.Type().Elem())))
	}
	return s.Index(n)
}

func (p pointer) toInt64() *int64 {
	return p.v.Interface().(*int64)
}
func (p pointer) toInt64Ptr() **int64 {
	return p.v.Interface().(**int64)
}
func (p pointer) toInt64Slice() *[]int64 {
	return p.v.Interface().(*[]int64)
}

var int32ptr = reflect.TypeOf((*int32)(nil))

func (p pointer) toInt32() *int32 {
	return p.v.Convert(int32ptr).Interface().(*int32)
}

// The toInt32Ptr/Slice methods don't work because of enums.
// Instead, we must use set/get methods for the int32ptr/slice case.
/*
	func (p pointer) toInt32Ptr() **int32 {
		return p.v.Interface().(**int32)
}
	func (p pointer) toInt32Slice() *[]int32 {
		return p.v.Interface().(*[]int32)
}
*/
func (p pointer) getInt32Ptr() *int32 {
	if p.v.Type().Elem().Elem() == reflect.TypeOf(int32(0)) {
		// raw int32 type
		return p.v.Elem().Interface().(*int32)
	}
	// an enum
	return p.v.Elem().Convert(int32PtrType).Interface().(*int32)
}
func (p pointer) setInt32Ptr(v int32) {
	// Allocate value in a *int32. Possibly convert that to a *enum.
	// Then assign it to a **int32 or **enum.
	// Note: we can convert *int32 to *enum, but we can't convert
	// **int32 to **enum!
	p.v.Elem().Set(reflect.ValueOf(&v).Convert(p.v.Type().Elem()))
}

// getInt32Slice copies []int32 from p as a new slice.
// This behavior differs from the implementation in pointer_unsafe.go.
func (p pointer) getInt32Slice() []int32 {
	if p.v.Type().Elem().Elem() == reflect.TypeOf(int32(0)) {
		// raw int32 type
		return p.v.Elem().Interface().([]int32)
	}
	// an enum
	// Allocate a []int32, then assign []enum's values into it.
	// Note: we can't convert []enum to []int32.
	slice := p.v.Elem()
	s := make([]int32, slice.Len())
	for i := 0; i < slice.Len(); i++ {
		s[i] = int32(slice.Index(i).Int())
	}
	return s
}

// setInt32Slice copies []int32 into p as a new slice.
// This behavior differs from the implementation in pointer_unsafe.go.
func (p pointer) setInt32Slice(v []int32) {
	if p.v.Type().Elem().Elem() == reflect.TypeOf(int32(0)) {
		// raw int32 type
		p.v.Elem().Set(reflect.ValueOf(v))
		return
	}
	// an enum
	// Allocate a []enum, then assign []int32's values into it.
	// Note: we can't convert []enum to []int32.
	slice := reflect.MakeSlice(p.v.Type().Elem(), len(v), cap(v))
	for i, x := range v {
		slice.Index(i).SetInt(int64(x))
	}
	p.v.Elem().Set(slice)
}
func (p pointer) appendInt32Slice(v int32) {
	grow(p.v.Elem()).SetInt(int64(v))
}

func (p pointer) toUint64() *uint64 {
	return p.v.Interface().(*uint64)
}
func (p pointer) toUint64Ptr() **uint64 {
	return p.v.Interface().(**uint64)
}
func (p pointer) toUint64Slice() *[]uint64 {
	return p.v.Interface().(*[]uint64)
}
func (p pointer) toUint32() *uint32 {
	return p.v.Interface().(*uint32)
}
func (p pointer) toUint32Ptr() **uint32 {
	return p.v.Interface().(**uint32)
}
func (p pointer) toUint32Slice() *[]uint32 {
	return p.v.Interface().(*[]uint32)
}
func (p pointer) toBool() *bool {
	return p.v.Interface().(*bool)
}
func (p pointer) toBoolPtr() **bool {
	return p.v.Interface().(**bool)
}
func (p pointer) toBoolSlice() *[]bool {
	return p.v.Interface().(*[]bool)
}
func (p pointer) toFloat64() *float64 {
	return p.v.Interface().(*float64)
}
func (p pointer) toFloat64Ptr() **float64 {
	return p.v.Interface().(**float64)
}
func (p pointer) toFloat64Slice() *[]float64 {
	return p.v.Interface().(*[]float64)
}
func (p pointer) toFloat32() *float32 {
	return p.v.Interface().(*float32)
}
func (p pointer) toFloat32Ptr() **float32 {
	return p.v.Interface().(**float32)
}
func (p pointer) toFloat32Slice() *[]float32 {
	return p.v.Interface().(*[]float32)
}
func (p pointer) toString() *string {
	return p.v.Interface().(*string)
}
func (p pointer) toStringPtr() **string {
	return p.v.Interface().(**string)
}
func (p pointer) toStringSlice() *[]string {
	return p.v.Interface().(*[]string)
}
func (p pointer) toBytes() *[]byte {
	return p.v.Interface().(*[]byte)
}
func (p pointer) toBytesSlice() *[][]byte {
	return p.v.Interface().(*[][]byte)
}
func (p pointer) toExtensions() *XXX_InternalExtensions {
	return p.v.Interface().(*XXX_InternalExtensions)
}
func (p pointer) toOldExtensions() *map[int32]Extension {
	return p.v.Interface().(*map[int32]Extension)
}
func (p pointer) getPointer() pointer {
	return pointer{v: p.v.Elem()}
}
func (p pointer) setPointer(q pointer) {
	p.v.Elem().Set(q.v)
}
func (p pointer) appendPointer(q pointer) {
	grow(p.v.Elem()).Set(q.v)
}

// getPointerSlice copies []*T from p as a new []pointer.
// This behavior differs from the implementation in pointer_unsafe.go.
func (p pointer) getPointerSlice() []pointer {
	if p.v.IsNil() {
		return nil
	}
	n := p.v.Elem().Len()
	s := make([]pointer, n)
	for i := 0; i < n; i++ {
		s[i] = pointer{v: p.v.Elem().Index(i)}
	}
	return s
}

// setPointerSlice copies []pointer into p as a new []*T.
// This behavior differs from the implementation in pointer_unsafe.go.
func (p pointer) setPointerSlice(v []pointer) {
	if v == nil {
		p.v.Elem().Set(reflect.New(p.v.Elem().Type()).Elem())
		return
	}
	s := reflect.MakeSlice(p.v.Elem().Type(), 0, len(v))
	for _, p := range v {
		s = reflect.Append(s, p.v)
	}
	p.v.Elem().Set(s)
}

// getInterfacePointer returns a pointer that points to the
// interface data of the interface pointed by p.
func (p pointer) getInterfacePointer() pointer {
	if p.v.Elem().IsNil() {
		return pointer{v: p.v.Elem()}
	}
	return pointer{v: p.v.Elem().Elem().Elem().Field(0).Addr()} // *interface -> interface -> *struct -> struct
}

func (p pointer) asPointerTo(t reflect.Type) reflect.Value {
	// TODO: check that p.v.Type().Elem() == t?
	return p.v
}

func atomicLoadUnmarshalInfo(p **unmarshalInfo) *unmarshalInfo {
	atomicLock.Lock()
	defer atomicLock.Unlock()
	return *p
}
func atomicStoreUnmarshalInfo(p **unmarshalInfo, v *unmarshalInfo) {
	atomicLock.Lock()
	defer atomicLock.Unlock()
	*p = v
}
func atomicLoadMarshalInfo(p **marshalInfo) *marshalInfo {
	atomicLock.Lock()
	defer atomicLock.Unlock()
	return *p
}
func atomicStoreMarshalInfo(p **marshalInfo, v *marshalInfo) {
	atomicLock.Lock()
	defer atomicLock.Unlock()
	*p = v
}
func atomicLoadMergeInfo(p **mergeInfo) *mergeInfo {
	atomicLock.Lock()
	defer atomicLock.Unlock()
	return *p
}
func atomicStoreMergeInfo(p **mergeInfo, v *mergeInfo) {
	atomicLock.Lock()
	defer atomicLock.Unlock()
	*p = v
}
func atomicLoadDiscardInfo(p **discardInfo) *discardInfo {
	atomicLock.Lock()
	defer atomicLock.Unlock()
	return *p
}
func atomicStoreDiscardInfo(p **discardInfo, v *discardInfo) {
	atomicLock.Lock()
	defer atomicLock.Unlock()
	*p = v
}

var atomicLock sync.Mutex
