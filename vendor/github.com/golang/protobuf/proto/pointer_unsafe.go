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

// +build !purego,!appengine,!js

// This file contains the implementation of the proto field accesses using package unsafe.

package proto

import (
	"reflect"
	"sync/atomic"
	"unsafe"
)

const unsafeAllowed = true

// A field identifies a field in a struct, accessible from a pointer.
// In this implementation, a field is identified by its byte offset from the start of the struct.
type field uintptr

// toField returns a field equivalent to the given reflect field.
func toField(f *reflect.StructField) field {
	return field(f.Offset)
}

// invalidField is an invalid field identifier.
const invalidField = ^field(0)

// zeroField is a noop when calling pointer.offset.
const zeroField = field(0)

// IsValid reports whether the field identifier is valid.
func (f field) IsValid() bool {
	return f != invalidField
}

// The pointer type below is for the new table-driven encoder/decoder.
// The implementation here uses unsafe.Pointer to create a generic pointer.
// In pointer_reflect.go we use reflect instead of unsafe to implement
// the same (but slower) interface.
type pointer struct {
	p unsafe.Pointer
}

// size of pointer
var ptrSize = unsafe.Sizeof(uintptr(0))

// toPointer converts an interface of pointer type to a pointer
// that points to the same target.
func toPointer(i *Message) pointer {
	// Super-tricky - read pointer out of data word of interface value.
	// Saves ~25ns over the equivalent:
	// return valToPointer(reflect.ValueOf(*i))
	return pointer{p: (*[2]unsafe.Pointer)(unsafe.Pointer(i))[1]}
}

// toAddrPointer converts an interface to a pointer that points to
// the interface data.
func toAddrPointer(i *interface{}, isptr, deref bool) (p pointer) {
	// Super-tricky - read or get the address of data word of interface value.
	if isptr {
		// The interface is of pointer type, thus it is a direct interface.
		// The data word is the pointer data itself. We take its address.
		p = pointer{p: unsafe.Pointer(uintptr(unsafe.Pointer(i)) + ptrSize)}
	} else {
		// The interface is not of pointer type. The data word is the pointer
		// to the data.
		p = pointer{p: (*[2]unsafe.Pointer)(unsafe.Pointer(i))[1]}
	}
	if deref {
		p.p = *(*unsafe.Pointer)(p.p)
	}
	return p
}

// valToPointer converts v to a pointer. v must be of pointer type.
func valToPointer(v reflect.Value) pointer {
	return pointer{p: unsafe.Pointer(v.Pointer())}
}

// offset converts from a pointer to a structure to a pointer to
// one of its fields.
func (p pointer) offset(f field) pointer {
	// For safety, we should panic if !f.IsValid, however calling panic causes
	// this to no longer be inlineable, which is a serious performance cost.
	/*
		if !f.IsValid() {
			panic("invalid field")
		}
	*/
	return pointer{p: unsafe.Pointer(uintptr(p.p) + uintptr(f))}
}

func (p pointer) isNil() bool {
	return p.p == nil
}

func (p pointer) toInt64() *int64 {
	return (*int64)(p.p)
}
func (p pointer) toInt64Ptr() **int64 {
	return (**int64)(p.p)
}
func (p pointer) toInt64Slice() *[]int64 {
	return (*[]int64)(p.p)
}
func (p pointer) toInt32() *int32 {
	return (*int32)(p.p)
}

// See pointer_reflect.go for why toInt32Ptr/Slice doesn't exist.
/*
	func (p pointer) toInt32Ptr() **int32 {
		return (**int32)(p.p)
	}
	func (p pointer) toInt32Slice() *[]int32 {
		return (*[]int32)(p.p)
	}
*/
func (p pointer) getInt32Ptr() *int32 {
	return *(**int32)(p.p)
}
func (p pointer) setInt32Ptr(v int32) {
	*(**int32)(p.p) = &v
}

// getInt32Slice loads a []int32 from p.
// The value returned is aliased with the original slice.
// This behavior differs from the implementation in pointer_reflect.go.
func (p pointer) getInt32Slice() []int32 {
	return *(*[]int32)(p.p)
}

// setInt32Slice stores a []int32 to p.
// The value set is aliased with the input slice.
// This behavior differs from the implementation in pointer_reflect.go.
func (p pointer) setInt32Slice(v []int32) {
	*(*[]int32)(p.p) = v
}

// TODO: Can we get rid of appendInt32Slice and use setInt32Slice instead?
func (p pointer) appendInt32Slice(v int32) {
	s := (*[]int32)(p.p)
	*s = append(*s, v)
}

func (p pointer) toUint64() *uint64 {
	return (*uint64)(p.p)
}
func (p pointer) toUint64Ptr() **uint64 {
	return (**uint64)(p.p)
}
func (p pointer) toUint64Slice() *[]uint64 {
	return (*[]uint64)(p.p)
}
func (p pointer) toUint32() *uint32 {
	return (*uint32)(p.p)
}
func (p pointer) toUint32Ptr() **uint32 {
	return (**uint32)(p.p)
}
func (p pointer) toUint32Slice() *[]uint32 {
	return (*[]uint32)(p.p)
}
func (p pointer) toBool() *bool {
	return (*bool)(p.p)
}
func (p pointer) toBoolPtr() **bool {
	return (**bool)(p.p)
}
func (p pointer) toBoolSlice() *[]bool {
	return (*[]bool)(p.p)
}
func (p pointer) toFloat64() *float64 {
	return (*float64)(p.p)
}
func (p pointer) toFloat64Ptr() **float64 {
	return (**float64)(p.p)
}
func (p pointer) toFloat64Slice() *[]float64 {
	return (*[]float64)(p.p)
}
func (p pointer) toFloat32() *float32 {
	return (*float32)(p.p)
}
func (p pointer) toFloat32Ptr() **float32 {
	return (**float32)(p.p)
}
func (p pointer) toFloat32Slice() *[]float32 {
	return (*[]float32)(p.p)
}
func (p pointer) toString() *string {
	return (*string)(p.p)
}
func (p pointer) toStringPtr() **string {
	return (**string)(p.p)
}
func (p pointer) toStringSlice() *[]string {
	return (*[]string)(p.p)
}
func (p pointer) toBytes() *[]byte {
	return (*[]byte)(p.p)
}
func (p pointer) toBytesSlice() *[][]byte {
	return (*[][]byte)(p.p)
}
func (p pointer) toExtensions() *XXX_InternalExtensions {
	return (*XXX_InternalExtensions)(p.p)
}
func (p pointer) toOldExtensions() *map[int32]Extension {
	return (*map[int32]Extension)(p.p)
}

// getPointerSlice loads []*T from p as a []pointer.
// The value returned is aliased with the original slice.
// This behavior differs from the implementation in pointer_reflect.go.
func (p pointer) getPointerSlice() []pointer {
	// Super-tricky - p should point to a []*T where T is a
	// message type. We load it as []pointer.
	return *(*[]pointer)(p.p)
}

// setPointerSlice stores []pointer into p as a []*T.
// The value set is aliased with the input slice.
// This behavior differs from the implementation in pointer_reflect.go.
func (p pointer) setPointerSlice(v []pointer) {
	// Super-tricky - p should point to a []*T where T is a
	// message type. We store it as []pointer.
	*(*[]pointer)(p.p) = v
}

// getPointer loads the pointer at p and returns it.
func (p pointer) getPointer() pointer {
	return pointer{p: *(*unsafe.Pointer)(p.p)}
}

// setPointer stores the pointer q at p.
func (p pointer) setPointer(q pointer) {
	*(*unsafe.Pointer)(p.p) = q.p
}

// append q to the slice pointed to by p.
func (p pointer) appendPointer(q pointer) {
	s := (*[]unsafe.Pointer)(p.p)
	*s = append(*s, q.p)
}

// getInterfacePointer returns a pointer that points to the
// interface data of the interface pointed by p.
func (p pointer) getInterfacePointer() pointer {
	// Super-tricky - read pointer out of data word of interface value.
	return pointer{p: (*(*[2]unsafe.Pointer)(p.p))[1]}
}

// asPointerTo returns a reflect.Value that is a pointer to an
// object of type t stored at p.
func (p pointer) asPointerTo(t reflect.Type) reflect.Value {
	return reflect.NewAt(t, p.p)
}

func atomicLoadUnmarshalInfo(p **unmarshalInfo) *unmarshalInfo {
	return (*unmarshalInfo)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(p))))
}
func atomicStoreUnmarshalInfo(p **unmarshalInfo, v *unmarshalInfo) {
	atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(p)), unsafe.Pointer(v))
}
func atomicLoadMarshalInfo(p **marshalInfo) *marshalInfo {
	return (*marshalInfo)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(p))))
}
func atomicStoreMarshalInfo(p **marshalInfo, v *marshalInfo) {
	atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(p)), unsafe.Pointer(v))
}
func atomicLoadMergeInfo(p **mergeInfo) *mergeInfo {
	return (*mergeInfo)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(p))))
}
func atomicStoreMergeInfo(p **mergeInfo, v *mergeInfo) {
	atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(p)), unsafe.Pointer(v))
}
func atomicLoadDiscardInfo(p **discardInfo) *discardInfo {
	return (*discardInfo)(atomic.LoadPointer((*unsafe.Pointer)(unsafe.Pointer(p))))
}
func atomicStoreDiscardInfo(p **discardInfo, v *discardInfo) {
	atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(p)), unsafe.Pointer(v))
}
