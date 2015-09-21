// Copyright (c) 2013, Vastech SA (PTY) LTD. All rights reserved.
// http://github.com/gogo/protobuf/gogoproto
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

// +build !appengine

// This file contains the implementation of the proto field accesses using package unsafe.

package proto

import (
	"reflect"
	"unsafe"
)

func structPointer_InterfaceAt(p structPointer, f field, t reflect.Type) interface{} {
	point := unsafe.Pointer(uintptr(p) + uintptr(f))
	r := reflect.NewAt(t, point)
	return r.Interface()
}

func structPointer_InterfaceRef(p structPointer, f field, t reflect.Type) interface{} {
	point := unsafe.Pointer(uintptr(p) + uintptr(f))
	r := reflect.NewAt(t, point)
	if r.Elem().IsNil() {
		return nil
	}
	return r.Elem().Interface()
}

func copyUintPtr(oldptr, newptr uintptr, size int) {
	oldbytes := make([]byte, 0)
	oldslice := (*reflect.SliceHeader)(unsafe.Pointer(&oldbytes))
	oldslice.Data = oldptr
	oldslice.Len = size
	oldslice.Cap = size
	newbytes := make([]byte, 0)
	newslice := (*reflect.SliceHeader)(unsafe.Pointer(&newbytes))
	newslice.Data = newptr
	newslice.Len = size
	newslice.Cap = size
	copy(newbytes, oldbytes)
}

func structPointer_Copy(oldptr structPointer, newptr structPointer, size int) {
	copyUintPtr(uintptr(oldptr), uintptr(newptr), size)
}

func appendStructPointer(base structPointer, f field, typ reflect.Type) structPointer {
	size := typ.Elem().Size()
	oldHeader := structPointer_GetSliceHeader(base, f)
	newLen := oldHeader.Len + 1
	slice := reflect.MakeSlice(typ, newLen, newLen)
	bas := toStructPointer(slice)
	for i := 0; i < oldHeader.Len; i++ {
		newElemptr := uintptr(bas) + uintptr(i)*size
		oldElemptr := oldHeader.Data + uintptr(i)*size
		copyUintPtr(oldElemptr, newElemptr, int(size))
	}

	oldHeader.Data = uintptr(bas)
	oldHeader.Len = newLen
	oldHeader.Cap = newLen

	return structPointer(unsafe.Pointer(uintptr(unsafe.Pointer(bas)) + uintptr(uintptr(newLen-1)*size)))
}

// RefBool returns a *bool field in the struct.
func structPointer_RefBool(p structPointer, f field) *bool {
	return (*bool)(unsafe.Pointer(uintptr(p) + uintptr(f)))
}

// RefString returns the address of a string field in the struct.
func structPointer_RefString(p structPointer, f field) *string {
	return (*string)(unsafe.Pointer(uintptr(p) + uintptr(f)))
}

func structPointer_FieldPointer(p structPointer, f field) structPointer {
	return structPointer(unsafe.Pointer(uintptr(p) + uintptr(f)))
}

func structPointer_GetRefStructPointer(p structPointer, f field) structPointer {
	return structPointer((*structPointer)(unsafe.Pointer(uintptr(p) + uintptr(f))))
}

func structPointer_GetSliceHeader(p structPointer, f field) *reflect.SliceHeader {
	return (*reflect.SliceHeader)(unsafe.Pointer(uintptr(p) + uintptr(f)))
}

func structPointer_Add(p structPointer, size field) structPointer {
	return structPointer(unsafe.Pointer(uintptr(p) + uintptr(size)))
}

func structPointer_Len(p structPointer, f field) int {
	return len(*(*[]interface{})(unsafe.Pointer(structPointer_GetRefStructPointer(p, f))))
}

// refWord32 is the address of a 32-bit value field.
type refWord32 *uint32

func refWord32_IsNil(p refWord32) bool {
	return p == nil
}

func refWord32_Set(p refWord32, o *Buffer, x uint32) {
	if len(o.uint32s) == 0 {
		o.uint32s = make([]uint32, uint32PoolSize)
	}
	o.uint32s[0] = x
	*p = o.uint32s[0]
	o.uint32s = o.uint32s[1:]
}

func refWord32_Get(p refWord32) uint32 {
	return *p
}

func structPointer_RefWord32(p structPointer, f field) refWord32 {
	return refWord32((*uint32)(unsafe.Pointer(uintptr(p) + uintptr(f))))
}

// refWord64 is like refWord32 but for 32-bit values.
type refWord64 *uint64

func refWord64_Set(p refWord64, o *Buffer, x uint64) {
	if len(o.uint64s) == 0 {
		o.uint64s = make([]uint64, uint64PoolSize)
	}
	o.uint64s[0] = x
	*p = o.uint64s[0]
	o.uint64s = o.uint64s[1:]
}

func refWord64_IsNil(p refWord64) bool {
	return p == nil
}

func refWord64_Get(p refWord64) uint64 {
	return *p
}

func structPointer_RefWord64(p structPointer, f field) refWord64 {
	return refWord64((*uint64)(unsafe.Pointer(uintptr(p) + uintptr(f))))
}
