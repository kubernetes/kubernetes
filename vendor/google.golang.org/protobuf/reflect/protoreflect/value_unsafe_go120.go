// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.21

package protoreflect

import (
	"unsafe"

	"google.golang.org/protobuf/internal/pragma"
)

type (
	stringHeader struct {
		Data unsafe.Pointer
		Len  int
	}
	sliceHeader struct {
		Data unsafe.Pointer
		Len  int
		Cap  int
	}
	ifaceHeader struct {
		Type unsafe.Pointer
		Data unsafe.Pointer
	}
)

var (
	nilType     = typeOf(nil)
	boolType    = typeOf(*new(bool))
	int32Type   = typeOf(*new(int32))
	int64Type   = typeOf(*new(int64))
	uint32Type  = typeOf(*new(uint32))
	uint64Type  = typeOf(*new(uint64))
	float32Type = typeOf(*new(float32))
	float64Type = typeOf(*new(float64))
	stringType  = typeOf(*new(string))
	bytesType   = typeOf(*new([]byte))
	enumType    = typeOf(*new(EnumNumber))
)

// typeOf returns a pointer to the Go type information.
// The pointer is comparable and equal if and only if the types are identical.
func typeOf(t any) unsafe.Pointer {
	return (*ifaceHeader)(unsafe.Pointer(&t)).Type
}

// value is a union where only one type can be represented at a time.
// The struct is 24B large on 64-bit systems and requires the minimum storage
// necessary to represent each possible type.
//
// The Go GC needs to be able to scan variables containing pointers.
// As such, pointers and non-pointers cannot be intermixed.
type value struct {
	pragma.DoNotCompare // 0B

	// typ stores the type of the value as a pointer to the Go type.
	typ unsafe.Pointer // 8B

	// ptr stores the data pointer for a String, Bytes, or interface value.
	ptr unsafe.Pointer // 8B

	// num stores a Bool, Int32, Int64, Uint32, Uint64, Float32, Float64, or
	// Enum value as a raw uint64.
	//
	// It is also used to store the length of a String or Bytes value;
	// the capacity is ignored.
	num uint64 // 8B
}

func valueOfString(v string) Value {
	p := (*stringHeader)(unsafe.Pointer(&v))
	return Value{typ: stringType, ptr: p.Data, num: uint64(len(v))}
}
func valueOfBytes(v []byte) Value {
	p := (*sliceHeader)(unsafe.Pointer(&v))
	return Value{typ: bytesType, ptr: p.Data, num: uint64(len(v))}
}
func valueOfIface(v any) Value {
	p := (*ifaceHeader)(unsafe.Pointer(&v))
	return Value{typ: p.Type, ptr: p.Data}
}

func (v Value) getString() (x string) {
	*(*stringHeader)(unsafe.Pointer(&x)) = stringHeader{Data: v.ptr, Len: int(v.num)}
	return x
}
func (v Value) getBytes() (x []byte) {
	*(*sliceHeader)(unsafe.Pointer(&x)) = sliceHeader{Data: v.ptr, Len: int(v.num), Cap: int(v.num)}
	return x
}
func (v Value) getIface() (x any) {
	*(*ifaceHeader)(unsafe.Pointer(&x)) = ifaceHeader{Type: v.typ, Data: v.ptr}
	return x
}
