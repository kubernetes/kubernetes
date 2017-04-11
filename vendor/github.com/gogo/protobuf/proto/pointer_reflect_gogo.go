// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2016, The GoGo Authors. All rights reserved.
// http://github.com/gogo/protobuf
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

// +build appengine js

package proto

import (
	"reflect"
)

func structPointer_FieldPointer(p structPointer, f field) structPointer {
	panic("not implemented")
}

func appendStructPointer(base structPointer, f field, typ reflect.Type) structPointer {
	panic("not implemented")
}

func structPointer_InterfaceAt(p structPointer, f field, t reflect.Type) interface{} {
	panic("not implemented")
}

func structPointer_InterfaceRef(p structPointer, f field, t reflect.Type) interface{} {
	panic("not implemented")
}

func structPointer_GetRefStructPointer(p structPointer, f field) structPointer {
	panic("not implemented")
}

func structPointer_Add(p structPointer, size field) structPointer {
	panic("not implemented")
}

func structPointer_Len(p structPointer, f field) int {
	panic("not implemented")
}

func structPointer_GetSliceHeader(p structPointer, f field) *reflect.SliceHeader {
	panic("not implemented")
}

func structPointer_Copy(oldptr structPointer, newptr structPointer, size int) {
	panic("not implemented")
}

func structPointer_StructRefSlice(p structPointer, f field, size uintptr) *structRefSlice {
	panic("not implemented")
}

type structRefSlice struct{}

func (v *structRefSlice) Len() int {
	panic("not implemented")
}

func (v *structRefSlice) Index(i int) structPointer {
	panic("not implemented")
}
