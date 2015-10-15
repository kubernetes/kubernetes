// Extensions for Protocol Buffers to create more go like structures.
//
// Copyright (c) 2013, Vastech SA (PTY) LTD. All rights reserved.
// http://github.com/gogo/protobuf/gogoproto
//
// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2010 The Go Authors.  All rights reserved.
// http://github.com/golang/protobuf/
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

package proto

import (
	"reflect"
)

type Sizer interface {
	Size() int
}

func (o *Buffer) enc_ext_slice_byte(p *Properties, base structPointer) error {
	s := *structPointer_Bytes(base, p.field)
	if s == nil {
		return ErrNil
	}
	o.buf = append(o.buf, s...)
	return nil
}

func size_ext_slice_byte(p *Properties, base structPointer) (n int) {
	s := *structPointer_Bytes(base, p.field)
	if s == nil {
		return 0
	}
	n += len(s)
	return
}

// Encode a reference to bool pointer.
func (o *Buffer) enc_ref_bool(p *Properties, base structPointer) error {
	v := structPointer_RefBool(base, p.field)
	if v == nil {
		return ErrNil
	}
	x := 0
	if *v {
		x = 1
	}
	o.buf = append(o.buf, p.tagcode...)
	p.valEnc(o, uint64(x))
	return nil
}

func size_ref_bool(p *Properties, base structPointer) int {
	v := structPointer_RefBool(base, p.field)
	if v == nil {
		return 0
	}
	return len(p.tagcode) + 1 // each bool takes exactly one byte
}

// Encode a reference to int32 pointer.
func (o *Buffer) enc_ref_int32(p *Properties, base structPointer) error {
	v := structPointer_RefWord32(base, p.field)
	if refWord32_IsNil(v) {
		return ErrNil
	}
	x := int32(refWord32_Get(v))
	o.buf = append(o.buf, p.tagcode...)
	p.valEnc(o, uint64(x))
	return nil
}

func size_ref_int32(p *Properties, base structPointer) (n int) {
	v := structPointer_RefWord32(base, p.field)
	if refWord32_IsNil(v) {
		return 0
	}
	x := int32(refWord32_Get(v))
	n += len(p.tagcode)
	n += p.valSize(uint64(x))
	return
}

func (o *Buffer) enc_ref_uint32(p *Properties, base structPointer) error {
	v := structPointer_RefWord32(base, p.field)
	if refWord32_IsNil(v) {
		return ErrNil
	}
	x := refWord32_Get(v)
	o.buf = append(o.buf, p.tagcode...)
	p.valEnc(o, uint64(x))
	return nil
}

func size_ref_uint32(p *Properties, base structPointer) (n int) {
	v := structPointer_RefWord32(base, p.field)
	if refWord32_IsNil(v) {
		return 0
	}
	x := refWord32_Get(v)
	n += len(p.tagcode)
	n += p.valSize(uint64(x))
	return
}

// Encode a reference to an int64 pointer.
func (o *Buffer) enc_ref_int64(p *Properties, base structPointer) error {
	v := structPointer_RefWord64(base, p.field)
	if refWord64_IsNil(v) {
		return ErrNil
	}
	x := refWord64_Get(v)
	o.buf = append(o.buf, p.tagcode...)
	p.valEnc(o, x)
	return nil
}

func size_ref_int64(p *Properties, base structPointer) (n int) {
	v := structPointer_RefWord64(base, p.field)
	if refWord64_IsNil(v) {
		return 0
	}
	x := refWord64_Get(v)
	n += len(p.tagcode)
	n += p.valSize(x)
	return
}

// Encode a reference to a string pointer.
func (o *Buffer) enc_ref_string(p *Properties, base structPointer) error {
	v := structPointer_RefString(base, p.field)
	if v == nil {
		return ErrNil
	}
	x := *v
	o.buf = append(o.buf, p.tagcode...)
	o.EncodeStringBytes(x)
	return nil
}

func size_ref_string(p *Properties, base structPointer) (n int) {
	v := structPointer_RefString(base, p.field)
	if v == nil {
		return 0
	}
	x := *v
	n += len(p.tagcode)
	n += sizeStringBytes(x)
	return
}

// Encode a reference to a message struct.
func (o *Buffer) enc_ref_struct_message(p *Properties, base structPointer) error {
	var state errorState
	structp := structPointer_GetRefStructPointer(base, p.field)
	if structPointer_IsNil(structp) {
		return ErrNil
	}

	// Can the object marshal itself?
	if p.isMarshaler {
		m := structPointer_Interface(structp, p.stype).(Marshaler)
		data, err := m.Marshal()
		if err != nil && !state.shouldContinue(err, nil) {
			return err
		}
		o.buf = append(o.buf, p.tagcode...)
		o.EncodeRawBytes(data)
		return nil
	}

	o.buf = append(o.buf, p.tagcode...)
	return o.enc_len_struct(p.sprop, structp, &state)
}

//TODO this is only copied, please fix this
func size_ref_struct_message(p *Properties, base structPointer) int {
	structp := structPointer_GetRefStructPointer(base, p.field)
	if structPointer_IsNil(structp) {
		return 0
	}

	// Can the object marshal itself?
	if p.isMarshaler {
		m := structPointer_Interface(structp, p.stype).(Marshaler)
		data, _ := m.Marshal()
		n0 := len(p.tagcode)
		n1 := sizeRawBytes(data)
		return n0 + n1
	}

	n0 := len(p.tagcode)
	n1 := size_struct(p.sprop, structp)
	n2 := sizeVarint(uint64(n1)) // size of encoded length
	return n0 + n1 + n2
}

// Encode a slice of references to message struct pointers ([]struct).
func (o *Buffer) enc_slice_ref_struct_message(p *Properties, base structPointer) error {
	var state errorState
	ss := structPointer_GetStructPointer(base, p.field)
	ss1 := structPointer_GetRefStructPointer(ss, field(0))
	size := p.stype.Size()
	l := structPointer_Len(base, p.field)
	for i := 0; i < l; i++ {
		structp := structPointer_Add(ss1, field(uintptr(i)*size))
		if structPointer_IsNil(structp) {
			return ErrRepeatedHasNil
		}

		// Can the object marshal itself?
		if p.isMarshaler {
			m := structPointer_Interface(structp, p.stype).(Marshaler)
			data, err := m.Marshal()
			if err != nil && !state.shouldContinue(err, nil) {
				return err
			}
			o.buf = append(o.buf, p.tagcode...)
			o.EncodeRawBytes(data)
			continue
		}

		o.buf = append(o.buf, p.tagcode...)
		err := o.enc_len_struct(p.sprop, structp, &state)
		if err != nil && !state.shouldContinue(err, nil) {
			if err == ErrNil {
				return ErrRepeatedHasNil
			}
			return err
		}

	}
	return state.err
}

//TODO this is only copied, please fix this
func size_slice_ref_struct_message(p *Properties, base structPointer) (n int) {
	ss := structPointer_GetStructPointer(base, p.field)
	ss1 := structPointer_GetRefStructPointer(ss, field(0))
	size := p.stype.Size()
	l := structPointer_Len(base, p.field)
	n += l * len(p.tagcode)
	for i := 0; i < l; i++ {
		structp := structPointer_Add(ss1, field(uintptr(i)*size))
		if structPointer_IsNil(structp) {
			return // return the size up to this point
		}

		// Can the object marshal itself?
		if p.isMarshaler {
			m := structPointer_Interface(structp, p.stype).(Marshaler)
			data, _ := m.Marshal()
			n += len(p.tagcode)
			n += sizeRawBytes(data)
			continue
		}

		n0 := size_struct(p.sprop, structp)
		n1 := sizeVarint(uint64(n0)) // size of encoded length
		n += n0 + n1
	}
	return
}

func (o *Buffer) enc_custom_bytes(p *Properties, base structPointer) error {
	i := structPointer_InterfaceRef(base, p.field, p.ctype)
	if i == nil {
		return ErrNil
	}
	custom := i.(Marshaler)
	data, err := custom.Marshal()
	if err != nil {
		return err
	}
	if data == nil {
		return ErrNil
	}
	o.buf = append(o.buf, p.tagcode...)
	o.EncodeRawBytes(data)
	return nil
}

func size_custom_bytes(p *Properties, base structPointer) (n int) {
	n += len(p.tagcode)
	i := structPointer_InterfaceRef(base, p.field, p.ctype)
	if i == nil {
		return 0
	}
	custom := i.(Marshaler)
	data, _ := custom.Marshal()
	n += sizeRawBytes(data)
	return
}

func (o *Buffer) enc_custom_ref_bytes(p *Properties, base structPointer) error {
	custom := structPointer_InterfaceAt(base, p.field, p.ctype).(Marshaler)
	data, err := custom.Marshal()
	if err != nil {
		return err
	}
	if data == nil {
		return ErrNil
	}
	o.buf = append(o.buf, p.tagcode...)
	o.EncodeRawBytes(data)
	return nil
}

func size_custom_ref_bytes(p *Properties, base structPointer) (n int) {
	n += len(p.tagcode)
	i := structPointer_InterfaceAt(base, p.field, p.ctype)
	if i == nil {
		return 0
	}
	custom := i.(Marshaler)
	data, _ := custom.Marshal()
	n += sizeRawBytes(data)
	return
}

func (o *Buffer) enc_custom_slice_bytes(p *Properties, base structPointer) error {
	inter := structPointer_InterfaceRef(base, p.field, p.ctype)
	if inter == nil {
		return ErrNil
	}
	slice := reflect.ValueOf(inter)
	l := slice.Len()
	for i := 0; i < l; i++ {
		v := slice.Index(i)
		custom := v.Interface().(Marshaler)
		data, err := custom.Marshal()
		if err != nil {
			return err
		}
		o.buf = append(o.buf, p.tagcode...)
		o.EncodeRawBytes(data)
	}
	return nil
}

func size_custom_slice_bytes(p *Properties, base structPointer) (n int) {
	inter := structPointer_InterfaceRef(base, p.field, p.ctype)
	if inter == nil {
		return 0
	}
	slice := reflect.ValueOf(inter)
	l := slice.Len()
	n += l * len(p.tagcode)
	for i := 0; i < l; i++ {
		v := slice.Index(i)
		custom := v.Interface().(Marshaler)
		data, _ := custom.Marshal()
		n += sizeRawBytes(data)
	}
	return
}
