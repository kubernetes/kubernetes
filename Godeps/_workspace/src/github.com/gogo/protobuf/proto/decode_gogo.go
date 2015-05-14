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

package proto

import (
	"reflect"
)

// Decode a reference to a bool pointer.
func (o *Buffer) dec_ref_bool(p *Properties, base structPointer) error {
	u, err := p.valDec(o)
	if err != nil {
		return err
	}
	if len(o.bools) == 0 {
		o.bools = make([]bool, boolPoolSize)
	}
	o.bools[0] = u != 0
	*structPointer_RefBool(base, p.field) = o.bools[0]
	o.bools = o.bools[1:]
	return nil
}

// Decode a reference to an int32 pointer.
func (o *Buffer) dec_ref_int32(p *Properties, base structPointer) error {
	u, err := p.valDec(o)
	if err != nil {
		return err
	}
	refWord32_Set(structPointer_RefWord32(base, p.field), o, uint32(u))
	return nil
}

// Decode a reference to an int64 pointer.
func (o *Buffer) dec_ref_int64(p *Properties, base structPointer) error {
	u, err := p.valDec(o)
	if err != nil {
		return err
	}
	refWord64_Set(structPointer_RefWord64(base, p.field), o, u)
	return nil
}

// Decode a reference to a string pointer.
func (o *Buffer) dec_ref_string(p *Properties, base structPointer) error {
	s, err := o.DecodeStringBytes()
	if err != nil {
		return err
	}
	*structPointer_RefString(base, p.field) = s
	return nil
}

// Decode a reference to a struct pointer.
func (o *Buffer) dec_ref_struct_message(p *Properties, base structPointer) (err error) {
	raw, e := o.DecodeRawBytes(false)
	if e != nil {
		return e
	}

	// If the object can unmarshal itself, let it.
	if p.isUnmarshaler {
		panic("not supported, since this is a pointer receiver")
	}

	obuf := o.buf
	oi := o.index
	o.buf = raw
	o.index = 0

	bas := structPointer_FieldPointer(base, p.field)

	err = o.unmarshalType(p.stype, p.sprop, false, bas)
	o.buf = obuf
	o.index = oi

	return err
}

// Decode a slice of references to struct pointers ([]struct).
func (o *Buffer) dec_slice_ref_struct(p *Properties, is_group bool, base structPointer) error {
	newBas := appendStructPointer(base, p.field, p.sstype)

	if is_group {
		panic("not supported, maybe in future, if requested.")
	}

	raw, err := o.DecodeRawBytes(false)
	if err != nil {
		return err
	}

	// If the object can unmarshal itself, let it.
	if p.isUnmarshaler {
		panic("not supported, since this is not a pointer receiver.")
	}

	obuf := o.buf
	oi := o.index
	o.buf = raw
	o.index = 0

	err = o.unmarshalType(p.stype, p.sprop, is_group, newBas)

	o.buf = obuf
	o.index = oi

	return err
}

// Decode a slice of references to struct pointers.
func (o *Buffer) dec_slice_ref_struct_message(p *Properties, base structPointer) error {
	return o.dec_slice_ref_struct(p, false, base)
}

func setPtrCustomType(base structPointer, f field, v interface{}) {
	if v == nil {
		return
	}
	structPointer_SetStructPointer(base, f, structPointer(reflect.ValueOf(v).Pointer()))
}

func setCustomType(base structPointer, f field, value interface{}) {
	if value == nil {
		return
	}
	v := reflect.ValueOf(value).Elem()
	t := reflect.TypeOf(value).Elem()
	kind := t.Kind()
	switch kind {
	case reflect.Slice:
		slice := reflect.MakeSlice(t, v.Len(), v.Cap())
		reflect.Copy(slice, v)
		oldHeader := structPointer_GetSliceHeader(base, f)
		oldHeader.Data = slice.Pointer()
		oldHeader.Len = v.Len()
		oldHeader.Cap = v.Cap()
	default:
		l := 1
		size := reflect.TypeOf(value).Elem().Size()
		if kind == reflect.Array {
			l = reflect.TypeOf(value).Elem().Len()
			size = reflect.TypeOf(value).Size()
		}
		total := int(size) * l
		structPointer_Copy(toStructPointer(reflect.ValueOf(value)), structPointer_Add(base, f), total)
	}
}

func (o *Buffer) dec_custom_bytes(p *Properties, base structPointer) error {
	b, err := o.DecodeRawBytes(true)
	if err != nil {
		return err
	}
	i := reflect.New(p.ctype.Elem()).Interface()
	custom := (i).(Unmarshaler)
	if err := custom.Unmarshal(b); err != nil {
		return err
	}
	setPtrCustomType(base, p.field, custom)
	return nil
}

func (o *Buffer) dec_custom_ref_bytes(p *Properties, base structPointer) error {
	b, err := o.DecodeRawBytes(true)
	if err != nil {
		return err
	}
	i := reflect.New(p.ctype).Interface()
	custom := (i).(Unmarshaler)
	if err := custom.Unmarshal(b); err != nil {
		return err
	}
	if custom != nil {
		setCustomType(base, p.field, custom)
	}
	return nil
}

// Decode a slice of bytes ([]byte) into a slice of custom types.
func (o *Buffer) dec_custom_slice_bytes(p *Properties, base structPointer) error {
	b, err := o.DecodeRawBytes(true)
	if err != nil {
		return err
	}
	i := reflect.New(p.ctype.Elem()).Interface()
	custom := (i).(Unmarshaler)
	if err := custom.Unmarshal(b); err != nil {
		return err
	}
	newBas := appendStructPointer(base, p.field, p.ctype)

	setCustomType(newBas, 0, custom)

	return nil
}
