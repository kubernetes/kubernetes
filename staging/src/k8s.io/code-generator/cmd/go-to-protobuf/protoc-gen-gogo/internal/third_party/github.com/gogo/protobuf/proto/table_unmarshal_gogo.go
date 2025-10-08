// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2018, The GoGo Authors. All rights reserved.
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

package proto

import (
	"io"
	"reflect"
)

func makeUnmarshalMessage(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return nil, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		// First read the message field to see if something is there.
		// The semantics of multiple submessages are weird.  Instead of
		// the last one winning (as it is for all other fields), multiple
		// submessages are merged.
		v := f // gogo: changed from v := f.getPointer()
		if v.isNil() {
			v = valToPointer(reflect.New(sub.typ))
			f.setPointer(v)
		}
		err := sub.unmarshal(v, b[:x])
		if err != nil {
			if r, ok := err.(*RequiredNotSetError); ok {
				r.field = name + "." + r.field
			} else {
				return nil, err
			}
		}
		return b[x:], err
	}
}

func makeUnmarshalMessageSlice(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return nil, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		v := valToPointer(reflect.New(sub.typ))
		err := sub.unmarshal(v, b[:x])
		if err != nil {
			if r, ok := err.(*RequiredNotSetError); ok {
				r.field = name + "." + r.field
			} else {
				return nil, err
			}
		}
		f.appendRef(v, sub.typ) // gogo: changed from f.appendPointer(v)
		return b[x:], err
	}
}

func makeUnmarshalCustomPtr(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return nil, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}

		s := f.asPointerTo(reflect.PtrTo(sub.typ)).Elem()
		s.Set(reflect.New(sub.typ))
		m := s.Interface().(custom)
		if err := m.Unmarshal(b[:x]); err != nil {
			return nil, err
		}
		return b[x:], nil
	}
}

func makeUnmarshalCustomSlice(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return nil, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		m := reflect.New(sub.typ)
		c := m.Interface().(custom)
		if err := c.Unmarshal(b[:x]); err != nil {
			return nil, err
		}
		v := valToPointer(m)
		f.appendRef(v, sub.typ)
		return b[x:], nil
	}
}

func makeUnmarshalCustom(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return nil, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}

		m := f.asPointerTo(sub.typ).Interface().(custom)
		if err := m.Unmarshal(b[:x]); err != nil {
			return nil, err
		}
		return b[x:], nil
	}
}

func makeUnmarshalTime(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return nil, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		m := &timestamp{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		t, err := timestampFromProto(m)
		if err != nil {
			return nil, err
		}
		s := f.asPointerTo(sub.typ).Elem()
		s.Set(reflect.ValueOf(t))
		return b[x:], nil
	}
}

func makeUnmarshalTimePtr(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return nil, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		m := &timestamp{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		t, err := timestampFromProto(m)
		if err != nil {
			return nil, err
		}
		s := f.asPointerTo(reflect.PtrTo(sub.typ)).Elem()
		s.Set(reflect.ValueOf(&t))
		return b[x:], nil
	}
}

func makeUnmarshalTimePtrSlice(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return nil, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		m := &timestamp{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		t, err := timestampFromProto(m)
		if err != nil {
			return nil, err
		}
		slice := f.getSlice(reflect.PtrTo(sub.typ))
		newSlice := reflect.Append(slice, reflect.ValueOf(&t))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeUnmarshalTimeSlice(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return nil, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		m := &timestamp{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		t, err := timestampFromProto(m)
		if err != nil {
			return nil, err
		}
		slice := f.getSlice(sub.typ)
		newSlice := reflect.Append(slice, reflect.ValueOf(t))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeUnmarshalDurationPtr(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return nil, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		m := &duration{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		d, err := durationFromProto(m)
		if err != nil {
			return nil, err
		}
		s := f.asPointerTo(reflect.PtrTo(sub.typ)).Elem()
		s.Set(reflect.ValueOf(&d))
		return b[x:], nil
	}
}

func makeUnmarshalDuration(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return nil, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		m := &duration{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		d, err := durationFromProto(m)
		if err != nil {
			return nil, err
		}
		s := f.asPointerTo(sub.typ).Elem()
		s.Set(reflect.ValueOf(d))
		return b[x:], nil
	}
}

func makeUnmarshalDurationPtrSlice(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return nil, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		m := &duration{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		d, err := durationFromProto(m)
		if err != nil {
			return nil, err
		}
		slice := f.getSlice(reflect.PtrTo(sub.typ))
		newSlice := reflect.Append(slice, reflect.ValueOf(&d))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeUnmarshalDurationSlice(sub *unmarshalInfo, name string) unmarshaler {
	return func(b []byte, f pointer, w int) ([]byte, error) {
		if w != WireBytes {
			return nil, errInternalBadWireType
		}
		x, n := decodeVarint(b)
		if n == 0 {
			return nil, io.ErrUnexpectedEOF
		}
		b = b[n:]
		if x > uint64(len(b)) {
			return nil, io.ErrUnexpectedEOF
		}
		m := &duration{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		d, err := durationFromProto(m)
		if err != nil {
			return nil, err
		}
		slice := f.getSlice(sub.typ)
		newSlice := reflect.Append(slice, reflect.ValueOf(d))
		slice.Set(newSlice)
		return b[x:], nil
	}
}
