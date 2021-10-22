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

func makeStdDoubleValueMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			t := ptr.asPointerTo(u.typ).Interface().(*float64)
			v := &float64Value{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			t := ptr.asPointerTo(u.typ).Interface().(*float64)
			v := &float64Value{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdDoubleValuePtrMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			if ptr.isNil() {
				return 0
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*float64)
			v := &float64Value{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			if ptr.isNil() {
				return b, nil
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*float64)
			v := &float64Value{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdDoubleValueSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(u.typ)
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(float64)
				v := &float64Value{t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(u.typ)
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(float64)
				v := &float64Value{t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdDoubleValuePtrSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*float64)
				v := &float64Value{*t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*float64)
				v := &float64Value{*t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdDoubleValueUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &float64Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(sub.typ).Elem()
		s.Set(reflect.ValueOf(m.Value))
		return b[x:], nil
	}
}

func makeStdDoubleValuePtrUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &float64Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(reflect.PtrTo(sub.typ)).Elem()
		s.Set(reflect.ValueOf(&m.Value))
		return b[x:], nil
	}
}

func makeStdDoubleValuePtrSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &float64Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(reflect.PtrTo(sub.typ))
		newSlice := reflect.Append(slice, reflect.ValueOf(&m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdDoubleValueSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &float64Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(sub.typ)
		newSlice := reflect.Append(slice, reflect.ValueOf(m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdFloatValueMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			t := ptr.asPointerTo(u.typ).Interface().(*float32)
			v := &float32Value{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			t := ptr.asPointerTo(u.typ).Interface().(*float32)
			v := &float32Value{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdFloatValuePtrMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			if ptr.isNil() {
				return 0
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*float32)
			v := &float32Value{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			if ptr.isNil() {
				return b, nil
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*float32)
			v := &float32Value{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdFloatValueSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(u.typ)
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(float32)
				v := &float32Value{t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(u.typ)
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(float32)
				v := &float32Value{t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdFloatValuePtrSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*float32)
				v := &float32Value{*t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*float32)
				v := &float32Value{*t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdFloatValueUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &float32Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(sub.typ).Elem()
		s.Set(reflect.ValueOf(m.Value))
		return b[x:], nil
	}
}

func makeStdFloatValuePtrUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &float32Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(reflect.PtrTo(sub.typ)).Elem()
		s.Set(reflect.ValueOf(&m.Value))
		return b[x:], nil
	}
}

func makeStdFloatValuePtrSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &float32Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(reflect.PtrTo(sub.typ))
		newSlice := reflect.Append(slice, reflect.ValueOf(&m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdFloatValueSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &float32Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(sub.typ)
		newSlice := reflect.Append(slice, reflect.ValueOf(m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdInt64ValueMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			t := ptr.asPointerTo(u.typ).Interface().(*int64)
			v := &int64Value{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			t := ptr.asPointerTo(u.typ).Interface().(*int64)
			v := &int64Value{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdInt64ValuePtrMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			if ptr.isNil() {
				return 0
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*int64)
			v := &int64Value{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			if ptr.isNil() {
				return b, nil
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*int64)
			v := &int64Value{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdInt64ValueSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(u.typ)
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(int64)
				v := &int64Value{t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(u.typ)
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(int64)
				v := &int64Value{t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdInt64ValuePtrSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*int64)
				v := &int64Value{*t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*int64)
				v := &int64Value{*t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdInt64ValueUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &int64Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(sub.typ).Elem()
		s.Set(reflect.ValueOf(m.Value))
		return b[x:], nil
	}
}

func makeStdInt64ValuePtrUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &int64Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(reflect.PtrTo(sub.typ)).Elem()
		s.Set(reflect.ValueOf(&m.Value))
		return b[x:], nil
	}
}

func makeStdInt64ValuePtrSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &int64Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(reflect.PtrTo(sub.typ))
		newSlice := reflect.Append(slice, reflect.ValueOf(&m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdInt64ValueSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &int64Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(sub.typ)
		newSlice := reflect.Append(slice, reflect.ValueOf(m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdUInt64ValueMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			t := ptr.asPointerTo(u.typ).Interface().(*uint64)
			v := &uint64Value{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			t := ptr.asPointerTo(u.typ).Interface().(*uint64)
			v := &uint64Value{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdUInt64ValuePtrMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			if ptr.isNil() {
				return 0
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*uint64)
			v := &uint64Value{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			if ptr.isNil() {
				return b, nil
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*uint64)
			v := &uint64Value{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdUInt64ValueSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(u.typ)
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(uint64)
				v := &uint64Value{t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(u.typ)
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(uint64)
				v := &uint64Value{t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdUInt64ValuePtrSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*uint64)
				v := &uint64Value{*t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*uint64)
				v := &uint64Value{*t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdUInt64ValueUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &uint64Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(sub.typ).Elem()
		s.Set(reflect.ValueOf(m.Value))
		return b[x:], nil
	}
}

func makeStdUInt64ValuePtrUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &uint64Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(reflect.PtrTo(sub.typ)).Elem()
		s.Set(reflect.ValueOf(&m.Value))
		return b[x:], nil
	}
}

func makeStdUInt64ValuePtrSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &uint64Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(reflect.PtrTo(sub.typ))
		newSlice := reflect.Append(slice, reflect.ValueOf(&m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdUInt64ValueSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &uint64Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(sub.typ)
		newSlice := reflect.Append(slice, reflect.ValueOf(m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdInt32ValueMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			t := ptr.asPointerTo(u.typ).Interface().(*int32)
			v := &int32Value{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			t := ptr.asPointerTo(u.typ).Interface().(*int32)
			v := &int32Value{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdInt32ValuePtrMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			if ptr.isNil() {
				return 0
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*int32)
			v := &int32Value{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			if ptr.isNil() {
				return b, nil
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*int32)
			v := &int32Value{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdInt32ValueSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(u.typ)
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(int32)
				v := &int32Value{t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(u.typ)
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(int32)
				v := &int32Value{t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdInt32ValuePtrSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*int32)
				v := &int32Value{*t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*int32)
				v := &int32Value{*t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdInt32ValueUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &int32Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(sub.typ).Elem()
		s.Set(reflect.ValueOf(m.Value))
		return b[x:], nil
	}
}

func makeStdInt32ValuePtrUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &int32Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(reflect.PtrTo(sub.typ)).Elem()
		s.Set(reflect.ValueOf(&m.Value))
		return b[x:], nil
	}
}

func makeStdInt32ValuePtrSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &int32Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(reflect.PtrTo(sub.typ))
		newSlice := reflect.Append(slice, reflect.ValueOf(&m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdInt32ValueSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &int32Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(sub.typ)
		newSlice := reflect.Append(slice, reflect.ValueOf(m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdUInt32ValueMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			t := ptr.asPointerTo(u.typ).Interface().(*uint32)
			v := &uint32Value{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			t := ptr.asPointerTo(u.typ).Interface().(*uint32)
			v := &uint32Value{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdUInt32ValuePtrMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			if ptr.isNil() {
				return 0
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*uint32)
			v := &uint32Value{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			if ptr.isNil() {
				return b, nil
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*uint32)
			v := &uint32Value{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdUInt32ValueSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(u.typ)
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(uint32)
				v := &uint32Value{t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(u.typ)
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(uint32)
				v := &uint32Value{t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdUInt32ValuePtrSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*uint32)
				v := &uint32Value{*t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*uint32)
				v := &uint32Value{*t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdUInt32ValueUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &uint32Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(sub.typ).Elem()
		s.Set(reflect.ValueOf(m.Value))
		return b[x:], nil
	}
}

func makeStdUInt32ValuePtrUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &uint32Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(reflect.PtrTo(sub.typ)).Elem()
		s.Set(reflect.ValueOf(&m.Value))
		return b[x:], nil
	}
}

func makeStdUInt32ValuePtrSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &uint32Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(reflect.PtrTo(sub.typ))
		newSlice := reflect.Append(slice, reflect.ValueOf(&m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdUInt32ValueSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &uint32Value{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(sub.typ)
		newSlice := reflect.Append(slice, reflect.ValueOf(m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdBoolValueMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			t := ptr.asPointerTo(u.typ).Interface().(*bool)
			v := &boolValue{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			t := ptr.asPointerTo(u.typ).Interface().(*bool)
			v := &boolValue{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdBoolValuePtrMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			if ptr.isNil() {
				return 0
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*bool)
			v := &boolValue{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			if ptr.isNil() {
				return b, nil
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*bool)
			v := &boolValue{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdBoolValueSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(u.typ)
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(bool)
				v := &boolValue{t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(u.typ)
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(bool)
				v := &boolValue{t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdBoolValuePtrSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*bool)
				v := &boolValue{*t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*bool)
				v := &boolValue{*t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdBoolValueUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &boolValue{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(sub.typ).Elem()
		s.Set(reflect.ValueOf(m.Value))
		return b[x:], nil
	}
}

func makeStdBoolValuePtrUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &boolValue{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(reflect.PtrTo(sub.typ)).Elem()
		s.Set(reflect.ValueOf(&m.Value))
		return b[x:], nil
	}
}

func makeStdBoolValuePtrSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &boolValue{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(reflect.PtrTo(sub.typ))
		newSlice := reflect.Append(slice, reflect.ValueOf(&m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdBoolValueSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &boolValue{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(sub.typ)
		newSlice := reflect.Append(slice, reflect.ValueOf(m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdStringValueMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			t := ptr.asPointerTo(u.typ).Interface().(*string)
			v := &stringValue{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			t := ptr.asPointerTo(u.typ).Interface().(*string)
			v := &stringValue{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdStringValuePtrMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			if ptr.isNil() {
				return 0
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*string)
			v := &stringValue{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			if ptr.isNil() {
				return b, nil
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*string)
			v := &stringValue{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdStringValueSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(u.typ)
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(string)
				v := &stringValue{t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(u.typ)
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(string)
				v := &stringValue{t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdStringValuePtrSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*string)
				v := &stringValue{*t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*string)
				v := &stringValue{*t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdStringValueUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &stringValue{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(sub.typ).Elem()
		s.Set(reflect.ValueOf(m.Value))
		return b[x:], nil
	}
}

func makeStdStringValuePtrUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &stringValue{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(reflect.PtrTo(sub.typ)).Elem()
		s.Set(reflect.ValueOf(&m.Value))
		return b[x:], nil
	}
}

func makeStdStringValuePtrSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &stringValue{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(reflect.PtrTo(sub.typ))
		newSlice := reflect.Append(slice, reflect.ValueOf(&m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdStringValueSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &stringValue{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(sub.typ)
		newSlice := reflect.Append(slice, reflect.ValueOf(m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdBytesValueMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			t := ptr.asPointerTo(u.typ).Interface().(*[]byte)
			v := &bytesValue{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			t := ptr.asPointerTo(u.typ).Interface().(*[]byte)
			v := &bytesValue{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdBytesValuePtrMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			if ptr.isNil() {
				return 0
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*[]byte)
			v := &bytesValue{*t}
			siz := Size(v)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			if ptr.isNil() {
				return b, nil
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*[]byte)
			v := &bytesValue{*t}
			buf, err := Marshal(v)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeStdBytesValueSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(u.typ)
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().([]byte)
				v := &bytesValue{t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(u.typ)
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().([]byte)
				v := &bytesValue{t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdBytesValuePtrSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*[]byte)
				v := &bytesValue{*t}
				siz := Size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*[]byte)
				v := &bytesValue{*t}
				siz := Size(v)
				buf, err := Marshal(v)
				if err != nil {
					return nil, err
				}
				b = appendVarint(b, wiretag)
				b = appendVarint(b, uint64(siz))
				b = append(b, buf...)
			}

			return b, nil
		}
}

func makeStdBytesValueUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &bytesValue{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(sub.typ).Elem()
		s.Set(reflect.ValueOf(m.Value))
		return b[x:], nil
	}
}

func makeStdBytesValuePtrUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &bytesValue{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		s := f.asPointerTo(reflect.PtrTo(sub.typ)).Elem()
		s.Set(reflect.ValueOf(&m.Value))
		return b[x:], nil
	}
}

func makeStdBytesValuePtrSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &bytesValue{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(reflect.PtrTo(sub.typ))
		newSlice := reflect.Append(slice, reflect.ValueOf(&m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}

func makeStdBytesValueSliceUnmarshaler(sub *unmarshalInfo, name string) unmarshaler {
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
		m := &bytesValue{}
		if err := Unmarshal(b[:x], m); err != nil {
			return nil, err
		}
		slice := f.getSlice(sub.typ)
		newSlice := reflect.Append(slice, reflect.ValueOf(m.Value))
		slice.Set(newSlice)
		return b[x:], nil
	}
}
