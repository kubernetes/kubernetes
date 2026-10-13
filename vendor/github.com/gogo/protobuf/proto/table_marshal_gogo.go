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
	"reflect"
	"time"
)

// makeMessageRefMarshaler differs a bit from makeMessageMarshaler
// It marshal a message T instead of a *T
func makeMessageRefMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			siz := u.size(ptr)
			return siz + SizeVarint(uint64(siz)) + tagsize
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			b = appendVarint(b, wiretag)
			siz := u.cachedsize(ptr)
			b = appendVarint(b, uint64(siz))
			return u.marshal(b, ptr, deterministic)
		}
}

// makeMessageRefSliceMarshaler differs quite a lot from makeMessageSliceMarshaler
// It marshals a slice of messages []T instead of []*T
func makeMessageRefSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(u.typ)
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				e := elem.Interface()
				v := toAddrPointer(&e, false)
				siz := u.size(v)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(u.typ)
			var err, errreq error
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				e := elem.Interface()
				v := toAddrPointer(&e, false)
				b = appendVarint(b, wiretag)
				siz := u.size(v)
				b = appendVarint(b, uint64(siz))
				b, err = u.marshal(b, v, deterministic)

				if err != nil {
					if _, ok := err.(*RequiredNotSetError); ok {
						// Required field in submessage is not set.
						// We record the error but keep going, to give a complete marshaling.
						if errreq == nil {
							errreq = err
						}
						continue
					}
					if err == ErrNil {
						err = errRepeatedHasNil
					}
					return b, err
				}
			}

			return b, errreq
		}
}

func makeCustomPtrMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			if ptr.isNil() {
				return 0
			}
			m := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(custom)
			siz := m.Size()
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			if ptr.isNil() {
				return b, nil
			}
			m := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(custom)
			siz := m.Size()
			buf, err := m.Marshal()
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(siz))
			b = append(b, buf...)
			return b, nil
		}
}

func makeCustomMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			m := ptr.asPointerTo(u.typ).Interface().(custom)
			siz := m.Size()
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			m := ptr.asPointerTo(u.typ).Interface().(custom)
			siz := m.Size()
			buf, err := m.Marshal()
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(siz))
			b = append(b, buf...)
			return b, nil
		}
}

func makeTimeMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			t := ptr.asPointerTo(u.typ).Interface().(*time.Time)
			ts, err := timestampProto(*t)
			if err != nil {
				return 0
			}
			siz := Size(ts)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			t := ptr.asPointerTo(u.typ).Interface().(*time.Time)
			ts, err := timestampProto(*t)
			if err != nil {
				return nil, err
			}
			buf, err := Marshal(ts)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeTimePtrMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			if ptr.isNil() {
				return 0
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*time.Time)
			ts, err := timestampProto(*t)
			if err != nil {
				return 0
			}
			siz := Size(ts)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			if ptr.isNil() {
				return b, nil
			}
			t := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*time.Time)
			ts, err := timestampProto(*t)
			if err != nil {
				return nil, err
			}
			buf, err := Marshal(ts)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeTimeSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(u.typ)
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(time.Time)
				ts, err := timestampProto(t)
				if err != nil {
					return 0
				}
				siz := Size(ts)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(u.typ)
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(time.Time)
				ts, err := timestampProto(t)
				if err != nil {
					return nil, err
				}
				siz := Size(ts)
				buf, err := Marshal(ts)
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

func makeTimePtrSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*time.Time)
				ts, err := timestampProto(*t)
				if err != nil {
					return 0
				}
				siz := Size(ts)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				t := elem.Interface().(*time.Time)
				ts, err := timestampProto(*t)
				if err != nil {
					return nil, err
				}
				siz := Size(ts)
				buf, err := Marshal(ts)
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

func makeDurationMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			d := ptr.asPointerTo(u.typ).Interface().(*time.Duration)
			dur := durationProto(*d)
			siz := Size(dur)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			d := ptr.asPointerTo(u.typ).Interface().(*time.Duration)
			dur := durationProto(*d)
			buf, err := Marshal(dur)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeDurationPtrMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			if ptr.isNil() {
				return 0
			}
			d := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*time.Duration)
			dur := durationProto(*d)
			siz := Size(dur)
			return tagsize + SizeVarint(uint64(siz)) + siz
		}, func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			if ptr.isNil() {
				return b, nil
			}
			d := ptr.asPointerTo(reflect.PtrTo(u.typ)).Elem().Interface().(*time.Duration)
			dur := durationProto(*d)
			buf, err := Marshal(dur)
			if err != nil {
				return nil, err
			}
			b = appendVarint(b, wiretag)
			b = appendVarint(b, uint64(len(buf)))
			b = append(b, buf...)
			return b, nil
		}
}

func makeDurationSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(u.typ)
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				d := elem.Interface().(time.Duration)
				dur := durationProto(d)
				siz := Size(dur)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(u.typ)
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				d := elem.Interface().(time.Duration)
				dur := durationProto(d)
				siz := Size(dur)
				buf, err := Marshal(dur)
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

func makeDurationPtrSliceMarshaler(u *marshalInfo) (sizer, marshaler) {
	return func(ptr pointer, tagsize int) int {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			n := 0
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				d := elem.Interface().(*time.Duration)
				dur := durationProto(*d)
				siz := Size(dur)
				n += siz + SizeVarint(uint64(siz)) + tagsize
			}
			return n
		},
		func(b []byte, ptr pointer, wiretag uint64, deterministic bool) ([]byte, error) {
			s := ptr.getSlice(reflect.PtrTo(u.typ))
			for i := 0; i < s.Len(); i++ {
				elem := s.Index(i)
				d := elem.Interface().(*time.Duration)
				dur := durationProto(*d)
				siz := Size(dur)
				buf, err := Marshal(dur)
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
