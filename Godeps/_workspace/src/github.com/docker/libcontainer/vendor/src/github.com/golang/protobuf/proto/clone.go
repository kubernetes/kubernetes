// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2011 The Go Authors.  All rights reserved.
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

// Protocol buffer deep copy and merge.
// TODO: MessageSet and RawMessage.

package proto

import (
	"log"
	"reflect"
	"strings"
)

// Clone returns a deep copy of a protocol buffer.
func Clone(pb Message) Message {
	in := reflect.ValueOf(pb)
	if in.IsNil() {
		return pb
	}

	out := reflect.New(in.Type().Elem())
	// out is empty so a merge is a deep copy.
	mergeStruct(out.Elem(), in.Elem())
	return out.Interface().(Message)
}

// Merge merges src into dst.
// Required and optional fields that are set in src will be set to that value in dst.
// Elements of repeated fields will be appended.
// Merge panics if src and dst are not the same type, or if dst is nil.
func Merge(dst, src Message) {
	in := reflect.ValueOf(src)
	out := reflect.ValueOf(dst)
	if out.IsNil() {
		panic("proto: nil destination")
	}
	if in.Type() != out.Type() {
		// Explicit test prior to mergeStruct so that mistyped nils will fail
		panic("proto: type mismatch")
	}
	if in.IsNil() {
		// Merging nil into non-nil is a quiet no-op
		return
	}
	mergeStruct(out.Elem(), in.Elem())
}

func mergeStruct(out, in reflect.Value) {
	for i := 0; i < in.NumField(); i++ {
		f := in.Type().Field(i)
		if strings.HasPrefix(f.Name, "XXX_") {
			continue
		}
		mergeAny(out.Field(i), in.Field(i))
	}

	if emIn, ok := in.Addr().Interface().(extendableProto); ok {
		emOut := out.Addr().Interface().(extendableProto)
		mergeExtension(emOut.ExtensionMap(), emIn.ExtensionMap())
	}

	uf := in.FieldByName("XXX_unrecognized")
	if !uf.IsValid() {
		return
	}
	uin := uf.Bytes()
	if len(uin) > 0 {
		out.FieldByName("XXX_unrecognized").SetBytes(append([]byte(nil), uin...))
	}
}

func mergeAny(out, in reflect.Value) {
	if in.Type() == protoMessageType {
		if !in.IsNil() {
			if out.IsNil() {
				out.Set(reflect.ValueOf(Clone(in.Interface().(Message))))
			} else {
				Merge(out.Interface().(Message), in.Interface().(Message))
			}
		}
		return
	}
	switch in.Kind() {
	case reflect.Bool, reflect.Float32, reflect.Float64, reflect.Int32, reflect.Int64,
		reflect.String, reflect.Uint32, reflect.Uint64:
		out.Set(in)
	case reflect.Map:
		if in.Len() == 0 {
			return
		}
		if out.IsNil() {
			out.Set(reflect.MakeMap(in.Type()))
		}
		// For maps with value types of *T or []byte we need to deep copy each value.
		elemKind := in.Type().Elem().Kind()
		for _, key := range in.MapKeys() {
			var val reflect.Value
			switch elemKind {
			case reflect.Ptr:
				val = reflect.New(in.Type().Elem().Elem())
				mergeAny(val, in.MapIndex(key))
			case reflect.Slice:
				val = in.MapIndex(key)
				val = reflect.ValueOf(append([]byte{}, val.Bytes()...))
			default:
				val = in.MapIndex(key)
			}
			out.SetMapIndex(key, val)
		}
	case reflect.Ptr:
		if in.IsNil() {
			return
		}
		if out.IsNil() {
			out.Set(reflect.New(in.Elem().Type()))
		}
		mergeAny(out.Elem(), in.Elem())
	case reflect.Slice:
		if in.IsNil() {
			return
		}
		if in.Type().Elem().Kind() == reflect.Uint8 {
			// []byte is a scalar bytes field, not a repeated field.
			// Make a deep copy.
			// Append to []byte{} instead of []byte(nil) so that we never end up
			// with a nil result.
			out.SetBytes(append([]byte{}, in.Bytes()...))
			return
		}
		n := in.Len()
		if out.IsNil() {
			out.Set(reflect.MakeSlice(in.Type(), 0, n))
		}
		switch in.Type().Elem().Kind() {
		case reflect.Bool, reflect.Float32, reflect.Float64, reflect.Int32, reflect.Int64,
			reflect.String, reflect.Uint32, reflect.Uint64:
			out.Set(reflect.AppendSlice(out, in))
		default:
			for i := 0; i < n; i++ {
				x := reflect.Indirect(reflect.New(in.Type().Elem()))
				mergeAny(x, in.Index(i))
				out.Set(reflect.Append(out, x))
			}
		}
	case reflect.Struct:
		mergeStruct(out, in)
	default:
		// unknown type, so not a protocol buffer
		log.Printf("proto: don't know how to copy %v", in)
	}
}

func mergeExtension(out, in map[int32]Extension) {
	for extNum, eIn := range in {
		eOut := Extension{desc: eIn.desc}
		if eIn.value != nil {
			v := reflect.New(reflect.TypeOf(eIn.value)).Elem()
			mergeAny(v, reflect.ValueOf(eIn.value))
			eOut.value = v.Interface()
		}
		if eIn.enc != nil {
			eOut.enc = make([]byte, len(eIn.enc))
			copy(eOut.enc, eIn.enc)
		}

		out[extNum] = eOut
	}
}
