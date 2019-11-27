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

// Protocol buffer comparison.

package proto

import (
	"bytes"
	"log"
	"reflect"
	"strings"
)

/*
Equal returns true iff protocol buffers a and b are equal.
The arguments must both be pointers to protocol buffer structs.

Equality is defined in this way:
  - Two messages are equal iff they are the same type,
    corresponding fields are equal, unknown field sets
    are equal, and extensions sets are equal.
  - Two set scalar fields are equal iff their values are equal.
    If the fields are of a floating-point type, remember that
    NaN != x for all x, including NaN. If the message is defined
    in a proto3 .proto file, fields are not "set"; specifically,
    zero length proto3 "bytes" fields are equal (nil == {}).
  - Two repeated fields are equal iff their lengths are the same,
    and their corresponding elements are equal. Note a "bytes" field,
    although represented by []byte, is not a repeated field and the
    rule for the scalar fields described above applies.
  - Two unset fields are equal.
  - Two unknown field sets are equal if their current
    encoded state is equal.
  - Two extension sets are equal iff they have corresponding
    elements that are pairwise equal.
  - Two map fields are equal iff their lengths are the same,
    and they contain the same set of elements. Zero-length map
    fields are equal.
  - Every other combination of things are not equal.

The return value is undefined if a and b are not protocol buffers.
*/
func Equal(a, b Message) bool {
	if a == nil || b == nil {
		return a == b
	}
	v1, v2 := reflect.ValueOf(a), reflect.ValueOf(b)
	if v1.Type() != v2.Type() {
		return false
	}
	if v1.Kind() == reflect.Ptr {
		if v1.IsNil() {
			return v2.IsNil()
		}
		if v2.IsNil() {
			return false
		}
		v1, v2 = v1.Elem(), v2.Elem()
	}
	if v1.Kind() != reflect.Struct {
		return false
	}
	return equalStruct(v1, v2)
}

// v1 and v2 are known to have the same type.
func equalStruct(v1, v2 reflect.Value) bool {
	sprop := GetProperties(v1.Type())
	for i := 0; i < v1.NumField(); i++ {
		f := v1.Type().Field(i)
		if strings.HasPrefix(f.Name, "XXX_") {
			continue
		}
		f1, f2 := v1.Field(i), v2.Field(i)
		if f.Type.Kind() == reflect.Ptr {
			if n1, n2 := f1.IsNil(), f2.IsNil(); n1 && n2 {
				// both unset
				continue
			} else if n1 != n2 {
				// set/unset mismatch
				return false
			}
			f1, f2 = f1.Elem(), f2.Elem()
		}
		if !equalAny(f1, f2, sprop.Prop[i]) {
			return false
		}
	}

	if em1 := v1.FieldByName("XXX_InternalExtensions"); em1.IsValid() {
		em2 := v2.FieldByName("XXX_InternalExtensions")
		if !equalExtensions(v1.Type(), em1.Interface().(XXX_InternalExtensions), em2.Interface().(XXX_InternalExtensions)) {
			return false
		}
	}

	if em1 := v1.FieldByName("XXX_extensions"); em1.IsValid() {
		em2 := v2.FieldByName("XXX_extensions")
		if !equalExtMap(v1.Type(), em1.Interface().(map[int32]Extension), em2.Interface().(map[int32]Extension)) {
			return false
		}
	}

	uf := v1.FieldByName("XXX_unrecognized")
	if !uf.IsValid() {
		return true
	}

	u1 := uf.Bytes()
	u2 := v2.FieldByName("XXX_unrecognized").Bytes()
	return bytes.Equal(u1, u2)
}

// v1 and v2 are known to have the same type.
// prop may be nil.
func equalAny(v1, v2 reflect.Value, prop *Properties) bool {
	if v1.Type() == protoMessageType {
		m1, _ := v1.Interface().(Message)
		m2, _ := v2.Interface().(Message)
		return Equal(m1, m2)
	}
	switch v1.Kind() {
	case reflect.Bool:
		return v1.Bool() == v2.Bool()
	case reflect.Float32, reflect.Float64:
		return v1.Float() == v2.Float()
	case reflect.Int32, reflect.Int64:
		return v1.Int() == v2.Int()
	case reflect.Interface:
		// Probably a oneof field; compare the inner values.
		n1, n2 := v1.IsNil(), v2.IsNil()
		if n1 || n2 {
			return n1 == n2
		}
		e1, e2 := v1.Elem(), v2.Elem()
		if e1.Type() != e2.Type() {
			return false
		}
		return equalAny(e1, e2, nil)
	case reflect.Map:
		if v1.Len() != v2.Len() {
			return false
		}
		for _, key := range v1.MapKeys() {
			val2 := v2.MapIndex(key)
			if !val2.IsValid() {
				// This key was not found in the second map.
				return false
			}
			if !equalAny(v1.MapIndex(key), val2, nil) {
				return false
			}
		}
		return true
	case reflect.Ptr:
		// Maps may have nil values in them, so check for nil.
		if v1.IsNil() && v2.IsNil() {
			return true
		}
		if v1.IsNil() != v2.IsNil() {
			return false
		}
		return equalAny(v1.Elem(), v2.Elem(), prop)
	case reflect.Slice:
		if v1.Type().Elem().Kind() == reflect.Uint8 {
			// short circuit: []byte

			// Edge case: if this is in a proto3 message, a zero length
			// bytes field is considered the zero value.
			if prop != nil && prop.proto3 && v1.Len() == 0 && v2.Len() == 0 {
				return true
			}
			if v1.IsNil() != v2.IsNil() {
				return false
			}
			return bytes.Equal(v1.Interface().([]byte), v2.Interface().([]byte))
		}

		if v1.Len() != v2.Len() {
			return false
		}
		for i := 0; i < v1.Len(); i++ {
			if !equalAny(v1.Index(i), v2.Index(i), prop) {
				return false
			}
		}
		return true
	case reflect.String:
		return v1.Interface().(string) == v2.Interface().(string)
	case reflect.Struct:
		return equalStruct(v1, v2)
	case reflect.Uint32, reflect.Uint64:
		return v1.Uint() == v2.Uint()
	}

	// unknown type, so not a protocol buffer
	log.Printf("proto: don't know how to compare %v", v1)
	return false
}

// base is the struct type that the extensions are based on.
// x1 and x2 are InternalExtensions.
func equalExtensions(base reflect.Type, x1, x2 XXX_InternalExtensions) bool {
	em1, _ := x1.extensionsRead()
	em2, _ := x2.extensionsRead()
	return equalExtMap(base, em1, em2)
}

func equalExtMap(base reflect.Type, em1, em2 map[int32]Extension) bool {
	if len(em1) != len(em2) {
		return false
	}

	for extNum, e1 := range em1 {
		e2, ok := em2[extNum]
		if !ok {
			return false
		}

		m1, m2 := e1.value, e2.value

		if m1 == nil && m2 == nil {
			// Both have only encoded form.
			if bytes.Equal(e1.enc, e2.enc) {
				continue
			}
			// The bytes are different, but the extensions might still be
			// equal. We need to decode them to compare.
		}

		if m1 != nil && m2 != nil {
			// Both are unencoded.
			if !equalAny(reflect.ValueOf(m1), reflect.ValueOf(m2), nil) {
				return false
			}
			continue
		}

		// At least one is encoded. To do a semantically correct comparison
		// we need to unmarshal them first.
		var desc *ExtensionDesc
		if m := extensionMaps[base]; m != nil {
			desc = m[extNum]
		}
		if desc == nil {
			// If both have only encoded form and the bytes are the same,
			// it is handled above. We get here when the bytes are different.
			// We don't know how to decode it, so just compare them as byte
			// slices.
			log.Printf("proto: don't know how to compare extension %d of %v", extNum, base)
			return false
		}
		var err error
		if m1 == nil {
			m1, err = decodeExtension(e1.enc, desc)
		}
		if m2 == nil && err == nil {
			m2, err = decodeExtension(e2.enc, desc)
		}
		if err != nil {
			// The encoded form is invalid.
			log.Printf("proto: badly encoded extension %d of %v: %v", extNum, base, err)
			return false
		}
		if !equalAny(reflect.ValueOf(m1), reflect.ValueOf(m2), nil) {
			return false
		}
	}

	return true
}
