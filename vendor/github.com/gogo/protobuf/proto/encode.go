// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2010 The Go Authors.  All rights reserved.
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

package proto

/*
 * Routines for encoding data into the wire format for protocol buffers.
 */

import (
	"errors"
	"fmt"
	"reflect"
	"sort"
)

// RequiredNotSetError is the error returned if Marshal is called with
// a protocol buffer struct whose required fields have not
// all been initialized. It is also the error returned if Unmarshal is
// called with an encoded protocol buffer that does not include all the
// required fields.
//
// When printed, RequiredNotSetError reports the first unset required field in a
// message. If the field cannot be precisely determined, it is reported as
// "{Unknown}".
type RequiredNotSetError struct {
	field string
}

func (e *RequiredNotSetError) Error() string {
	return fmt.Sprintf("proto: required field %q not set", e.field)
}

var (
	// errRepeatedHasNil is the error returned if Marshal is called with
	// a struct with a repeated field containing a nil element.
	errRepeatedHasNil = errors.New("proto: repeated field has nil element")

	// errOneofHasNil is the error returned if Marshal is called with
	// a struct with a oneof field containing a nil element.
	errOneofHasNil = errors.New("proto: oneof field has nil value")

	// ErrNil is the error returned if Marshal is called with nil.
	ErrNil = errors.New("proto: Marshal called with nil")

	// ErrTooLarge is the error returned if Marshal is called with a
	// message that encodes to >2GB.
	ErrTooLarge = errors.New("proto: message encodes to over 2 GB")
)

// The fundamental encoders that put bytes on the wire.
// Those that take integer types all accept uint64 and are
// therefore of type valueEncoder.

const maxVarintBytes = 10 // maximum length of a varint

// maxMarshalSize is the largest allowed size of an encoded protobuf,
// since C++ and Java use signed int32s for the size.
const maxMarshalSize = 1<<31 - 1

// EncodeVarint returns the varint encoding of x.
// This is the format for the
// int32, int64, uint32, uint64, bool, and enum
// protocol buffer types.
// Not used by the package itself, but helpful to clients
// wishing to use the same encoding.
func EncodeVarint(x uint64) []byte {
	var buf [maxVarintBytes]byte
	var n int
	for n = 0; x > 127; n++ {
		buf[n] = 0x80 | uint8(x&0x7F)
		x >>= 7
	}
	buf[n] = uint8(x)
	n++
	return buf[0:n]
}

// EncodeVarint writes a varint-encoded integer to the Buffer.
// This is the format for the
// int32, int64, uint32, uint64, bool, and enum
// protocol buffer types.
func (p *Buffer) EncodeVarint(x uint64) error {
	for x >= 1<<7 {
		p.buf = append(p.buf, uint8(x&0x7f|0x80))
		x >>= 7
	}
	p.buf = append(p.buf, uint8(x))
	return nil
}

// SizeVarint returns the varint encoding size of an integer.
func SizeVarint(x uint64) int {
	return sizeVarint(x)
}

func sizeVarint(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}

// EncodeFixed64 writes a 64-bit integer to the Buffer.
// This is the format for the
// fixed64, sfixed64, and double protocol buffer types.
func (p *Buffer) EncodeFixed64(x uint64) error {
	p.buf = append(p.buf,
		uint8(x),
		uint8(x>>8),
		uint8(x>>16),
		uint8(x>>24),
		uint8(x>>32),
		uint8(x>>40),
		uint8(x>>48),
		uint8(x>>56))
	return nil
}

func sizeFixed64(x uint64) int {
	return 8
}

// EncodeFixed32 writes a 32-bit integer to the Buffer.
// This is the format for the
// fixed32, sfixed32, and float protocol buffer types.
func (p *Buffer) EncodeFixed32(x uint64) error {
	p.buf = append(p.buf,
		uint8(x),
		uint8(x>>8),
		uint8(x>>16),
		uint8(x>>24))
	return nil
}

func sizeFixed32(x uint64) int {
	return 4
}

// EncodeZigzag64 writes a zigzag-encoded 64-bit integer
// to the Buffer.
// This is the format used for the sint64 protocol buffer type.
func (p *Buffer) EncodeZigzag64(x uint64) error {
	// use signed number to get arithmetic right shift.
	return p.EncodeVarint((x << 1) ^ uint64((int64(x) >> 63)))
}

func sizeZigzag64(x uint64) int {
	return sizeVarint((x << 1) ^ uint64((int64(x) >> 63)))
}

// EncodeZigzag32 writes a zigzag-encoded 32-bit integer
// to the Buffer.
// This is the format used for the sint32 protocol buffer type.
func (p *Buffer) EncodeZigzag32(x uint64) error {
	// use signed number to get arithmetic right shift.
	return p.EncodeVarint(uint64((uint32(x) << 1) ^ uint32((int32(x) >> 31))))
}

func sizeZigzag32(x uint64) int {
	return sizeVarint(uint64((uint32(x) << 1) ^ uint32((int32(x) >> 31))))
}

// EncodeRawBytes writes a count-delimited byte buffer to the Buffer.
// This is the format used for the bytes protocol buffer
// type and for embedded messages.
func (p *Buffer) EncodeRawBytes(b []byte) error {
	p.EncodeVarint(uint64(len(b)))
	p.buf = append(p.buf, b...)
	return nil
}

func sizeRawBytes(b []byte) int {
	return sizeVarint(uint64(len(b))) +
		len(b)
}

// EncodeStringBytes writes an encoded string to the Buffer.
// This is the format used for the proto2 string type.
func (p *Buffer) EncodeStringBytes(s string) error {
	p.EncodeVarint(uint64(len(s)))
	p.buf = append(p.buf, s...)
	return nil
}

func sizeStringBytes(s string) int {
	return sizeVarint(uint64(len(s))) +
		len(s)
}

// Marshaler is the interface representing objects that can marshal themselves.
type Marshaler interface {
	Marshal() ([]byte, error)
}

// Marshal takes the protocol buffer
// and encodes it into the wire format, returning the data.
func Marshal(pb Message) ([]byte, error) {
	// Can the object marshal itself?
	if m, ok := pb.(Marshaler); ok {
		return m.Marshal()
	}
	p := NewBuffer(nil)
	err := p.Marshal(pb)
	if p.buf == nil && err == nil {
		// Return a non-nil slice on success.
		return []byte{}, nil
	}
	return p.buf, err
}

// EncodeMessage writes the protocol buffer to the Buffer,
// prefixed by a varint-encoded length.
func (p *Buffer) EncodeMessage(pb Message) error {
	t, base, err := getbase(pb)
	if structPointer_IsNil(base) {
		return ErrNil
	}
	if err == nil {
		var state errorState
		err = p.enc_len_struct(GetProperties(t.Elem()), base, &state)
	}
	return err
}

// Marshal takes the protocol buffer
// and encodes it into the wire format, writing the result to the
// Buffer.
func (p *Buffer) Marshal(pb Message) error {
	// Can the object marshal itself?
	if m, ok := pb.(Marshaler); ok {
		data, err := m.Marshal()
		p.buf = append(p.buf, data...)
		return err
	}

	t, base, err := getbase(pb)
	if structPointer_IsNil(base) {
		return ErrNil
	}
	if err == nil {
		err = p.enc_struct(GetProperties(t.Elem()), base)
	}

	if collectStats {
		(stats).Encode++ // Parens are to work around a goimports bug.
	}

	if len(p.buf) > maxMarshalSize {
		return ErrTooLarge
	}
	return err
}

// Size returns the encoded size of a protocol buffer.
func Size(pb Message) (n int) {
	// Can the object marshal itself?  If so, Size is slow.
	// TODO: add Size to Marshaler, or add a Sizer interface.
	if m, ok := pb.(Marshaler); ok {
		b, _ := m.Marshal()
		return len(b)
	}

	t, base, err := getbase(pb)
	if structPointer_IsNil(base) {
		return 0
	}
	if err == nil {
		n = size_struct(GetProperties(t.Elem()), base)
	}

	if collectStats {
		(stats).Size++ // Parens are to work around a goimports bug.
	}

	return
}

// Individual type encoders.

// Encode a bool.
func (o *Buffer) enc_bool(p *Properties, base structPointer) error {
	v := *structPointer_Bool(base, p.field)
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

func (o *Buffer) enc_proto3_bool(p *Properties, base structPointer) error {
	v := *structPointer_BoolVal(base, p.field)
	if !v {
		return ErrNil
	}
	o.buf = append(o.buf, p.tagcode...)
	p.valEnc(o, 1)
	return nil
}

func size_bool(p *Properties, base structPointer) int {
	v := *structPointer_Bool(base, p.field)
	if v == nil {
		return 0
	}
	return len(p.tagcode) + 1 // each bool takes exactly one byte
}

func size_proto3_bool(p *Properties, base structPointer) int {
	v := *structPointer_BoolVal(base, p.field)
	if !v && !p.oneof {
		return 0
	}
	return len(p.tagcode) + 1 // each bool takes exactly one byte
}

// Encode an int32.
func (o *Buffer) enc_int32(p *Properties, base structPointer) error {
	v := structPointer_Word32(base, p.field)
	if word32_IsNil(v) {
		return ErrNil
	}
	x := int32(word32_Get(v)) // permit sign extension to use full 64-bit range
	o.buf = append(o.buf, p.tagcode...)
	p.valEnc(o, uint64(x))
	return nil
}

func (o *Buffer) enc_proto3_int32(p *Properties, base structPointer) error {
	v := structPointer_Word32Val(base, p.field)
	x := int32(word32Val_Get(v)) // permit sign extension to use full 64-bit range
	if x == 0 {
		return ErrNil
	}
	o.buf = append(o.buf, p.tagcode...)
	p.valEnc(o, uint64(x))
	return nil
}

func size_int32(p *Properties, base structPointer) (n int) {
	v := structPointer_Word32(base, p.field)
	if word32_IsNil(v) {
		return 0
	}
	x := int32(word32_Get(v)) // permit sign extension to use full 64-bit range
	n += len(p.tagcode)
	n += p.valSize(uint64(x))
	return
}

func size_proto3_int32(p *Properties, base structPointer) (n int) {
	v := structPointer_Word32Val(base, p.field)
	x := int32(word32Val_Get(v)) // permit sign extension to use full 64-bit range
	if x == 0 && !p.oneof {
		return 0
	}
	n += len(p.tagcode)
	n += p.valSize(uint64(x))
	return
}

// Encode a uint32.
// Exactly the same as int32, except for no sign extension.
func (o *Buffer) enc_uint32(p *Properties, base structPointer) error {
	v := structPointer_Word32(base, p.field)
	if word32_IsNil(v) {
		return ErrNil
	}
	x := word32_Get(v)
	o.buf = append(o.buf, p.tagcode...)
	p.valEnc(o, uint64(x))
	return nil
}

func (o *Buffer) enc_proto3_uint32(p *Properties, base structPointer) error {
	v := structPointer_Word32Val(base, p.field)
	x := word32Val_Get(v)
	if x == 0 {
		return ErrNil
	}
	o.buf = append(o.buf, p.tagcode...)
	p.valEnc(o, uint64(x))
	return nil
}

func size_uint32(p *Properties, base structPointer) (n int) {
	v := structPointer_Word32(base, p.field)
	if word32_IsNil(v) {
		return 0
	}
	x := word32_Get(v)
	n += len(p.tagcode)
	n += p.valSize(uint64(x))
	return
}

func size_proto3_uint32(p *Properties, base structPointer) (n int) {
	v := structPointer_Word32Val(base, p.field)
	x := word32Val_Get(v)
	if x == 0 && !p.oneof {
		return 0
	}
	n += len(p.tagcode)
	n += p.valSize(uint64(x))
	return
}

// Encode an int64.
func (o *Buffer) enc_int64(p *Properties, base structPointer) error {
	v := structPointer_Word64(base, p.field)
	if word64_IsNil(v) {
		return ErrNil
	}
	x := word64_Get(v)
	o.buf = append(o.buf, p.tagcode...)
	p.valEnc(o, x)
	return nil
}

func (o *Buffer) enc_proto3_int64(p *Properties, base structPointer) error {
	v := structPointer_Word64Val(base, p.field)
	x := word64Val_Get(v)
	if x == 0 {
		return ErrNil
	}
	o.buf = append(o.buf, p.tagcode...)
	p.valEnc(o, x)
	return nil
}

func size_int64(p *Properties, base structPointer) (n int) {
	v := structPointer_Word64(base, p.field)
	if word64_IsNil(v) {
		return 0
	}
	x := word64_Get(v)
	n += len(p.tagcode)
	n += p.valSize(x)
	return
}

func size_proto3_int64(p *Properties, base structPointer) (n int) {
	v := structPointer_Word64Val(base, p.field)
	x := word64Val_Get(v)
	if x == 0 && !p.oneof {
		return 0
	}
	n += len(p.tagcode)
	n += p.valSize(x)
	return
}

// Encode a string.
func (o *Buffer) enc_string(p *Properties, base structPointer) error {
	v := *structPointer_String(base, p.field)
	if v == nil {
		return ErrNil
	}
	x := *v
	o.buf = append(o.buf, p.tagcode...)
	o.EncodeStringBytes(x)
	return nil
}

func (o *Buffer) enc_proto3_string(p *Properties, base structPointer) error {
	v := *structPointer_StringVal(base, p.field)
	if v == "" {
		return ErrNil
	}
	o.buf = append(o.buf, p.tagcode...)
	o.EncodeStringBytes(v)
	return nil
}

func size_string(p *Properties, base structPointer) (n int) {
	v := *structPointer_String(base, p.field)
	if v == nil {
		return 0
	}
	x := *v
	n += len(p.tagcode)
	n += sizeStringBytes(x)
	return
}

func size_proto3_string(p *Properties, base structPointer) (n int) {
	v := *structPointer_StringVal(base, p.field)
	if v == "" && !p.oneof {
		return 0
	}
	n += len(p.tagcode)
	n += sizeStringBytes(v)
	return
}

// All protocol buffer fields are nillable, but be careful.
func isNil(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return v.IsNil()
	}
	return false
}

// Encode a message struct.
func (o *Buffer) enc_struct_message(p *Properties, base structPointer) error {
	var state errorState
	structp := structPointer_GetStructPointer(base, p.field)
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
		return state.err
	}

	o.buf = append(o.buf, p.tagcode...)
	return o.enc_len_struct(p.sprop, structp, &state)
}

func size_struct_message(p *Properties, base structPointer) int {
	structp := structPointer_GetStructPointer(base, p.field)
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

// Encode a group struct.
func (o *Buffer) enc_struct_group(p *Properties, base structPointer) error {
	var state errorState
	b := structPointer_GetStructPointer(base, p.field)
	if structPointer_IsNil(b) {
		return ErrNil
	}

	o.EncodeVarint(uint64((p.Tag << 3) | WireStartGroup))
	err := o.enc_struct(p.sprop, b)
	if err != nil && !state.shouldContinue(err, nil) {
		return err
	}
	o.EncodeVarint(uint64((p.Tag << 3) | WireEndGroup))
	return state.err
}

func size_struct_group(p *Properties, base structPointer) (n int) {
	b := structPointer_GetStructPointer(base, p.field)
	if structPointer_IsNil(b) {
		return 0
	}

	n += sizeVarint(uint64((p.Tag << 3) | WireStartGroup))
	n += size_struct(p.sprop, b)
	n += sizeVarint(uint64((p.Tag << 3) | WireEndGroup))
	return
}

// Encode a slice of bools ([]bool).
func (o *Buffer) enc_slice_bool(p *Properties, base structPointer) error {
	s := *structPointer_BoolSlice(base, p.field)
	l := len(s)
	if l == 0 {
		return ErrNil
	}
	for _, x := range s {
		o.buf = append(o.buf, p.tagcode...)
		v := uint64(0)
		if x {
			v = 1
		}
		p.valEnc(o, v)
	}
	return nil
}

func size_slice_bool(p *Properties, base structPointer) int {
	s := *structPointer_BoolSlice(base, p.field)
	l := len(s)
	if l == 0 {
		return 0
	}
	return l * (len(p.tagcode) + 1) // each bool takes exactly one byte
}

// Encode a slice of bools ([]bool) in packed format.
func (o *Buffer) enc_slice_packed_bool(p *Properties, base structPointer) error {
	s := *structPointer_BoolSlice(base, p.field)
	l := len(s)
	if l == 0 {
		return ErrNil
	}
	o.buf = append(o.buf, p.tagcode...)
	o.EncodeVarint(uint64(l)) // each bool takes exactly one byte
	for _, x := range s {
		v := uint64(0)
		if x {
			v = 1
		}
		p.valEnc(o, v)
	}
	return nil
}

func size_slice_packed_bool(p *Properties, base structPointer) (n int) {
	s := *structPointer_BoolSlice(base, p.field)
	l := len(s)
	if l == 0 {
		return 0
	}
	n += len(p.tagcode)
	n += sizeVarint(uint64(l))
	n += l // each bool takes exactly one byte
	return
}

// Encode a slice of bytes ([]byte).
func (o *Buffer) enc_slice_byte(p *Properties, base structPointer) error {
	s := *structPointer_Bytes(base, p.field)
	if s == nil {
		return ErrNil
	}
	o.buf = append(o.buf, p.tagcode...)
	o.EncodeRawBytes(s)
	return nil
}

func (o *Buffer) enc_proto3_slice_byte(p *Properties, base structPointer) error {
	s := *structPointer_Bytes(base, p.field)
	if len(s) == 0 {
		return ErrNil
	}
	o.buf = append(o.buf, p.tagcode...)
	o.EncodeRawBytes(s)
	return nil
}

func size_slice_byte(p *Properties, base structPointer) (n int) {
	s := *structPointer_Bytes(base, p.field)
	if s == nil && !p.oneof {
		return 0
	}
	n += len(p.tagcode)
	n += sizeRawBytes(s)
	return
}

func size_proto3_slice_byte(p *Properties, base structPointer) (n int) {
	s := *structPointer_Bytes(base, p.field)
	if len(s) == 0 && !p.oneof {
		return 0
	}
	n += len(p.tagcode)
	n += sizeRawBytes(s)
	return
}

// Encode a slice of int32s ([]int32).
func (o *Buffer) enc_slice_int32(p *Properties, base structPointer) error {
	s := structPointer_Word32Slice(base, p.field)
	l := s.Len()
	if l == 0 {
		return ErrNil
	}
	for i := 0; i < l; i++ {
		o.buf = append(o.buf, p.tagcode...)
		x := int32(s.Index(i)) // permit sign extension to use full 64-bit range
		p.valEnc(o, uint64(x))
	}
	return nil
}

func size_slice_int32(p *Properties, base structPointer) (n int) {
	s := structPointer_Word32Slice(base, p.field)
	l := s.Len()
	if l == 0 {
		return 0
	}
	for i := 0; i < l; i++ {
		n += len(p.tagcode)
		x := int32(s.Index(i)) // permit sign extension to use full 64-bit range
		n += p.valSize(uint64(x))
	}
	return
}

// Encode a slice of int32s ([]int32) in packed format.
func (o *Buffer) enc_slice_packed_int32(p *Properties, base structPointer) error {
	s := structPointer_Word32Slice(base, p.field)
	l := s.Len()
	if l == 0 {
		return ErrNil
	}
	// TODO: Reuse a Buffer.
	buf := NewBuffer(nil)
	for i := 0; i < l; i++ {
		x := int32(s.Index(i)) // permit sign extension to use full 64-bit range
		p.valEnc(buf, uint64(x))
	}

	o.buf = append(o.buf, p.tagcode...)
	o.EncodeVarint(uint64(len(buf.buf)))
	o.buf = append(o.buf, buf.buf...)
	return nil
}

func size_slice_packed_int32(p *Properties, base structPointer) (n int) {
	s := structPointer_Word32Slice(base, p.field)
	l := s.Len()
	if l == 0 {
		return 0
	}
	var bufSize int
	for i := 0; i < l; i++ {
		x := int32(s.Index(i)) // permit sign extension to use full 64-bit range
		bufSize += p.valSize(uint64(x))
	}

	n += len(p.tagcode)
	n += sizeVarint(uint64(bufSize))
	n += bufSize
	return
}

// Encode a slice of uint32s ([]uint32).
// Exactly the same as int32, except for no sign extension.
func (o *Buffer) enc_slice_uint32(p *Properties, base structPointer) error {
	s := structPointer_Word32Slice(base, p.field)
	l := s.Len()
	if l == 0 {
		return ErrNil
	}
	for i := 0; i < l; i++ {
		o.buf = append(o.buf, p.tagcode...)
		x := s.Index(i)
		p.valEnc(o, uint64(x))
	}
	return nil
}

func size_slice_uint32(p *Properties, base structPointer) (n int) {
	s := structPointer_Word32Slice(base, p.field)
	l := s.Len()
	if l == 0 {
		return 0
	}
	for i := 0; i < l; i++ {
		n += len(p.tagcode)
		x := s.Index(i)
		n += p.valSize(uint64(x))
	}
	return
}

// Encode a slice of uint32s ([]uint32) in packed format.
// Exactly the same as int32, except for no sign extension.
func (o *Buffer) enc_slice_packed_uint32(p *Properties, base structPointer) error {
	s := structPointer_Word32Slice(base, p.field)
	l := s.Len()
	if l == 0 {
		return ErrNil
	}
	// TODO: Reuse a Buffer.
	buf := NewBuffer(nil)
	for i := 0; i < l; i++ {
		p.valEnc(buf, uint64(s.Index(i)))
	}

	o.buf = append(o.buf, p.tagcode...)
	o.EncodeVarint(uint64(len(buf.buf)))
	o.buf = append(o.buf, buf.buf...)
	return nil
}

func size_slice_packed_uint32(p *Properties, base structPointer) (n int) {
	s := structPointer_Word32Slice(base, p.field)
	l := s.Len()
	if l == 0 {
		return 0
	}
	var bufSize int
	for i := 0; i < l; i++ {
		bufSize += p.valSize(uint64(s.Index(i)))
	}

	n += len(p.tagcode)
	n += sizeVarint(uint64(bufSize))
	n += bufSize
	return
}

// Encode a slice of int64s ([]int64).
func (o *Buffer) enc_slice_int64(p *Properties, base structPointer) error {
	s := structPointer_Word64Slice(base, p.field)
	l := s.Len()
	if l == 0 {
		return ErrNil
	}
	for i := 0; i < l; i++ {
		o.buf = append(o.buf, p.tagcode...)
		p.valEnc(o, s.Index(i))
	}
	return nil
}

func size_slice_int64(p *Properties, base structPointer) (n int) {
	s := structPointer_Word64Slice(base, p.field)
	l := s.Len()
	if l == 0 {
		return 0
	}
	for i := 0; i < l; i++ {
		n += len(p.tagcode)
		n += p.valSize(s.Index(i))
	}
	return
}

// Encode a slice of int64s ([]int64) in packed format.
func (o *Buffer) enc_slice_packed_int64(p *Properties, base structPointer) error {
	s := structPointer_Word64Slice(base, p.field)
	l := s.Len()
	if l == 0 {
		return ErrNil
	}
	// TODO: Reuse a Buffer.
	buf := NewBuffer(nil)
	for i := 0; i < l; i++ {
		p.valEnc(buf, s.Index(i))
	}

	o.buf = append(o.buf, p.tagcode...)
	o.EncodeVarint(uint64(len(buf.buf)))
	o.buf = append(o.buf, buf.buf...)
	return nil
}

func size_slice_packed_int64(p *Properties, base structPointer) (n int) {
	s := structPointer_Word64Slice(base, p.field)
	l := s.Len()
	if l == 0 {
		return 0
	}
	var bufSize int
	for i := 0; i < l; i++ {
		bufSize += p.valSize(s.Index(i))
	}

	n += len(p.tagcode)
	n += sizeVarint(uint64(bufSize))
	n += bufSize
	return
}

// Encode a slice of slice of bytes ([][]byte).
func (o *Buffer) enc_slice_slice_byte(p *Properties, base structPointer) error {
	ss := *structPointer_BytesSlice(base, p.field)
	l := len(ss)
	if l == 0 {
		return ErrNil
	}
	for i := 0; i < l; i++ {
		o.buf = append(o.buf, p.tagcode...)
		o.EncodeRawBytes(ss[i])
	}
	return nil
}

func size_slice_slice_byte(p *Properties, base structPointer) (n int) {
	ss := *structPointer_BytesSlice(base, p.field)
	l := len(ss)
	if l == 0 {
		return 0
	}
	n += l * len(p.tagcode)
	for i := 0; i < l; i++ {
		n += sizeRawBytes(ss[i])
	}
	return
}

// Encode a slice of strings ([]string).
func (o *Buffer) enc_slice_string(p *Properties, base structPointer) error {
	ss := *structPointer_StringSlice(base, p.field)
	l := len(ss)
	for i := 0; i < l; i++ {
		o.buf = append(o.buf, p.tagcode...)
		o.EncodeStringBytes(ss[i])
	}
	return nil
}

func size_slice_string(p *Properties, base structPointer) (n int) {
	ss := *structPointer_StringSlice(base, p.field)
	l := len(ss)
	n += l * len(p.tagcode)
	for i := 0; i < l; i++ {
		n += sizeStringBytes(ss[i])
	}
	return
}

// Encode a slice of message structs ([]*struct).
func (o *Buffer) enc_slice_struct_message(p *Properties, base structPointer) error {
	var state errorState
	s := structPointer_StructPointerSlice(base, p.field)
	l := s.Len()

	for i := 0; i < l; i++ {
		structp := s.Index(i)
		if structPointer_IsNil(structp) {
			return errRepeatedHasNil
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
				return errRepeatedHasNil
			}
			return err
		}
	}
	return state.err
}

func size_slice_struct_message(p *Properties, base structPointer) (n int) {
	s := structPointer_StructPointerSlice(base, p.field)
	l := s.Len()
	n += l * len(p.tagcode)
	for i := 0; i < l; i++ {
		structp := s.Index(i)
		if structPointer_IsNil(structp) {
			return // return the size up to this point
		}

		// Can the object marshal itself?
		if p.isMarshaler {
			m := structPointer_Interface(structp, p.stype).(Marshaler)
			data, _ := m.Marshal()
			n += sizeRawBytes(data)
			continue
		}

		n0 := size_struct(p.sprop, structp)
		n1 := sizeVarint(uint64(n0)) // size of encoded length
		n += n0 + n1
	}
	return
}

// Encode a slice of group structs ([]*struct).
func (o *Buffer) enc_slice_struct_group(p *Properties, base structPointer) error {
	var state errorState
	s := structPointer_StructPointerSlice(base, p.field)
	l := s.Len()

	for i := 0; i < l; i++ {
		b := s.Index(i)
		if structPointer_IsNil(b) {
			return errRepeatedHasNil
		}

		o.EncodeVarint(uint64((p.Tag << 3) | WireStartGroup))

		err := o.enc_struct(p.sprop, b)

		if err != nil && !state.shouldContinue(err, nil) {
			if err == ErrNil {
				return errRepeatedHasNil
			}
			return err
		}

		o.EncodeVarint(uint64((p.Tag << 3) | WireEndGroup))
	}
	return state.err
}

func size_slice_struct_group(p *Properties, base structPointer) (n int) {
	s := structPointer_StructPointerSlice(base, p.field)
	l := s.Len()

	n += l * sizeVarint(uint64((p.Tag<<3)|WireStartGroup))
	n += l * sizeVarint(uint64((p.Tag<<3)|WireEndGroup))
	for i := 0; i < l; i++ {
		b := s.Index(i)
		if structPointer_IsNil(b) {
			return // return size up to this point
		}

		n += size_struct(p.sprop, b)
	}
	return
}

// Encode an extension map.
func (o *Buffer) enc_map(p *Properties, base structPointer) error {
	exts := structPointer_ExtMap(base, p.field)
	if err := encodeExtensionsMap(*exts); err != nil {
		return err
	}

	return o.enc_map_body(*exts)
}

func (o *Buffer) enc_exts(p *Properties, base structPointer) error {
	exts := structPointer_Extensions(base, p.field)

	v, mu := exts.extensionsRead()
	if v == nil {
		return nil
	}

	mu.Lock()
	defer mu.Unlock()
	if err := encodeExtensionsMap(v); err != nil {
		return err
	}

	return o.enc_map_body(v)
}

func (o *Buffer) enc_map_body(v map[int32]Extension) error {
	// Fast-path for common cases: zero or one extensions.
	if len(v) <= 1 {
		for _, e := range v {
			o.buf = append(o.buf, e.enc...)
		}
		return nil
	}

	// Sort keys to provide a deterministic encoding.
	keys := make([]int, 0, len(v))
	for k := range v {
		keys = append(keys, int(k))
	}
	sort.Ints(keys)

	for _, k := range keys {
		o.buf = append(o.buf, v[int32(k)].enc...)
	}
	return nil
}

func size_map(p *Properties, base structPointer) int {
	v := structPointer_ExtMap(base, p.field)
	return extensionsMapSize(*v)
}

func size_exts(p *Properties, base structPointer) int {
	v := structPointer_Extensions(base, p.field)
	return extensionsSize(v)
}

// Encode a map field.
func (o *Buffer) enc_new_map(p *Properties, base structPointer) error {
	var state errorState // XXX: or do we need to plumb this through?

	/*
		A map defined as
			map<key_type, value_type> map_field = N;
		is encoded in the same way as
			message MapFieldEntry {
				key_type key = 1;
				value_type value = 2;
			}
			repeated MapFieldEntry map_field = N;
	*/

	v := structPointer_NewAt(base, p.field, p.mtype).Elem() // map[K]V
	if v.Len() == 0 {
		return nil
	}

	keycopy, valcopy, keybase, valbase := mapEncodeScratch(p.mtype)

	enc := func() error {
		if err := p.mkeyprop.enc(o, p.mkeyprop, keybase); err != nil {
			return err
		}
		if err := p.mvalprop.enc(o, p.mvalprop, valbase); err != nil && err != ErrNil {
			return err
		}
		return nil
	}

	// Don't sort map keys. It is not required by the spec, and C++ doesn't do it.
	for _, key := range v.MapKeys() {
		val := v.MapIndex(key)

		keycopy.Set(key)
		valcopy.Set(val)

		o.buf = append(o.buf, p.tagcode...)
		if err := o.enc_len_thing(enc, &state); err != nil {
			return err
		}
	}
	return nil
}

func size_new_map(p *Properties, base structPointer) int {
	v := structPointer_NewAt(base, p.field, p.mtype).Elem() // map[K]V

	keycopy, valcopy, keybase, valbase := mapEncodeScratch(p.mtype)

	n := 0
	for _, key := range v.MapKeys() {
		val := v.MapIndex(key)
		keycopy.Set(key)
		valcopy.Set(val)

		// Tag codes for key and val are the responsibility of the sub-sizer.
		keysize := p.mkeyprop.size(p.mkeyprop, keybase)
		valsize := p.mvalprop.size(p.mvalprop, valbase)
		entry := keysize + valsize
		// Add on tag code and length of map entry itself.
		n += len(p.tagcode) + sizeVarint(uint64(entry)) + entry
	}
	return n
}

// mapEncodeScratch returns a new reflect.Value matching the map's value type,
// and a structPointer suitable for passing to an encoder or sizer.
func mapEncodeScratch(mapType reflect.Type) (keycopy, valcopy reflect.Value, keybase, valbase structPointer) {
	// Prepare addressable doubly-indirect placeholders for the key and value types.
	// This is needed because the element-type encoders expect **T, but the map iteration produces T.

	keycopy = reflect.New(mapType.Key()).Elem()                 // addressable K
	keyptr := reflect.New(reflect.PtrTo(keycopy.Type())).Elem() // addressable *K
	keyptr.Set(keycopy.Addr())                                  //
	keybase = toStructPointer(keyptr.Addr())                    // **K

	// Value types are more varied and require special handling.
	switch mapType.Elem().Kind() {
	case reflect.Slice:
		// []byte
		var dummy []byte
		valcopy = reflect.ValueOf(&dummy).Elem() // addressable []byte
		valbase = toStructPointer(valcopy.Addr())
	case reflect.Ptr:
		// message; the generated field type is map[K]*Msg (so V is *Msg),
		// so we only need one level of indirection.
		valcopy = reflect.New(mapType.Elem()).Elem() // addressable V
		valbase = toStructPointer(valcopy.Addr())
	default:
		// everything else
		valcopy = reflect.New(mapType.Elem()).Elem()                // addressable V
		valptr := reflect.New(reflect.PtrTo(valcopy.Type())).Elem() // addressable *V
		valptr.Set(valcopy.Addr())                                  //
		valbase = toStructPointer(valptr.Addr())                    // **V
	}
	return
}

// Encode a struct.
func (o *Buffer) enc_struct(prop *StructProperties, base structPointer) error {
	var state errorState
	// Encode fields in tag order so that decoders may use optimizations
	// that depend on the ordering.
	// https://developers.google.com/protocol-buffers/docs/encoding#order
	for _, i := range prop.order {
		p := prop.Prop[i]
		if p.enc != nil {
			err := p.enc(o, p, base)
			if err != nil {
				if err == ErrNil {
					if p.Required && state.err == nil {
						state.err = &RequiredNotSetError{p.Name}
					}
				} else if err == errRepeatedHasNil {
					// Give more context to nil values in repeated fields.
					return errors.New("repeated field " + p.OrigName + " has nil element")
				} else if !state.shouldContinue(err, p) {
					return err
				}
			}
			if len(o.buf) > maxMarshalSize {
				return ErrTooLarge
			}
		}
	}

	// Do oneof fields.
	if prop.oneofMarshaler != nil {
		m := structPointer_Interface(base, prop.stype).(Message)
		if err := prop.oneofMarshaler(m, o); err == ErrNil {
			return errOneofHasNil
		} else if err != nil {
			return err
		}
	}

	// Add unrecognized fields at the end.
	if prop.unrecField.IsValid() {
		v := *structPointer_Bytes(base, prop.unrecField)
		if len(o.buf)+len(v) > maxMarshalSize {
			return ErrTooLarge
		}
		if len(v) > 0 {
			o.buf = append(o.buf, v...)
		}
	}

	return state.err
}

func size_struct(prop *StructProperties, base structPointer) (n int) {
	for _, i := range prop.order {
		p := prop.Prop[i]
		if p.size != nil {
			n += p.size(p, base)
		}
	}

	// Add unrecognized fields at the end.
	if prop.unrecField.IsValid() {
		v := *structPointer_Bytes(base, prop.unrecField)
		n += len(v)
	}

	// Factor in any oneof fields.
	if prop.oneofSizer != nil {
		m := structPointer_Interface(base, prop.stype).(Message)
		n += prop.oneofSizer(m)
	}

	return
}

var zeroes [20]byte // longer than any conceivable sizeVarint

// Encode a struct, preceded by its encoded length (as a varint).
func (o *Buffer) enc_len_struct(prop *StructProperties, base structPointer, state *errorState) error {
	return o.enc_len_thing(func() error { return o.enc_struct(prop, base) }, state)
}

// Encode something, preceded by its encoded length (as a varint).
func (o *Buffer) enc_len_thing(enc func() error, state *errorState) error {
	iLen := len(o.buf)
	o.buf = append(o.buf, 0, 0, 0, 0) // reserve four bytes for length
	iMsg := len(o.buf)
	err := enc()
	if err != nil && !state.shouldContinue(err, nil) {
		return err
	}
	lMsg := len(o.buf) - iMsg
	lLen := sizeVarint(uint64(lMsg))
	switch x := lLen - (iMsg - iLen); {
	case x > 0: // actual length is x bytes larger than the space we reserved
		// Move msg x bytes right.
		o.buf = append(o.buf, zeroes[:x]...)
		copy(o.buf[iMsg+x:], o.buf[iMsg:iMsg+lMsg])
	case x < 0: // actual length is x bytes smaller than the space we reserved
		// Move msg x bytes left.
		copy(o.buf[iMsg+x:], o.buf[iMsg:iMsg+lMsg])
		o.buf = o.buf[:len(o.buf)+x] // x is negative
	}
	// Encode the length in the reserved space.
	o.buf = o.buf[:iLen]
	o.EncodeVarint(uint64(lMsg))
	o.buf = o.buf[:len(o.buf)+lMsg]
	return state.err
}

// errorState maintains the first error that occurs and updates that error
// with additional context.
type errorState struct {
	err error
}

// shouldContinue reports whether encoding should continue upon encountering the
// given error. If the error is RequiredNotSetError, shouldContinue returns true
// and, if this is the first appearance of that error, remembers it for future
// reporting.
//
// If prop is not nil, it may update any error with additional context about the
// field with the error.
func (s *errorState) shouldContinue(err error, prop *Properties) bool {
	// Ignore unset required fields.
	reqNotSet, ok := err.(*RequiredNotSetError)
	if !ok {
		return false
	}
	if s.err == nil {
		if prop != nil {
			err = &RequiredNotSetError{prop.Name + "." + reqNotSet.field}
		}
		s.err = err
	}
	return true
}
