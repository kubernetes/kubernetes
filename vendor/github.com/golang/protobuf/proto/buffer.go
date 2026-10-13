// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"errors"
	"fmt"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/runtime/protoimpl"
)

const (
	WireVarint     = 0
	WireFixed32    = 5
	WireFixed64    = 1
	WireBytes      = 2
	WireStartGroup = 3
	WireEndGroup   = 4
)

// EncodeVarint returns the varint encoded bytes of v.
func EncodeVarint(v uint64) []byte {
	return protowire.AppendVarint(nil, v)
}

// SizeVarint returns the length of the varint encoded bytes of v.
// This is equal to len(EncodeVarint(v)).
func SizeVarint(v uint64) int {
	return protowire.SizeVarint(v)
}

// DecodeVarint parses a varint encoded integer from b,
// returning the integer value and the length of the varint.
// It returns (0, 0) if there is a parse error.
func DecodeVarint(b []byte) (uint64, int) {
	v, n := protowire.ConsumeVarint(b)
	if n < 0 {
		return 0, 0
	}
	return v, n
}

// Buffer is a buffer for encoding and decoding the protobuf wire format.
// It may be reused between invocations to reduce memory usage.
type Buffer struct {
	buf           []byte
	idx           int
	deterministic bool
}

// NewBuffer allocates a new Buffer initialized with buf,
// where the contents of buf are considered the unread portion of the buffer.
func NewBuffer(buf []byte) *Buffer {
	return &Buffer{buf: buf}
}

// SetDeterministic specifies whether to use deterministic serialization.
//
// Deterministic serialization guarantees that for a given binary, equal
// messages will always be serialized to the same bytes. This implies:
//
//   - Repeated serialization of a message will return the same bytes.
//   - Different processes of the same binary (which may be executing on
//     different machines) will serialize equal messages to the same bytes.
//
// Note that the deterministic serialization is NOT canonical across
// languages. It is not guaranteed to remain stable over time. It is unstable
// across different builds with schema changes due to unknown fields.
// Users who need canonical serialization (e.g., persistent storage in a
// canonical form, fingerprinting, etc.) should define their own
// canonicalization specification and implement their own serializer rather
// than relying on this API.
//
// If deterministic serialization is requested, map entries will be sorted
// by keys in lexographical order. This is an implementation detail and
// subject to change.
func (b *Buffer) SetDeterministic(deterministic bool) {
	b.deterministic = deterministic
}

// SetBuf sets buf as the internal buffer,
// where the contents of buf are considered the unread portion of the buffer.
func (b *Buffer) SetBuf(buf []byte) {
	b.buf = buf
	b.idx = 0
}

// Reset clears the internal buffer of all written and unread data.
func (b *Buffer) Reset() {
	b.buf = b.buf[:0]
	b.idx = 0
}

// Bytes returns the internal buffer.
func (b *Buffer) Bytes() []byte {
	return b.buf
}

// Unread returns the unread portion of the buffer.
func (b *Buffer) Unread() []byte {
	return b.buf[b.idx:]
}

// Marshal appends the wire-format encoding of m to the buffer.
func (b *Buffer) Marshal(m Message) error {
	var err error
	b.buf, err = marshalAppend(b.buf, m, b.deterministic)
	return err
}

// Unmarshal parses the wire-format message in the buffer and
// places the decoded results in m.
// It does not reset m before unmarshaling.
func (b *Buffer) Unmarshal(m Message) error {
	err := UnmarshalMerge(b.Unread(), m)
	b.idx = len(b.buf)
	return err
}

type unknownFields struct{ XXX_unrecognized protoimpl.UnknownFields }

func (m *unknownFields) String() string { panic("not implemented") }
func (m *unknownFields) Reset()         { panic("not implemented") }
func (m *unknownFields) ProtoMessage()  { panic("not implemented") }

// DebugPrint dumps the encoded bytes of b with a header and footer including s
// to stdout. This is only intended for debugging.
func (*Buffer) DebugPrint(s string, b []byte) {
	m := MessageReflect(new(unknownFields))
	m.SetUnknown(b)
	b, _ = prototext.MarshalOptions{AllowPartial: true, Indent: "\t"}.Marshal(m.Interface())
	fmt.Printf("==== %s ====\n%s==== %s ====\n", s, b, s)
}

// EncodeVarint appends an unsigned varint encoding to the buffer.
func (b *Buffer) EncodeVarint(v uint64) error {
	b.buf = protowire.AppendVarint(b.buf, v)
	return nil
}

// EncodeZigzag32 appends a 32-bit zig-zag varint encoding to the buffer.
func (b *Buffer) EncodeZigzag32(v uint64) error {
	return b.EncodeVarint(uint64((uint32(v) << 1) ^ uint32((int32(v) >> 31))))
}

// EncodeZigzag64 appends a 64-bit zig-zag varint encoding to the buffer.
func (b *Buffer) EncodeZigzag64(v uint64) error {
	return b.EncodeVarint(uint64((uint64(v) << 1) ^ uint64((int64(v) >> 63))))
}

// EncodeFixed32 appends a 32-bit little-endian integer to the buffer.
func (b *Buffer) EncodeFixed32(v uint64) error {
	b.buf = protowire.AppendFixed32(b.buf, uint32(v))
	return nil
}

// EncodeFixed64 appends a 64-bit little-endian integer to the buffer.
func (b *Buffer) EncodeFixed64(v uint64) error {
	b.buf = protowire.AppendFixed64(b.buf, uint64(v))
	return nil
}

// EncodeRawBytes appends a length-prefixed raw bytes to the buffer.
func (b *Buffer) EncodeRawBytes(v []byte) error {
	b.buf = protowire.AppendBytes(b.buf, v)
	return nil
}

// EncodeStringBytes appends a length-prefixed raw bytes to the buffer.
// It does not validate whether v contains valid UTF-8.
func (b *Buffer) EncodeStringBytes(v string) error {
	b.buf = protowire.AppendString(b.buf, v)
	return nil
}

// EncodeMessage appends a length-prefixed encoded message to the buffer.
func (b *Buffer) EncodeMessage(m Message) error {
	var err error
	b.buf = protowire.AppendVarint(b.buf, uint64(Size(m)))
	b.buf, err = marshalAppend(b.buf, m, b.deterministic)
	return err
}

// DecodeVarint consumes an encoded unsigned varint from the buffer.
func (b *Buffer) DecodeVarint() (uint64, error) {
	v, n := protowire.ConsumeVarint(b.buf[b.idx:])
	if n < 0 {
		return 0, protowire.ParseError(n)
	}
	b.idx += n
	return uint64(v), nil
}

// DecodeZigzag32 consumes an encoded 32-bit zig-zag varint from the buffer.
func (b *Buffer) DecodeZigzag32() (uint64, error) {
	v, err := b.DecodeVarint()
	if err != nil {
		return 0, err
	}
	return uint64((uint32(v) >> 1) ^ uint32((int32(v&1)<<31)>>31)), nil
}

// DecodeZigzag64 consumes an encoded 64-bit zig-zag varint from the buffer.
func (b *Buffer) DecodeZigzag64() (uint64, error) {
	v, err := b.DecodeVarint()
	if err != nil {
		return 0, err
	}
	return uint64((uint64(v) >> 1) ^ uint64((int64(v&1)<<63)>>63)), nil
}

// DecodeFixed32 consumes a 32-bit little-endian integer from the buffer.
func (b *Buffer) DecodeFixed32() (uint64, error) {
	v, n := protowire.ConsumeFixed32(b.buf[b.idx:])
	if n < 0 {
		return 0, protowire.ParseError(n)
	}
	b.idx += n
	return uint64(v), nil
}

// DecodeFixed64 consumes a 64-bit little-endian integer from the buffer.
func (b *Buffer) DecodeFixed64() (uint64, error) {
	v, n := protowire.ConsumeFixed64(b.buf[b.idx:])
	if n < 0 {
		return 0, protowire.ParseError(n)
	}
	b.idx += n
	return uint64(v), nil
}

// DecodeRawBytes consumes a length-prefixed raw bytes from the buffer.
// If alloc is specified, it returns a copy the raw bytes
// rather than a sub-slice of the buffer.
func (b *Buffer) DecodeRawBytes(alloc bool) ([]byte, error) {
	v, n := protowire.ConsumeBytes(b.buf[b.idx:])
	if n < 0 {
		return nil, protowire.ParseError(n)
	}
	b.idx += n
	if alloc {
		v = append([]byte(nil), v...)
	}
	return v, nil
}

// DecodeStringBytes consumes a length-prefixed raw bytes from the buffer.
// It does not validate whether the raw bytes contain valid UTF-8.
func (b *Buffer) DecodeStringBytes() (string, error) {
	v, n := protowire.ConsumeString(b.buf[b.idx:])
	if n < 0 {
		return "", protowire.ParseError(n)
	}
	b.idx += n
	return v, nil
}

// DecodeMessage consumes a length-prefixed message from the buffer.
// It does not reset m before unmarshaling.
func (b *Buffer) DecodeMessage(m Message) error {
	v, err := b.DecodeRawBytes(false)
	if err != nil {
		return err
	}
	return UnmarshalMerge(v, m)
}

// DecodeGroup consumes a message group from the buffer.
// It assumes that the start group marker has already been consumed and
// consumes all bytes until (and including the end group marker).
// It does not reset m before unmarshaling.
func (b *Buffer) DecodeGroup(m Message) error {
	v, n, err := consumeGroup(b.buf[b.idx:])
	if err != nil {
		return err
	}
	b.idx += n
	return UnmarshalMerge(v, m)
}

// consumeGroup parses b until it finds an end group marker, returning
// the raw bytes of the message (excluding the end group marker) and the
// the total length of the message (including the end group marker).
func consumeGroup(b []byte) ([]byte, int, error) {
	b0 := b
	depth := 1 // assume this follows a start group marker
	for {
		_, wtyp, tagLen := protowire.ConsumeTag(b)
		if tagLen < 0 {
			return nil, 0, protowire.ParseError(tagLen)
		}
		b = b[tagLen:]

		var valLen int
		switch wtyp {
		case protowire.VarintType:
			_, valLen = protowire.ConsumeVarint(b)
		case protowire.Fixed32Type:
			_, valLen = protowire.ConsumeFixed32(b)
		case protowire.Fixed64Type:
			_, valLen = protowire.ConsumeFixed64(b)
		case protowire.BytesType:
			_, valLen = protowire.ConsumeBytes(b)
		case protowire.StartGroupType:
			depth++
		case protowire.EndGroupType:
			depth--
		default:
			return nil, 0, errors.New("proto: cannot parse reserved wire type")
		}
		if valLen < 0 {
			return nil, 0, protowire.ParseError(valLen)
		}
		b = b[valLen:]

		if depth == 0 {
			return b0[:len(b0)-len(b)-tagLen], len(b0) - len(b), nil
		}
	}
}
