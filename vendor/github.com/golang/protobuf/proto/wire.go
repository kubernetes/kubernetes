// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	protoV2 "google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/runtime/protoiface"
)

// Size returns the size in bytes of the wire-format encoding of m.
func Size(m Message) int {
	if m == nil {
		return 0
	}
	mi := MessageV2(m)
	return protoV2.Size(mi)
}

// Marshal returns the wire-format encoding of m.
func Marshal(m Message) ([]byte, error) {
	b, err := marshalAppend(nil, m, false)
	if b == nil {
		b = zeroBytes
	}
	return b, err
}

var zeroBytes = make([]byte, 0, 0)

func marshalAppend(buf []byte, m Message, deterministic bool) ([]byte, error) {
	if m == nil {
		return nil, ErrNil
	}
	mi := MessageV2(m)
	nbuf, err := protoV2.MarshalOptions{
		Deterministic: deterministic,
		AllowPartial:  true,
	}.MarshalAppend(buf, mi)
	if err != nil {
		return buf, err
	}
	if len(buf) == len(nbuf) {
		if !mi.ProtoReflect().IsValid() {
			return buf, ErrNil
		}
	}
	return nbuf, checkRequiredNotSet(mi)
}

// Unmarshal parses a wire-format message in b and places the decoded results in m.
//
// Unmarshal resets m before starting to unmarshal, so any existing data in m is always
// removed. Use UnmarshalMerge to preserve and append to existing data.
func Unmarshal(b []byte, m Message) error {
	m.Reset()
	return UnmarshalMerge(b, m)
}

// UnmarshalMerge parses a wire-format message in b and places the decoded results in m.
func UnmarshalMerge(b []byte, m Message) error {
	mi := MessageV2(m)
	out, err := protoV2.UnmarshalOptions{
		AllowPartial: true,
		Merge:        true,
	}.UnmarshalState(protoiface.UnmarshalInput{
		Buf:     b,
		Message: mi.ProtoReflect(),
	})
	if err != nil {
		return err
	}
	if out.Flags&protoiface.UnmarshalInitialized > 0 {
		return nil
	}
	return checkRequiredNotSet(mi)
}
